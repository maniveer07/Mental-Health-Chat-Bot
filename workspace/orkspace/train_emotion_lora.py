"""
Train Emotion Detection Model with LoRA
Efficient fine-tuning with fewer parameters and faster training
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader_multi import MultiDatasetLoader
from src.emotion_detector import EmotionDetector
from src.trainer import EmotionTrainer


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Configure logging so trainer logs appear in terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("EMOTION DETECTION WITH LORA - MULTI-DATASET TRAINING")
    print("=" * 80)
    print()

    # Load configuration
    print("Loading configuration...")
    config = load_config()

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
    print()

    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/emotion_lora", exist_ok=True)
    os.makedirs("models/checkpoints_lora", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Initialize multi-dataset loader
    print("Initializing multi-dataset loader...")
    print("Datasets to load:")
    print("  1. GoEmotions (58K Reddit comments, 28 emotions)")
    print("  2. Emotion (20K tweets, 6 emotions)")
    print("  3. EmpatheticDialogues (25K conversations, 32 emotions)")
    print("  4. DailyDialog (13K conversations, 7 emotions)")
    print()
    print("This will download ~150MB on first run...")
    print()

    data_loader = MultiDatasetLoader(config)

    try:
        train_data, val_data, test_data = data_loader.combine_datasets(
            datasets_to_use=[
                "goemotions",
                "emotion",
                "empathetic_dialogues",
                "dailydialog",
            ],
            use_cache=True,
        )

        print(f"\n✓ All datasets loaded and combined successfully!")
        print(f"  Training samples: {len(train_data):,}")
        print(f"  Validation samples: {len(val_data):,}")
        print(f"  Test samples: {len(test_data):,}")
        print()

        # Apply hierarchical emotion grouping for higher accuracy
        hierarchy_mode = config["model"]["emotion_detection"].get("hierarchy_mode", "basic")
        print(f"Applying hierarchical emotion grouping (mode: {hierarchy_mode})...")

        if hierarchy_mode != "fine":
            train_data, id_to_emotion, emotion_to_id = data_loader.apply_hierarchical_grouping(
                train_data, mode=hierarchy_mode
            )
            val_data, _, _ = data_loader.apply_hierarchical_grouping(
                val_data, mode=hierarchy_mode
            )
            test_data, _, _ = data_loader.apply_hierarchical_grouping(
                test_data, mode=hierarchy_mode
            )

            # Update num_labels from config
            num_labels = len(id_to_emotion)
            config["model"]["emotion_detection"]["num_labels"] = num_labels

            print(f"\n✓ Hierarchical grouping applied!")
            print(f"  Classes: {num_labels}")
            print(f"  Target accuracy: {'95%' if hierarchy_mode == 'basic' else '87%' if hierarchy_mode == 'grouped' else '70%'}")
            print()
        else:
            print("Using fine-grained 44-class taxonomy (no grouping)")
            id_to_emotion = data_loader.id_to_emotion
            emotion_to_id = data_loader.emotion_to_id
            num_labels = 44
            print()

    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try running again (datasets will be cached)")
        print("  3. Check disk space (need ~2GB)")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Initialize model
    emotion_config = config["model"]["emotion_detection"]
    print("Initializing backbone model...")
    emotion_detector = EmotionDetector(
        model_name=emotion_config["model_name"],
        num_labels=num_labels,
        max_length=emotion_config["max_length"],
        device=config["training"]["device"],
    )

    # Update emotion mappings from hierarchical grouping
    emotion_detector.id_to_emotion = id_to_emotion
    emotion_detector.emotion_to_id = emotion_to_id

    print(f"Model initialized: {emotion_config['model_name']}")
    print(f"Num labels: {num_labels}")
    print(f"Emotions: {list(id_to_emotion.values())}")
    print()

    # Decide LoRA target modules based on model type
    model_type = getattr(emotion_detector.model.config, "model_type", "")
    print(f"Detected model_type: {model_type}")

    if model_type == "distilbert":
        # DistilBERT naming
        lora_target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"]
    elif model_type in ["bert", "roberta", "camembert", "albert", "xlnet"]:
        # Standard transformer attention naming
        lora_target_modules = ["query", "key", "value", "dense"]
    else:
        # Fallback
        print(
            "Warning: Unknown model_type for LoRA target modules, "
            "using generic ['query', 'key', 'value', 'dense']."
        )
        lora_target_modules = ["query", "key", "value", "dense"]

    # Apply LoRA configuration with optimized settings
    print("Applying LoRA adapters...")
    lora_rank = emotion_config.get("lora_rank", 64)
    lora_alpha = emotion_config.get("lora_alpha", 128)
    lora_dropout = emotion_config.get("lora_dropout", 0.15)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_rank,  # Increased capacity for better accuracy
        lora_alpha=lora_alpha,  # Scaled accordingly
        lora_dropout=lora_dropout,  # Increased for regularization
        target_modules=lora_target_modules,
        bias="none",
        modules_to_save=["classifier"],  # Ensure classification head is kept
    )

    # Wrap model with LoRA
    emotion_detector.model = get_peft_model(emotion_detector.model, lora_config)
    emotion_detector.model.print_trainable_parameters()
    print()

    # Create data loaders
    print("Creating data loaders...")
    batch_size = emotion_config["batch_size"]

    from torch.utils.data import DataLoader as TorchDataLoader, WeightedRandomSampler
    from src.emotion_detector import EmotionDataset
    from collections import Counter

    train_dataset = EmotionDataset(
        train_data["text"].tolist(),
        train_data["label"].tolist(),
        emotion_detector.tokenizer,
        emotion_config["max_length"],
    )

    val_dataset = EmotionDataset(
        val_data["text"].tolist(),
        val_data["label"].tolist(),
        emotion_detector.tokenizer,
        emotion_config["max_length"],
    )

    test_dataset = EmotionDataset(
        test_data["text"].tolist(),
        test_data["label"].tolist(),
        emotion_detector.tokenizer,
        emotion_config["max_length"],
    )

    # Compute class weights for balanced sampling
    if emotion_config.get("use_class_weights", False):
        print("Computing class weights for balanced sampling...")
        label_counts = Counter(train_data["label"].tolist())
        total_samples = len(train_data)
        num_classes = emotion_config["num_labels"]

        # Compute weights: inverse frequency
        class_weights = torch.zeros(num_classes)
        for label_id in range(num_classes):
            count = label_counts.get(label_id, 1)  # Avoid division by zero
            class_weights[label_id] = total_samples / (num_classes * count)

        # Assign weight to each sample
        sample_weights = torch.tensor(
            [class_weights[label] for label in train_data["label"].tolist()]
        )

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_dataloader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
        print("  ✓ Weighted sampling enabled (class balance improved)")
    else:
        train_dataloader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    val_dataloader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = TorchDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Data loaders created")
    print(f"  Training batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(val_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")
    print()

    # Initialize trainer
    print("Initializing trainer...")
    trainer = EmotionTrainer(
        emotion_detector,
        config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        checkpoint_dir="models/checkpoints_lora",
    )
    print("Trainer ready")
    print()

    # Training configuration
    model_config = config["model"]["emotion_detection"]
    gradient_accum = config["training"].get("gradient_accumulation_steps", 1)
    effective_batch = model_config['batch_size'] * gradient_accum

    print("OPTIMIZED Training Configuration for 90%+ Accuracy:")
    print(f"  Hierarchy Mode: {hierarchy_mode.upper()} ({num_labels} classes)")
    print(f"  Target Accuracy: {'95%' if hierarchy_mode == 'basic' else '87%' if hierarchy_mode == 'grouped' else '70%'}")
    print(f"  Backbone: {emotion_config['model_name']}")
    print(f"  Max Length: {model_config['max_length']} tokens (increased from 128)")
    print(f"  Epochs: {model_config['epochs']}")
    print(f"  Batch Size: {model_config['batch_size']} (effective: {effective_batch} with gradient accumulation)")
    print(f"  Learning Rate: {model_config['learning_rate']:.2e} (reduced for stability)")
    print(f"  LoRA Rank: {lora_rank} (increased capacity)")
    print(f"  LoRA Alpha: {lora_alpha}")
    print(f"  LoRA Dropout: {lora_dropout}")
    print(f"  LoRA Target Modules: {lora_target_modules}")
    print(f"  Mixed Precision: {config['training']['mixed_precision']}")
    print(f"  Gradient Accumulation: {gradient_accum} steps")
    print(f"  Scheduler: {model_config.get('scheduler_type', 'linear')}")
    print(f"  Label Smoothing: {model_config.get('label_smoothing', 0.0)} (enabled for regularization)")
    print(f"  Focal Loss: {model_config.get('use_focal_loss', False)} (disabled, using weighted sampling)")
    print(f"  Weighted Sampling: {model_config.get('use_class_weights', True)}")
    print(f"  Early Stopping Patience: {model_config['early_stopping_patience']}")
    print()

    # Display expectations
    if torch.cuda.is_available():
        print(
            f"Early Stopping: Patience {model_config['early_stopping_patience']} epochs (accuracy-driven)"
        )
        if hierarchy_mode == "basic":
            print("Target: 95%+ validation accuracy on 6 basic emotion classes")
        elif hierarchy_mode == "grouped":
            print("Target: 87%+ validation accuracy on 15 emotion groups")
        else:
            print("Target: 70%+ validation accuracy on 44 fine-grained emotions")
    else:
        print("Estimated Training Time: 20-30 hours (CPU mode)")
    print()

    print("Starting training...")
    print("=" * 80)
    print()

    try:
        # Train the model
        history = trainer.train()

        print()
        print("=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print()

        # Display results
        best_epoch = history.get("best_epoch", 0)
        best_acc = history.get("best_val_accuracy", 0.0)
        best_f1 = history.get("best_val_f1", 0.0)

        print("Final Results:")
        print(f"  Best Epoch: {best_epoch + 1}")
        print(f"  Best Validation Accuracy: {best_acc:.4f}")
        print(f"  Best Validation F1 Score: {best_f1:.4f}")
        print()

        print("Saved Files:")
        print(f"  Best Model (PTH): models/best_model.pth")
        print(f"  Last Model (PTH): models/last_model.pth")
        print(f"  Model (HuggingFace): models/checkpoints_lora/best_model/")
        print(f"  Visualizations: visualizations/emotion_training/")
        print()

        print("=" * 80)
        print("Emotion detection model with LoRA is ready!")
        print()
        print("Benefits of LoRA:")
        print("  30-50% faster training")
        print("  Only a small fraction of parameters are trainable vs full fine tuning")
        print("  Better generalization with less overfitting")
        print()
        print("Next Steps:")
        print("  1. Test the model: python examples/test_emotion_detection.py")
        print("  2. Train response model: python train_response_model.py")
        print("  3. Run chatbot: python -m src.chatbot_interface")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Partial progress may be saved in checkpoints_lora/")
        sys.exit(1)

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
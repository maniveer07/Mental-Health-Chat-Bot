"""
Direct Training Script for Response Generation Model
Optimized for RTX 4070 GPU
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader_multi import MultiDatasetLoader
from src.response_generator import ResponseGenerator

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("=" * 80)
    print("🎯 RESPONSE GENERATION MODEL TRAINING")
    print("=" * 80)
    print()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config()
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print()
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/response_generation", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize data loader
    print("📦 Loading EmpatheticDialogues dataset...")
    print("   (This will download ~20MB on first run)")
    data_loader = MultiDatasetLoader(config)

    try:
        # Load dataset (using EmpatheticDialogues for response generation)
        train_data, val_data, _ = data_loader.combine_datasets(datasets_to_use=['empathetic_dialogues'])
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  • Training conversations: {len(train_data)}")
        print(f"  • Validation conversations: {len(val_data)}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Try running again (dataset will be cached)")
        print("  3. Check disk space (need ~1GB)")
        sys.exit(1)
    
    # Initialize model
    print("🤖 Initializing GPT-2 model...")
    response_generator = ResponseGenerator(config)
    print(f"✓ Model initialized: {config['model']['response_generation']['model_name']}")
    print(f"✓ Parameters: ~124 million")
    print()
    
    # Training configuration
    model_config = config['model']['response_generation']
    print("📊 Training Configuration:")
    print(f"  • Epochs: {model_config['epochs']}")
    print(f"  • Batch Size: {model_config['batch_size']}")
    print(f"  • Gradient Accumulation: {model_config.get('gradient_accumulation_steps', 1)}")
    print(f"  • Effective Batch Size: {model_config['batch_size'] * model_config.get('gradient_accumulation_steps', 1)}")
    print(f"  • Learning Rate: {model_config['learning_rate']}")
    print(f"  • Mixed Precision: {config['training']['mixed_precision']}")
    print()
    
    # Estimate training time
    if torch.cuda.is_available():
        print("⏱️  Estimated Training Time: 2.5-3.5 hours")
    else:
        print("⏱️  Estimated Training Time: 30-40 hours (CPU mode)")
    print()
    
    print("💡 Tips:")
    print("  • Keep laptop plugged in")
    print("  • Ensure good cooling")
    print("  • You can work on other things while training")
    print("  • Monitor GPU: nvidia-smi -l 1 (in another terminal)")
    print()
    
    # Confirm start
    confirm = input("Ready to start training? This will take ~3 hours. (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Training cancelled.")
        sys.exit(0)
    
    print()
    print("🚀 Starting training...")
    print("=" * 80)
    print()
    
    try:
        # Fine-tune the model
        history = response_generator.fine_tune(train_data, val_data)
        
        print()
        print("=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print()
        
        # Display results
        best_epoch = min(range(len(history['val_loss'])), key=lambda i: history['val_loss'][i])
        best_loss = history['val_loss'][best_epoch]
        best_perplexity = history['val_perplexity'][best_epoch]
        
        print("📈 Final Results:")
        print(f"  • Best Epoch: {best_epoch + 1}")
        print(f"  • Best Validation Loss: {best_loss:.4f}")
        print(f"  • Best Validation Perplexity: {best_perplexity:.2f}")
        print()
        
        print("💾 Saved Files:")
        print(f"  • Model: models/response_generation/best_model/")
        print()
        
        print("=" * 80)
        print("🎉 Response generation model is ready!")
        print()
        print("Next Steps:")
        print("  1. Launch chatbot: python -m src.chatbot_interface")
        print("  2. Test the full system with emotion detection + response generation")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Partial progress may be saved in checkpoints/")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

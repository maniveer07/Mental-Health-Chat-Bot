"""
Emotion Detection Module using Hugging Face Transformers

This module implements emotion detection using a fine tuned Transformer model.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoConfig,
)

from peft import PeftModel

logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion detection."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 128,
    ):
        """
        Initialize emotion dataset.

        Args:
            texts: List of text samples.
            labels: List of emotion labels as integer IDs.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()

        return item


class EmotionDetector:
    """
    Emotion detection model using a Hugging Face Transformer.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 44,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        """
        Initialize emotion detector.

        Args:
            model_name: Pretrained model name or path.
            num_labels: Number of emotion classes.
            max_length: Maximum sequence length.
            device: Device to use ("cuda", "cpu", or None for auto).
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading tokenizer and model from: {model_name}")

        # Model agnostic tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

        # Generic 44 class mapping.
        # This will be overridden when loading from emotion_config.json or a checkpoint.
        self.id_to_emotion = {i: f"emotion_{i}" for i in range(self.num_labels)}
        self.emotion_to_id = {v: k for k, v in self.id_to_emotion.items()}

        logger.info(f"EmotionDetector initialized with {num_labels} emotion classes")

    def preprocess_input(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single text input for prediction.

        Args:
            text: Input text.

        Returns:
            Tokenized inputs as tensors on the correct device.
        """
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

        if "token_type_ids" in encoding:
            inputs["token_type_ids"] = encoding["token_type_ids"].to(self.device)

        return inputs

    def create_data_loader(
        self,
        df: pd.DataFrame,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader from a DataFrame.

        Args:
            df: DataFrame with "text" and "label" columns.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.

        Returns:
            PyTorch DataLoader.
        """
        dataset = EmotionDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def train_epoch(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
        criterion: Optional[nn.Module] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data_loader: Training data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler, optional.
            max_grad_norm: Maximum gradient norm for clipping.
            criterion: Custom loss function. Uses model default loss if None.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(data_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

            if criterion is None:
                kwargs["labels"] = labels

            outputs = self.model(**kwargs)

            if criterion is not None:
                logits = outputs.logits
                loss = criterion(logits, labels)
            else:
                loss = outputs.loss

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            optimizer.step()

            if scheduler is not None and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on validation or test set.

        Args:
            data_loader: Evaluation data loader.

        Returns:
            Tuple of (average_loss, predictions, true_labels).
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating")
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                kwargs: Dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**kwargs)

                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        return avg_loss, np.array(all_predictions), np.array(all_labels)

    def predict_emotion(
        self,
        text: str,
        return_probabilities: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict emotion for a single text input.

        Args:
            text: Input text.
            return_probabilities: Whether to return probability distribution.

        Returns:
            Dictionary with prediction results.
        """
        self.model.eval()
        inputs = self.preprocess_input(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        result: Dict[str, Any] = {
            "emotion": self.id_to_emotion.get(predicted_class, str(predicted_class)),
            "emotion_id": predicted_class,
            "confidence": confidence,
        }

        if return_probabilities:
            top_probs, top_indices = torch.topk(probabilities[0], k=5)
            result["top_emotions"] = [
                {
                    "emotion": self.id_to_emotion.get(idx.item(), str(idx.item())),
                    "probability": prob.item(),
                }
                for prob, idx in zip(top_probs, top_indices)
            ]

        return result

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predict emotions for multiple texts.

        Args:
            texts: List of input texts.
            batch_size: Batch size for inference.

        Returns:
            List of prediction result dictionaries.
        """
        self.model.eval()
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            if "token_type_ids" in encodings:
                kwargs["token_type_ids"] = encodings["token_type_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(**kwargs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

            for j, pred in enumerate(predictions):
                pred_id = pred.item()
                confidence = probabilities[j][pred_id].item()
                results.append(
                    {
                        "text": batch_texts[j],
                        "emotion": self.id_to_emotion.get(pred_id, str(pred_id)),
                        "emotion_id": pred_id,
                        "confidence": confidence,
                    }
                )

        return results

    def save_model(self, path: str):
        """
        Save model and tokenizer to disk.

        Args:
            path: Directory path to save model.
        """
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "id_to_emotion": self.id_to_emotion,
        }

        with open(os.path.join(path, "emotion_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model and tokenizer from disk.

        Handles both plain HF checkpoints and LoRA adapter checkpoints.

        Args:
            path: Directory path containing saved model or adapter.
        """
        # Load custom configuration if present
        config_path = os.path.join(path, "emotion_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.num_labels = cfg.get("num_labels", self.num_labels)
            self.max_length = cfg.get("max_length", self.max_length)
            if "id_to_emotion" in cfg:
                self.id_to_emotion = {
                    int(k): v for k, v in cfg["id_to_emotion"].items()
                }
                self.emotion_to_id = {v: k for k, v in self.id_to_emotion.items()}

        adapter_config_path = os.path.join(path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            logger.info(f"Detected LoRA adapter at {path}")

            base_config = AutoConfig.from_pretrained(self.model_name)
            base_config.num_labels = self.num_labels

            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=base_config,
            )

            self.model = PeftModel.from_pretrained(base_model, path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            logger.info(
                f"LoRA adapter loaded from {path} with {self.num_labels} labels"
            )
        else:
            logger.info(f"Loading base model from {path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                num_labels=self.num_labels,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            logger.info(
                f"Base model loaded from {path} with {self.num_labels} labels"
            )

        self.model.to(self.device)
        logger.info(f"Model loaded and moved to device: {self.device}")

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model from a .pth checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "id_to_emotion" in checkpoint:
            self.id_to_emotion = {
                int(k): v for k, v in checkpoint["id_to_emotion"].items()
            }
            self.emotion_to_id = {
                v: int(k) for k, v in checkpoint["id_to_emotion"].items()
            }

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        epoch = checkpoint.get("epoch", -1)
        metrics = checkpoint.get("metrics", {})
        acc = metrics.get("accuracy")
        f1 = metrics.get("weighted_f1")

        logger.info(f"  Epoch: {epoch + 1}")
        if acc is not None:
            logger.info(f"  Validation Accuracy: {acc:.4f}")
        if f1 is not None:
            logger.info(f"  Validation F1: {f1:.4f}")

    def _get_base_encoder(self):
        """
        Try to get the underlying encoder model for embeddings.
        Handles plain HF models and PeftModel wrapped models.
        """
        model = self.model

        # PeftModel case
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()
        elif hasattr(model, "base_model"):
            base = model.base_model
        else:
            base = model

        return base

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get Transformer embedding for text, useful for analysis.

        Args:
            text: Input text.

        Returns:
            Embedding vector as numpy array.
        """
        self.model.eval()
        inputs = self.preprocess_input(text)

        base_model = self._get_base_encoder()

        with torch.no_grad():
            outputs = base_model(
                **inputs,
                output_hidden_states=True,
            )

            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden = outputs.hidden_states[-1]
            else:
                raise ValueError("Model outputs do not contain hidden states")

            # CLS token representation
            embedding = hidden[:, 0, :].cpu().numpy()

        return embedding.squeeze()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    detector = EmotionDetector(
        model_name="distilbert-base-uncased",
        num_labels=44,
        max_length=128,
        device="auto",
    )

    test_texts = [
        "I've been feeling really anxious lately and can't sleep",
        "I'm so grateful for your support during this difficult time",
        "I feel hopeless and don't know what to do anymore",
        "Today was a good day, I'm feeling much better",
        "I'm confused about my emotions and need help",
    ]

    print("\nTesting emotion detection:")
    for text in test_texts:
        result = detector.predict_emotion(text, return_probabilities=True)
        print(f"\nText: {text}")
        print(
            f"Predicted emotion: {result['emotion']} "
            f"(confidence: {result['confidence']:.3f})"
        )
        if "top_emotions" in result:
            print("Top 5 emotions:")
            for emotion_data in result["top_emotions"]:
                print(
                    f"  - {emotion_data['emotion']}: "
                    f"{emotion_data['probability']:.3f}"
                )
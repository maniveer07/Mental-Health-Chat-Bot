"""
Training Module for Emotion Detection Model

This module orchestrates the training process with checkpointing, early stopping,
logging, and evaluation metrics.
"""

import os
import time
import json
import yaml
from typing import Dict, Optional
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

try:
    from .losses import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
except ImportError:
    from losses import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss

logger = logging.getLogger(__name__)


class EmotionTrainer:
    """
    Trainer class for emotion detection model with comprehensive training features.
    """
    
    def __init__(
        self,
        model,
        config: Dict,
        train_dataloader,
        val_dataloader,
        test_dataloader=None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: EmotionDetector instance
            config: Configuration dictionary
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader (optional)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        # Training configuration
        self.epochs = config['model']['emotion_detection']['epochs']
        self.learning_rate = config['model']['emotion_detection']['learning_rate']
        self.weight_decay = config['model']['emotion_detection']['weight_decay']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.early_stopping_patience = config['model']['emotion_detection']['early_stopping_patience']
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = config['paths']['checkpoints_dir']
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup visualizations directory
        self.viz_dir = config['paths'].get('visualizations_dir', 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        logger.info("EmotionTrainer initialized")
        logger.info(f"Training for {self.epochs} epochs with learning rate {self.learning_rate}")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_dataloader) * self.epochs
        num_warmup_steps = self.config['model']['emotion_detection'].get('warmup_steps', 500)
        scheduler_type = self.config['model']['emotion_detection'].get('scheduler_type', 'linear')

        if scheduler_type == 'plateau':
            # Torch version here does not support verbose arg, so we keep it minimal
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',     # Maximize F1 score or accuracy
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
            self.scheduler_type = 'plateau'
            logger.info("Using ReduceLROnPlateau scheduler")
        elif scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            self.scheduler_type = 'warmup'
            logger.info("Using cosine annealing scheduler")
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            self.scheduler_type = 'warmup'
            logger.info("Using linear scheduler")
        
        # Setup custom loss function
        self._setup_loss_function()
        
        logger.info("Optimizer and scheduler configured")
        logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
    
    def _setup_loss_function(self):
        """Setup custom loss function based on config."""
        emotion_config = self.config['model']['emotion_detection']
        
        use_focal_loss = emotion_config.get('use_focal_loss', False)
        label_smoothing = emotion_config.get('label_smoothing', 0.0)
        
        if use_focal_loss and label_smoothing > 0:
            focal_alpha = emotion_config.get('focal_alpha', 0.25)
            focal_gamma = emotion_config.get('focal_gamma', 2.0)
            self.criterion = CombinedLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                smoothing=label_smoothing,
                focal_weight=0.6
            )
            logger.info(
                f"Using combined focal plus label smoothing loss "
                f"(alpha={focal_alpha}, gamma={focal_gamma}, smoothing={label_smoothing})"
            )
        elif use_focal_loss:
            focal_alpha = emotion_config.get('focal_alpha', 0.25)
            focal_gamma = emotion_config.get('focal_gamma', 2.0)
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using focal loss (alpha={focal_alpha}, gamma={focal_gamma})")
        elif label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            logger.info(f"Using label smoothing loss (smoothing={label_smoothing})")
        else:
            self.criterion = None
            logger.info("Using standard cross entropy loss")
    
    def train(self, start_epoch: int = 0) -> Dict:
        """
        Execute complete training loop.
        
        Args:
            start_epoch: Epoch to start from
        
        Returns:
            Dictionary containing training history and final metrics
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            logger.info("\n" + "=" * 60)
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info("=" * 60)
            
            # Training phase
            train_loss = self.model.train_epoch(
                self.train_dataloader,
                self.optimizer,
                self.scheduler,
                self.max_grad_norm,
                criterion=self.criterion if hasattr(self, 'criterion') else None
            )
            
            # Validation phase
            val_loss, val_predictions, val_labels = self.model.evaluate(self.val_dataloader)
            
            # Calculate metrics
            val_metrics = self._calculate_metrics(val_predictions, val_labels)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['weighted_f1'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
            logger.info(f"  Val Weighted Precision: {val_metrics['weighted_precision']:.4f}")
            logger.info(f"  Val Weighted Recall: {val_metrics['weighted_recall']:.4f}")
            logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"  Epoch Time: {epoch_time:.2f}s")

            # Also print to stdout so you see it clearly in the terminal
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Weighted F1: {val_metrics['weighted_f1']:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print()

            # Step scheduler for ReduceLROnPlateau, which needs metric value
            if self.scheduler_type == 'plateau':
                scheduler_metric = self.config['model']['emotion_detection'].get('scheduler_metric', 'f1')
                metric_value = val_metrics['accuracy'] if scheduler_metric == 'accuracy' else val_metrics['weighted_f1']
                self.scheduler.step(metric_value)

            # Save checkpoint if best model based on accuracy
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['weighted_f1']
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                self.patience_counter = 0
                logger.info(
                    f"  ✓ New best model saved! "
                    f"(Accuracy: {self.best_val_accuracy:.4f}, F1: {self.best_val_f1:.4f})"
                )
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement for {self.patience_counter} epoch(s)")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Best epoch: {self.best_epoch + 1}")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
        
        # Test evaluation
        final_metrics = {}
        if self.test_dataloader is not None:
            logger.info("\nEvaluating on test set...")
            self._load_best_checkpoint()
            test_loss, test_predictions, test_labels = self.model.evaluate(self.test_dataloader)
            test_metrics = self._calculate_metrics(test_predictions, test_labels)
            
            logger.info("\nTest Set Results:")
            logger.info(f"  Test Loss: {test_loss:.4f}")
            logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
            logger.info(f"  Test Weighted Precision: {test_metrics['weighted_precision']:.4f}")
            logger.info(f"  Test Weighted Recall: {test_metrics['weighted_recall']:.4f}")
            
            final_metrics['test'] = test_metrics
            
            # Generate visualizations
            self._generate_visualizations(test_predictions, test_labels, prefix='test')
        
        # Save training history
        self._save_training_history()
        
        # Generate training curves
        self._plot_training_curves()
        
        final_metrics['training_history'] = self.training_history
        final_metrics['best_val_accuracy'] = self.best_val_accuracy
        final_metrics['best_val_f1'] = self.best_val_f1
        final_metrics['best_val_loss'] = self.best_val_loss
        final_metrics['best_epoch'] = self.best_epoch

        return final_metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary containing various metrics
        """
        accuracy = accuracy_score(labels, predictions)

        # Fixed label space based on id_to_emotion so we always get length 44
        label_ids = sorted(self.model.id_to_emotion.keys())
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average='weighted',
            zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_ids,
            average=None,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'per_class_support': support_per_class.tolist()
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint_name = 'best_model' if is_best else f'checkpoint_epoch_{epoch+1}'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # Save model in HuggingFace format
        self.model.save_model(checkpoint_path)

        # Save training state
        state = {
            'epoch': epoch,
            'metrics': metrics,
            'training_history': self.training_history,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }

        torch.save(state, os.path.join(checkpoint_path, 'training_state.pt'))

        complete_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1': self.best_val_f1,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'emotion_to_id': self.model.emotion_to_id,
            'id_to_emotion': self.model.id_to_emotion
        }

        if is_best:
            best_pth_path = os.path.join(self.config['paths']['models_dir'], 'best_model.pth')
            torch.save(complete_checkpoint, best_pth_path)
            logger.info(f"Best model saved to {best_pth_path}")

        last_pth_path = os.path.join(self.config['paths']['models_dir'], 'last_model.pth')
        torch.save(complete_checkpoint, last_pth_path)
        logger.info(f"Last model saved to {last_pth_path}")
    
    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model')
        if os.path.exists(best_model_path):
            self.model.load_model(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning("Best model checkpoint not found")
    
    def _save_training_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def _plot_training_curves(self):
        """Generate and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(epochs, self.training_history['val_accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curve
        axes[1, 0].plot(epochs, self.training_history['val_f1'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Weighted F1 Score', fontsize=12)
        axes[1, 0].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 1].plot(epochs, self.training_history['learning_rate'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
        plt.close()
    
    def _generate_visualizations(self, predictions: np.ndarray, labels: np.ndarray, prefix: str = 'test'):
        """
        Generate and save evaluation visualizations.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            prefix: Prefix for saved filenames
        """
        self._plot_confusion_matrix(predictions, labels, prefix)
        self._plot_per_class_f1(predictions, labels, prefix)
    
    def _plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, prefix: str):
        """Plot and save confusion matrix."""
        label_ids = sorted(self.model.id_to_emotion.keys())
        cm = confusion_matrix(labels, predictions, labels=label_ids)
        
        emotions = [self.model.id_to_emotion[i] for i in label_ids]
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix ({prefix.capitalize()} Set)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, f'{prefix}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def _plot_per_class_f1(self, predictions: np.ndarray, labels: np.ndarray, prefix: str):
        """Plot per class F1 scores."""
        label_ids = sorted(self.model.id_to_emotion.keys())
        emotions = [self.model.id_to_emotion[i] for i in label_ids]
        
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            labels,
            predictions,
            labels=label_ids,
            average=None,
            zero_division=0
        )
        
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
        plt.bar(range(len(emotions)), f1_per_class, color=colors)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title(f'Per Class F1 Scores ({prefix.capitalize()} Set)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(emotions)), emotions, rotation=45, ha='right')
        plt.axhline(y=f1_per_class.mean(), color='r', linestyle='--', label=f'Mean: {f1_per_class.mean():.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, f'{prefix}_per_class_f1.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per class F1 scores saved to {save_path}")
        plt.close()
        
        # Print classification report
        report = classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=emotions,
            zero_division=0
        )
        
        report_path = os.path.join(self.checkpoint_dir, f'{prefix}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {report_path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Trainer module example usage")
    logger.info("See training notebooks for complete training pipeline")
"""
Custom Loss Functions for Emotion Detection

Implements focal loss and label smoothing for better training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy ones.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] for class balance
            gamma: Focusing parameter (>= 0). Higher gamma = more focus on hard examples
            reduction: 'none', 'mean', or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    
    Prevents overconfidence by encouraging the model to be less certain.
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor in [0, 1]
            reduction: 'none', 'mean', or 'sum'
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute label smoothing cross entropy.
        
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combines Focal Loss and Label Smoothing.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1, focal_weight=0.7):
        """
        Initialize combined loss.
        
        Args:
            alpha: Focal loss alpha
            gamma: Focal loss gamma
            smoothing: Label smoothing factor
            focal_weight: Weight for focal loss (1-focal_weight for label smoothing)
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.focal_weight = focal_weight
    
    def forward(self, inputs, targets):
        """
        Compute combined loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        smooth = self.label_smoothing_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth

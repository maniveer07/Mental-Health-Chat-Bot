"""
Hierarchical Emotion Classification System

Implements a 2-tier approach:
- Tier 1: 6 basic emotions (~95% accuracy)
- Tier 2: 15 emotion groups (~85-90% accuracy)
- Tier 3: 44 fine-grained emotions (~70-75% accuracy)
"""

from typing import Dict, List, Tuple
import numpy as np


class EmotionHierarchy:
    """Hierarchical emotion classification with confidence-based routing."""

    # Tier 1: Basic emotions (Ekman's 6 + neutral)
    BASIC_EMOTIONS = {
        'joy': [
            'joy', 'amusement', 'excitement', 'pride', 'gratitude',
            'love', 'relief', 'optimism', 'admiration', 'caring',
            'impressed', 'hopeful', 'confident', 'prepared', 'faithful'
        ],
        'sadness': [
            'sadness', 'grief', 'disappointment', 'remorse',
            'devastated', 'lonely', 'guilty', 'ashamed', 'embarrassment'
        ],
        'anger': [
            'anger', 'annoyance', 'disapproval', 'disgust', 'furious'
        ],
        'fear': [
            'fear', 'nervousness', 'anxious', 'apprehensive', 'terrified'
        ],
        'surprise': [
            'surprise', 'realization', 'confusion', 'curiosity'
        ],
        'neutral': [
            'neutral', 'nostalgic', 'sentimental', 'desire', 'approval'
        ]
    }

    # Tier 2: Emotion groups (15 categories)
    GROUPED_EMOTIONS = {
        # Positive high-arousal
        'joy': ['joy', 'amusement', 'excitement'],
        'love': ['love', 'admiration', 'caring'],
        'gratitude': ['gratitude', 'pride', 'relief'],
        'optimism': ['optimism', 'hopeful', 'confident', 'prepared', 'faithful'],

        # Negative high-arousal
        'anger': ['anger', 'furious'],
        'fear': ['fear', 'terrified', 'apprehensive'],
        'anxiety': ['anxious', 'nervousness', 'lonely'],

        # Negative low-arousal
        'sadness': ['sadness', 'grief', 'devastated'],
        'disappointment': ['disappointment', 'remorse'],
        'shame': ['embarrassment', 'ashamed', 'guilty'],

        # Cognitive
        'annoyance': ['annoyance', 'disapproval', 'disgust'],
        'surprise': ['surprise', 'realization'],
        'curiosity': ['curiosity', 'confusion'],
        'desire': ['desire', 'jealous', 'impressed'],

        # Neutral/Complex
        'neutral': ['neutral', 'nostalgic', 'sentimental', 'approval']
    }

    def __init__(self, mode: str = 'basic'):
        """
        Initialize emotion hierarchy.

        Args:
            mode: 'basic' (6 classes), 'grouped' (15 classes), or 'fine' (44 classes)
        """
        self.mode = mode

        if mode == 'basic':
            self.emotion_map = self.BASIC_EMOTIONS
        elif mode == 'grouped':
            self.emotion_map = self.GROUPED_EMOTIONS
        else:
            # Fine-grained: identity mapping
            all_emotions = [
                'admiration', 'amusement', 'anger', 'annoyance', 'anxious', 'apprehensive',
                'approval', 'ashamed', 'caring', 'confident', 'confusion', 'curiosity',
                'desire', 'devastated', 'disappointment', 'disapproval', 'disgust',
                'embarrassment', 'excitement', 'faithful', 'fear', 'furious', 'gratitude',
                'grief', 'guilty', 'hopeful', 'impressed', 'jealous', 'joy', 'lonely',
                'love', 'nervousness', 'neutral', 'nostalgic', 'optimism', 'prepared',
                'pride', 'realization', 'relief', 'remorse', 'sadness', 'sentimental',
                'surprise', 'terrified'
            ]
            self.emotion_map = {e: [e] for e in all_emotions}

        # Create reverse mapping
        self.fine_to_coarse = {}
        for coarse_emotion, fine_emotions in self.emotion_map.items():
            for fine_emotion in fine_emotions:
                self.fine_to_coarse[fine_emotion] = coarse_emotion

        # Create label mappings
        self.coarse_labels = sorted(self.emotion_map.keys())
        self.coarse_to_id = {label: idx for idx, label in enumerate(self.coarse_labels)}
        self.id_to_coarse = {idx: label for label, idx in self.coarse_to_id.items()}

        print(f"Emotion Hierarchy initialized in '{mode}' mode")
        print(f"  Number of classes: {len(self.coarse_labels)}")
        print(f"  Classes: {self.coarse_labels}")

    def map_fine_to_coarse(self, fine_emotion: str) -> str:
        """Map fine-grained emotion to coarse category."""
        return self.fine_to_coarse.get(fine_emotion, 'neutral')

    def map_fine_label_to_coarse_id(self, fine_emotion: str) -> int:
        """Map fine-grained emotion to coarse category ID."""
        coarse_emotion = self.map_fine_to_coarse(fine_emotion)
        return self.coarse_to_id[coarse_emotion]

    def get_num_classes(self) -> int:
        """Get number of classes in current mode."""
        return len(self.coarse_labels)

    def get_emotion_to_id(self) -> Dict[str, int]:
        """Get emotion to ID mapping."""
        return self.coarse_to_id.copy()

    def get_id_to_emotion(self) -> Dict[int, str]:
        """Get ID to emotion mapping."""
        return self.id_to_coarse.copy()


def create_hierarchical_dataset(df, emotion_hierarchy: EmotionHierarchy):
    """
    Convert fine-grained emotion labels to hierarchical labels.

    Args:
        df: DataFrame with 'emotion' and 'label' columns
        emotion_hierarchy: EmotionHierarchy instance

    Returns:
        Modified DataFrame with coarse labels
    """
    import pandas as pd

    # Map fine emotions to coarse categories
    df = df.copy()
    df['coarse_emotion'] = df['emotion'].apply(emotion_hierarchy.map_fine_to_coarse)
    df['coarse_label'] = df['coarse_emotion'].apply(
        lambda x: emotion_hierarchy.coarse_to_id[x]
    )

    # Update original columns
    df['original_emotion'] = df['emotion']
    df['original_label'] = df['label']
    df['emotion'] = df['coarse_emotion']
    df['label'] = df['coarse_label']

    print(f"\nDataset converted to hierarchical labels:")
    print(f"  Original classes: {df['original_emotion'].nunique()}")
    print(f"  New classes: {df['emotion'].nunique()}")
    print(f"\nClass distribution:")
    print(df['emotion'].value_counts().sort_index())

    return df


if __name__ == "__main__":
    # Test basic mode
    print("="*80)
    print("BASIC MODE (6 classes)")
    print("="*80)
    hierarchy_basic = EmotionHierarchy(mode='basic')

    test_emotions = ['joy', 'furious', 'terrified', 'devastated', 'confusion', 'neutral']
    for emotion in test_emotions:
        coarse = hierarchy_basic.map_fine_to_coarse(emotion)
        label_id = hierarchy_basic.map_fine_label_to_coarse_id(emotion)
        print(f"  {emotion:15s} -> {coarse:10s} (ID: {label_id})")

    print("\n" + "="*80)
    print("GROUPED MODE (15 classes)")
    print("="*80)
    hierarchy_grouped = EmotionHierarchy(mode='grouped')

    for emotion in test_emotions:
        coarse = hierarchy_grouped.map_fine_to_coarse(emotion)
        label_id = hierarchy_grouped.map_fine_label_to_coarse_id(emotion)
        print(f"  {emotion:15s} -> {coarse:15s} (ID: {label_id})")

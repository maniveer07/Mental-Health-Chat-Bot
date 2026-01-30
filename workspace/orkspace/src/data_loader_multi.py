"""
Extended Data Loader with Multiple Emotion Datasets

This module loads and combines multiple emotion/mental health datasets
for comprehensive emotion detection training.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class MultiDatasetLoader:
    """
    Load and combine multiple emotion datasets for better coverage.
    Unified taxonomy: 44 unique emotions across all datasets.
    """

    # Unified emotion labels (44 emotions - all unique emotions across datasets)
    UNIFIED_EMOTIONS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'anxious', 'apprehensive',
        'approval', 'ashamed', 'caring', 'confident', 'confusion', 'curiosity',
        'desire', 'devastated', 'disappointment', 'disapproval', 'disgust',
        'embarrassment', 'excitement', 'faithful', 'fear', 'furious', 'gratitude',
        'grief', 'guilty', 'hopeful', 'impressed', 'jealous', 'joy', 'lonely',
        'love', 'nervousness', 'neutral', 'nostalgic', 'optimism', 'prepared',
        'pride', 'realization', 'relief', 'remorse', 'sadness', 'sentimental',
        'surprise', 'terrified'
    ]

    def __init__(self, config: Dict, cache_dir: str = "data/processed"):
        """
        Initialize multi-dataset loader.

        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching processed data
        """
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(self.UNIFIED_EMOTIONS)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}
    
    def load_goemotions(self) -> pd.DataFrame:
        """Load GoEmotions dataset (58K Reddit comments, 28 emotions)."""
        logger.info("Loading GoEmotions dataset...")

        # GoEmotions original label order
        goemotions_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        try:
            dataset = load_dataset("go_emotions", "simplified")

            all_data = []
            for split in ['train', 'validation', 'test']:
                for item in dataset[split]:
                    # Take first label if multi-label
                    original_label = item['labels'][0] if isinstance(item['labels'], list) and len(item['labels']) > 0 else 27
                    emotion_name = goemotions_labels[original_label]

                    # Map to unified taxonomy
                    unified_label = self.emotion_to_id[emotion_name]

                    all_data.append({
                        'text': item['text'].strip(),
                        'label': unified_label,
                        'emotion': emotion_name,
                        'source': 'goemotions'
                    })

            df = pd.DataFrame(all_data)
            logger.info(f"✓ GoEmotions loaded: {len(df):,} samples")
            return df

        except Exception as e:
            logger.error(f"Failed to load GoEmotions: {e}")
            raise
    
    def load_emotion_dataset(self) -> pd.DataFrame:
        """Load Emotion dataset (20K tweets, 6 emotions)."""
        logger.info("Loading Emotion dataset (tweets)...")

        # Map emotion dataset labels to emotion names
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        try:
            dataset = load_dataset("emotion")

            all_data = []
            for split in ['train', 'validation', 'test']:
                for item in dataset[split]:
                    emotion_name = emotion_labels[item['label']]
                    # Map to unified taxonomy
                    unified_label = self.emotion_to_id[emotion_name]

                    all_data.append({
                        'text': item['text'].strip(),
                        'label': unified_label,
                        'emotion': emotion_name,
                        'source': 'emotion_tweets'
                    })

            df = pd.DataFrame(all_data)
            logger.info(f"✓ Emotion dataset loaded: {len(df):,} samples")
            return df

        except Exception as e:
            logger.warning(f"Could not load Emotion dataset: {e}")
            return pd.DataFrame()
    
    def load_empathetic_dialogues(self) -> pd.DataFrame:
        """Load EmpatheticDialogues (25K conversations, 32 emotions)."""
        logger.info("Loading EmpatheticDialogues dataset...")

        # Map EmpatheticDialogues emotions to unified taxonomy (keeping granularity!)
        # Synonyms merged: angry→anger, joyful→joy, sad→sadness, surprised→surprise,
        # content→joy, disgusted→disgust, embarrassed→embarrassment, grateful→gratitude,
        # disappointed→disappointment, annoyed→annoyance, proud→pride, trusting→caring
        ed_to_unified = {
            'afraid': 'fear',           # Keep as fear (basic)
            'angry': 'anger',           # Merge with anger
            'annoyed': 'annoyance',     # Merge with annoyance
            'anxious': 'anxious',       # KEEP SEPARATE
            'apprehensive': 'apprehensive',  # KEEP SEPARATE
            'ashamed': 'ashamed',       # KEEP SEPARATE
            'caring': 'caring',
            'confident': 'confident',   # KEEP SEPARATE
            'content': 'joy',           # Merge with joy
            'devastated': 'devastated', # KEEP SEPARATE
            'disappointed': 'disappointment',  # Merge with disappointment
            'disgusted': 'disgust',     # Merge with disgust
            'embarrassed': 'embarrassment',  # Merge with embarrassment
            'excited': 'excitement',    # Merge with excitement
            'faithful': 'faithful',     # KEEP SEPARATE
            'furious': 'furious',       # KEEP SEPARATE (stronger than anger)
            'grateful': 'gratitude',    # Merge with gratitude
            'guilty': 'guilty',         # KEEP SEPARATE
            'hopeful': 'hopeful',       # KEEP SEPARATE
            'impressed': 'impressed',   # KEEP SEPARATE
            'jealous': 'jealous',       # KEEP SEPARATE
            'joyful': 'joy',            # Merge with joy
            'lonely': 'lonely',         # KEEP SEPARATE
            'nostalgic': 'nostalgic',   # KEEP SEPARATE
            'prepared': 'prepared',     # KEEP SEPARATE
            'proud': 'pride',           # Merge with pride
            'sad': 'sadness',           # Merge with sadness
            'sentimental': 'sentimental', # KEEP SEPARATE
            'surprised': 'surprise',    # Merge with surprise
            'terrified': 'terrified',   # KEEP SEPARATE (stronger than fear)
            'trusting': 'caring'        # Merge with caring
        }

        try:
            # Use trust_remote_code for datasets with custom scripts
            dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)

            all_data = []
            for split in ['train', 'validation', 'test']:
                for item in dataset[split]:
                    # Get emotion label (context field contains emotion)
                    emotion_str = item.get('context', '').lower() if 'context' in item else ''

                    # Try to get text from utterance or prompt field
                    text = item.get('utterance', item.get('prompt', '')).strip()

                    if not text:
                        continue

                    # Extract emotion from context
                    for emotion_key, unified_emotion in ed_to_unified.items():
                        if emotion_key in emotion_str:
                            unified_label = self.emotion_to_id[unified_emotion]
                            all_data.append({
                                'text': text,
                                'label': unified_label,
                                'emotion': unified_emotion,
                                'source': 'empathetic_dialogues'
                            })
                            break

            df = pd.DataFrame(all_data)
            logger.info(f"✓ EmpatheticDialogues loaded: {len(df):,} samples")
            return df

        except Exception as e:
            logger.warning(f"Could not load EmpatheticDialogues: {e}")
            return pd.DataFrame()
    
    def load_dailydialog(self) -> pd.DataFrame:
        """Load DailyDialog (13K conversations, 7 emotions)."""
        logger.info("Loading DailyDialog dataset...")

        # Map DailyDialog emotions to unified taxonomy
        dd_emotion_names = {
            0: 'neutral',   # no emotion
            1: 'anger',     # anger
            2: 'disgust',   # disgust
            3: 'fear',      # fear
            4: 'joy',       # happiness -> joy
            5: 'sadness',   # sadness
            6: 'surprise'   # surprise
        }

        try:
            # Use trust_remote_code for datasets with custom scripts
            dataset = load_dataset("daily_dialog", trust_remote_code=True)

            all_data = []
            for split in ['train', 'validation', 'test']:
                for item in dataset[split]:
                    # DailyDialog has dialog and emotion fields as lists
                    dialogs = item.get('dialog', [])
                    emotions = item.get('emotion', [])

                    for utterance, emotion in zip(dialogs, emotions):
                        if emotion in dd_emotion_names and utterance.strip():
                            emotion_name = dd_emotion_names[emotion]
                            unified_label = self.emotion_to_id[emotion_name]

                            all_data.append({
                                'text': utterance.strip(),
                                'label': unified_label,
                                'emotion': emotion_name,
                                'source': 'dailydialog'
                            })

            df = pd.DataFrame(all_data)
            logger.info(f"✓ DailyDialog loaded: {len(df):,} samples")
            return df

        except Exception as e:
            logger.warning(f"Could not load DailyDialog: {e}")
            return pd.DataFrame()
    
    def combine_datasets(
        self,
        datasets_to_use: List[str] = ['goemotions', 'emotion', 'empathetic_dialogues', 'dailydialog'],
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and combine multiple datasets.
        
        Args:
            datasets_to_use: List of dataset names to include
            use_cache: Whether to use cached data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        cache_file = os.path.join(self.cache_dir, 'multi_dataset_combined.pkl')
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading combined datasets from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                return cached['train'], cached['val'], cached['test']
            except Exception as e:
                logger.warning(f"Cache load failed: {e}. Reloading datasets...")
        
        logger.info("=" * 80)
        logger.info("LOADING MULTIPLE EMOTION DATASETS")
        logger.info("=" * 80)
        
        all_dfs = []
        
        # Load each dataset
        if 'goemotions' in datasets_to_use:
            df = self.load_goemotions()
            if not df.empty:
                all_dfs.append(df)
        
        if 'emotion' in datasets_to_use or 'emotion_tweets' in datasets_to_use:
            df = self.load_emotion_dataset()
            if not df.empty:
                all_dfs.append(df)
        
        if 'empathetic_dialogues' in datasets_to_use:
            df = self.load_empathetic_dialogues()
            if not df.empty:
                all_dfs.append(df)
        
        if 'dailydialog' in datasets_to_use:
            df = self.load_dailydialog()
            if not df.empty:
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No datasets could be loaded!")
        
        # Combine all datasets
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates and clean
        combined_df = combined_df.drop_duplicates(subset=['text'])
        combined_df = combined_df[combined_df['text'].str.len() > 10]  # Remove very short texts
        
        logger.info(f"\n{'='*80}")
        logger.info(f"COMBINED DATASET STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Total samples: {len(combined_df):,}")
        logger.info(f"\nSource distribution:")
        for source, count in combined_df['source'].value_counts().items():
            logger.info(f"  {source}: {count:,} ({count/len(combined_df)*100:.1f}%)")
        
        # Split into train/val/test
        train_ratio = self.config['data']['goemotions'].get('train_ratio', 0.8)
        val_ratio = self.config['data']['goemotions'].get('val_ratio', 0.1)
        test_ratio = self.config['data']['goemotions'].get('test_ratio', 0.1)
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            combined_df,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=combined_df['label']
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_df['label']
        )
        
        logger.info(f"\nSplit distribution:")
        logger.info(f"  Train: {len(train_df):,} samples ({len(train_df)/len(combined_df)*100:.1f}%)")
        logger.info(f"  Val: {len(val_df):,} samples ({len(val_df)/len(combined_df)*100:.1f}%)")
        logger.info(f"  Test: {len(test_df):,} samples ({len(test_df)/len(combined_df)*100:.1f}%)")
        
        # Cache the processed data
        if use_cache:
            logger.info(f"\nCaching combined datasets to: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({'train': train_df, 'val': val_df, 'test': test_df}, f)
        
        return train_df, val_df, test_df
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def apply_hierarchical_grouping(
        self,
        df: pd.DataFrame,
        mode: str = 'basic'
    ) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, int]]:
        """
        Group emotions into hierarchical categories for higher accuracy.

        Args:
            df: DataFrame with 'emotion' column
            mode: 'basic' (6 classes, target 95%), 'grouped' (15 classes, target 87%), 'fine' (44 classes)

        Returns:
            Tuple of (modified_df, id_to_emotion, emotion_to_id)
        """
        logger.info(f"\nApplying hierarchical emotion grouping (mode: {mode})...")

        if mode == 'basic':
            # 6 basic emotions - highest accuracy target
            emotion_groups = {
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
        elif mode == 'grouped':
            # 15 emotion groups - balanced accuracy/granularity
            emotion_groups = {
                'joy': ['joy', 'amusement', 'excitement'],
                'love': ['love', 'admiration', 'caring'],
                'gratitude': ['gratitude', 'pride', 'relief'],
                'optimism': ['optimism', 'hopeful', 'confident', 'prepared', 'faithful'],
                'anger': ['anger', 'furious'],
                'fear': ['fear', 'terrified', 'apprehensive'],
                'anxiety': ['anxious', 'nervousness', 'lonely'],
                'sadness': ['sadness', 'grief', 'devastated'],
                'disappointment': ['disappointment', 'remorse'],
                'shame': ['embarrassment', 'ashamed', 'guilty'],
                'annoyance': ['annoyance', 'disapproval', 'disgust'],
                'surprise': ['surprise', 'realization'],
                'curiosity': ['curiosity', 'confusion'],
                'desire': ['desire', 'jealous', 'impressed'],
                'neutral': ['neutral', 'nostalgic', 'sentimental', 'approval']
            }
        else:
            # Fine-grained: keep all 44 emotions
            logger.info("Using fine-grained 44-class taxonomy (no grouping)")
            return df, self.id_to_emotion, self.emotion_to_id

        # Create reverse mapping
        fine_to_coarse = {}
        for coarse_emotion, fine_emotions in emotion_groups.items():
            for fine_emotion in fine_emotions:
                fine_to_coarse[fine_emotion] = coarse_emotion

        # Map emotions
        df = df.copy()
        df['original_emotion'] = df['emotion']
        df['original_label'] = df['label']
        df['emotion'] = df['emotion'].map(fine_to_coarse)

        # Handle unmapped emotions (shouldn't happen, but safety)
        unmapped = df['emotion'].isna().sum()
        if unmapped > 0:
            logger.warning(f"Found {unmapped} unmapped emotions, setting to 'neutral'")
            df['emotion'].fillna('neutral', inplace=True)

        # Create new label mappings
        unique_emotions = sorted(df['emotion'].unique())
        new_emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
        new_id_to_emotion = {idx: emotion for emotion, idx in new_emotion_to_id.items()}

        df['label'] = df['emotion'].map(new_emotion_to_id)

        logger.info(f"✓ Emotion grouping applied")
        logger.info(f"  Original classes: {df['original_emotion'].nunique()}")
        logger.info(f"  New classes: {df['emotion'].nunique()}")
        logger.info(f"  Classes: {list(unique_emotions)}")
        logger.info(f"\nClass distribution:")
        for emotion, count in df['emotion'].value_counts().sort_index().items():
            logger.info(f"  {emotion:15s}: {count:6,} ({count/len(df)*100:.1f}%)")

        return df, new_id_to_emotion, new_emotion_to_id

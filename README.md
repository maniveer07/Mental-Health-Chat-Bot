#  Emotion-Aware Mental Health Chatbot

An empathetic AI assistant that detects emotions from text and generates supportive responses for mental health support. This system combines state-of-the-art NLP models with comprehensive safety mechanisms to provide compassionate, culturally-sensitive conversational support.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

##  Important Disclaimer

**This chatbot is for research and educational purposes only. It is NOT a replacement for professional mental health care. If you're in crisis, please contact:**
- **US:** 988 (National Suicide Prevention Lifeline)
- **Crisis Text Line:** Text HOME to 741741
- **Emergency:** 911

##  Features

- **Emotion Detection**: Fine-tuned RoBERTa-base model with LoRA (6 emotion classes: anger, fear, joy, neutral, sadness, surprise)
- **Empathetic Response Generation**: GPT-4o API with emotion conditioning
- **Crisis Detection**: Comprehensive safety module with keyword-based risk assessment
- **Multi-turn Conversations**: Context-aware dialogue management
- **Real-time Visualizations**: Emotion trends and confidence scores
- **Safety Mechanisms**: Content filtering, response validation, crisis resource provision
- **Efficient Training**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **GPU Optimized**: Mixed precision training with CUDA support

##  Architecture

```
┌─────────────────┐
│  User Input     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Emotion         │
│ Detection       │◄─── RoBERTa-base + LoRA
│ (RoBERTa)       │     (6 basic emotions)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Safety Module   │◄─── Crisis Detection
│ (Risk Assessment)│     Content Filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Response        │
│ Generation      │◄─── GPT-4o API
│ (GPT-4o)        │     Emotion-conditioned
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Visualization   │◄─── Real-time Charts
│ Manager         │     Emotion Tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chainlit UI     │◄─── Interactive Chat
│ (Web Interface) │     Emotion Display
└─────────────────┘
```

##  Model Performance

**Emotion Detection Model:**
- **Validation Accuracy:** 76%
- **Validation F1-Score:** 75.5%
- **Training Epochs:** 17 (with early stopping)
- **Architecture:** RoBERTa-base + LoRA (r=16, alpha=32)
- **Dataset:** Combined GoEmotions + DailyDialog + EmotionLines

**Visualizations:**
- Training/Validation Loss curves
- Confusion Matrix
- Per-class F1 scores

See `visualizations/` folder for detailed results.

##  Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- OpenAI API key for GPT-4o

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd orkspace
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file or set environment variables:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-api-key-here"
$env:RESPONSE_MODE = "gpt4_api"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export RESPONSE_MODE="gpt4_api"
```

### Usage

#### 1. Train the Emotion Detection Model (Optional - Pre-trained model included)

```bash
python train_emotion_lora.py
```

**Configuration:** Edit `config/config.yaml` to adjust:
- Training epochs
- Batch size
- Learning rate
- LoRA parameters

**Output:**
- Model checkpoint: `models/checkpoints_lora/best_model/`
- Training history: `models/checkpoints_lora/training_history.json`
- Visualizations: `visualizations/`

#### 2. Run the Chatbot

**Option A: Using Launch Script (Easiest)**

Windows (Command Prompt):
```bash
run_chatbot.bat
```

Windows (PowerShell):
```bash
.\run_chatbot.ps1
```

Linux/Mac:
```bash
./run_chatbot.sh
```

**Option B: Manual Launch**
```bash
# Set environment variables first
export OPENAI_API_KEY="your-api-key"  # Linux/Mac
$env:OPENAI_API_KEY="your-api-key"    # Windows PowerShell

# Run chatbot
chainlit run src/chatbot_interface_chainlit.py -w --host localhost --port 8000
```

Then open your browser to: **http://localhost:8000**

#### 3. Run Tests

```bash
python tests/test_all.py
```

##  Project Structure

```
orkspace/
├── config/
│   └── config.yaml                 # Training and model configuration
├── data/
│   └── processed/
│       └── multi_dataset_combined.pkl  # Pre-processed training data
├── models/
│   ├── best_model.pth             # Legacy checkpoint
│   ├── last_model.pth             # Legacy checkpoint
│   └── checkpoints_lora/          # LoRA fine-tuned model (MAIN)
│       ├── best_model/            # Best performing model
│       │   ├── adapter_model.safetensors
│       │   ├── adapter_config.json
│       │   ├── emotion_config.json
│       │   └── tokenizer files
│       ├── training_history.json  # Training metrics
│       └── test_classification_report.txt
├── src/
│   ├── chatbot_interface_chainlit.py    # Main chatbot interface
│   ├── conversation_manager.py          # Conversation context tracking
│   ├── data_loader_multi.py             # Dataset loading utilities
│   ├── emotion_detector.py              # Emotion classification model
│   ├── emotion_hierarchy.py             # Emotion categorization
│   ├── losses.py                        # Custom loss functions
│   ├── response_generator.py            # GPT-4o API integration
│   ├── safety_module.py                 # Crisis detection & safety
│   ├── trainer.py                       # Model training logic
│   └── visualization_manager.py         # Real-time visualizations
├── tests/
│   └── test_all.py                # Comprehensive test suite
├── visualizations/
│   ├── training_curves.png        # Training/validation metrics
│   ├── test_confusion_matrix.png  # Model confusion matrix
│   └── test_per_class_f1.png      # Per-class performance
├── logs/                          # (Empty - runtime logs generated here)
├── .chainlit/                     # Chainlit configuration
│   └── config.toml
├── train_emotion_lora.py          # Main training script
├── train_response_model.py        # Response model training (optional)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

##  Configuration

### Model Parameters (config/config.yaml)

```yaml
model:
  emotion_model_name: "roberta-base"
  lora:
    r: 16                          # LoRA rank
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["query", "value"]

training:
  num_epochs: 20
  batch_size: 8
  learning_rate: 2.0e-05
  weight_decay: 0.01
  scheduler_type: "cosine"
  early_stopping_patience: 5

emotion_classes:
  - anger
  - fear
  - joy
  - neutral
  - sadness
  - surprise
```

### Response Generation Modes

Set via `RESPONSE_MODE` environment variable:
- `gpt4_api`: GPT-4o API (recommended, requires API key)
- `local`: Local DialoGPT model (no API key needed, lower quality)

##  Testing

The project includes comprehensive tests:

```bash
python tests/test_all.py
```

**Test Coverage:**
-  Configuration loading
-  Emotion detection accuracy
-  Safety module (crisis detection)
-  Response generation
-  Conversation context management
-  Visualization generation

##  Training Results

### Final Metrics
- **Best Validation Accuracy:** 76.0%
- **Best Validation F1-Score:** 75.5%
- **Training Time:** ~2.5 hours (RTX 5070 Ti)
- **Model Size:** ~550MB (LoRA adapters only)

### Key Observations
1. **Overfitting Prevention:** Early stopping at epoch 17 prevented overfitting
2. **Balanced Performance:** F1-score close to accuracy indicates balanced class performance
3. **Computational Efficiency:** LoRA reduced trainable parameters by ~99.7%

##  Safety Features

1. **Crisis Detection Keywords:**
   - Suicide-related terms
   - Self-harm indicators
   - Immediate danger signals

2. **Automatic Response:**
   - Crisis resources provided
   - Professional help encouraged
   - Emergency contacts displayed

3. **Content Filtering:**
   - Harmful content detection
   - Response validation
   - Context-aware safety checks

##  Cost Analysis (GPT-4o)

- **Input:** $2.50 per 1M tokens
- **Output:** $10.00 per 1M tokens
- **Average Cost per Message:** ~$0.0023
- **1000 messages:** ~$2.30

##  Development

### Adding New Features

1. **New Emotion Classes:** Update `config.yaml` and retrain
2. **Custom Safety Rules:** Modify `src/safety_module.py`
3. **UI Customization:** Edit `.chainlit/config.toml`

### Debugging

- **Check logs:** `logs/chatbot_chainlit.log`
- **Verify GPU:** Check CUDA availability in training logs
- **Test individual components:** Use `tests/test_all.py`

##  Dependencies

Key packages:
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - Hugging Face models
- `peft>=0.4.0` - LoRA implementation
- `chainlit>=1.0.0` - Web interface
- `openai>=1.0.0` - GPT-4o API
- `plotly>=5.0.0` - Visualizations
- `pandas>=1.5.0` - Data processing
- `pyyaml>=6.0` - Configuration management

See `requirements.txt` for complete list.



##  License

This project is licensed under the MIT License - see the LICENSE file for details.



##  Acknowledgments

- **Datasets:** GoEmotions, DailyDialog, EmotionLines
- **Models:** Hugging Face Transformers, OpenAI GPT-4o
- **Framework:** PyTorch, Chainlit

---

**Remember:** This is a research tool. Always seek professional help for mental health concerns.

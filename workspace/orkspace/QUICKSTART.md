#  Quick Start Guide

## Run the Chatbot in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Chatbot

**Option A: Using Launch Script (Easiest)**
```bash
# Windows Command Prompt
run_chatbot.bat

# Windows PowerShell
.\run_chatbot.ps1

# Linux/Mac
./run_chatbot.sh
```
 **No API key setup needed - the scripts handle everything automatically!**

**Option B: Manual Launch** (if you want to customize)
```bash
# Set environment variables first
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:RESPONSE_MODE = "gpt4_api"

# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key"
export RESPONSE_MODE="gpt4_api"

# Then run
chainlit run src/chatbot_interface_chainlit.py -w --host localhost --port 8000
```

Then open: **http://localhost:8000**

---

##  Run Tests
```bash
python tests/test_all.py
```

---

##  Retrain Model (Optional - Pre-trained included)
```bash
python train_emotion_lora.py
```
**Note:** Takes ~2.5 hours on RTX 5070 Ti. Pre-trained model already included at `models/checkpoints_lora/best_model/`

---

##  Key Files

| File | Description |
|------|-------------|
| `README.md` | Complete documentation |
| `SUBMISSION_CHECKLIST.md` | Files included & removed |
| `requirements.txt` | Python dependencies |
| `config/config.yaml` | Model configuration |
| `models/checkpoints_lora/best_model/` | Pre-trained model (76% accuracy) |
| `visualizations/training_curves.png` | Training results |
| `tests/test_all.py` | Test suite |

---

##  What to Demonstrate

1. **Start the chatbot** → Web interface opens
2. **Type a message** → e.g., "I'm feeling really stressed about my exams"
3. **Observe:**
   -  Detected emotion displayed (e.g., "fear" or "sadness")
   -  Confidence score shown
   -  Empathetic response generated
   -  Emotion chart updated (ASCII bar chart in chat)

4. **Test crisis detection** → Type "I want to end it all"
   -  Crisis resources automatically displayed
   -  Professional help encouraged

5. **Check conversation context** → Ask follow-up questions
   -  Bot remembers previous messages

---

##  Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "OpenAI API error"
**Solution:** Check API key is set: `echo $env:OPENAI_API_KEY` (Windows) or `echo $OPENAI_API_KEY` (Linux)

### Issue: "CUDA out of memory"
**Solution:** Reduce batch_size in `config/config.yaml` or use CPU

### Issue: Chainlit won't start
**Solution:** 
```bash
pip install chainlit --upgrade
chainlit run src/chatbot_interface_chainlit.py -w
```

---

##  Expected Performance

- **Emotion Detection Accuracy:** 76%
- **F1-Score:** 75.5%
- **Response Time:** ~1-2 seconds (with GPT-4o API)
- **Cost per Message:** ~$0.0023 (GPT-4o)

---

##  For Evaluation

**Model Performance:**
- See `visualizations/training_curves.png` for training metrics
- See `visualizations/test_confusion_matrix.png` for confusion matrix
- See `models/checkpoints_lora/test_classification_report.txt` for detailed report

**Code Quality:**
- Clean modular architecture in `src/`
- Comprehensive tests in `tests/test_all.py`
- Full configuration in `config/config.yaml`

**Safety Features:**
- Crisis detection in `src/safety_module.py`
- Automatic resource provision
- Content filtering

---

**Questions?** Check `README.md` for full documentation.


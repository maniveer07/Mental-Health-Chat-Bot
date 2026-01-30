#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#   Launch Script for Emotion-Aware Mental Health Chatbot
# ═══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Starting Emotion-Aware Mental Health Chatbot"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Set environment variables
# Please set your OPENAI_API_KEY environment variable before running this script
# Example: export OPENAI_API_KEY="your-api-key-here"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set!"
    echo "Please set it using: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi
export RESPONSE_MODE="gpt4_api"

echo "[*] Configuration:"
echo "    - Emotion Model: RoBERTa + LoRA (76% accuracy)"
echo "    - Response Model: GPT-4o API"
echo "    - Safety Module: Active"
echo "    - Port: 8000"
echo ""
echo "[*] Starting Chainlit server..."
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Open your browser to: http://localhost:8000"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch the chatbot
chainlit run src/chatbot_interface_chainlit.py -w --host localhost --port 8000


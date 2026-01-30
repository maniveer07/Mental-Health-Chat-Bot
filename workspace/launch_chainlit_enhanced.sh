#!/bin/bash
# Bash script to launch Enhanced Chainlit chatbot with visualizations

echo "========================================"
echo "Emotion-Aware Mental Health Chatbot"
echo "Enhanced Chainlit Interface with Analytics"
echo "========================================"
echo ""

# Check if in correct directory
if [ ! -f "src/chatbot_interface_chainlit_enhanced.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. Please enter it:"
    read -p "OpenAI API Key: " apiKey
    export OPENAI_API_KEY="$apiKey"
fi

echo "Select Response Mode:"
echo "1. GPT-4o API (Recommended, ~\$0.002/message)"
echo "2. Local DialoGPT (Free, uses your GPU)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    export RESPONSE_MODE="gpt4_api"
    echo "Using GPT-4o API mode"
else
    export RESPONSE_MODE="dialogpt_local"
    echo "Using Local DialoGPT mode"
fi

echo ""
echo "Starting Enhanced Chainlit server with visualizations..."
echo ""
echo "Features:"
echo "  - Real-time emotion tracking"
echo "  - Interactive charts and graphs"
echo "  - Conversation analytics"
echo "  - Export capabilities"
echo ""
echo "Once started, your browser will open automatically"
echo "Or visit: http://localhost:8000"
echo ""
echo "Special Commands:"
echo "  /stats - View detailed statistics"
echo "  /timeline - See emotion timeline"
echo "  /export - Export conversation data"
echo "  /help - Show all commands"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

chainlit run src/chatbot_interface_chainlit_enhanced.py -w --host localhost --port 8000


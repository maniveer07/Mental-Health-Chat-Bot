@echo off
REM ═══════════════════════════════════════════════════════════════
REM   Launch Script for Emotion-Aware Mental Health Chatbot
REM ═══════════════════════════════════════════════════════════════

echo.
echo ════════════════════════════════════════════════════════════════
echo   Starting Emotion-Aware Mental Health Chatbot
echo ════════════════════════════════════════════════════════════════
echo.

REM Set environment variables
REM Please set your OPENAI_API_KEY environment variable before running this script
REM Example: set OPENAI_API_KEY=your-api-key-here
if "%OPENAI_API_KEY%"=="" (
    echo ERROR: OPENAI_API_KEY environment variable is not set!
    echo Please set it using: set OPENAI_API_KEY=your-api-key-here
    pause
    exit /b 1
)
set RESPONSE_MODE=gpt4_api

echo [*] Configuration:
echo     - Emotion Model: RoBERTa + LoRA (76%% accuracy)
echo     - Response Model: GPT-4o API
echo     - Safety Module: Active
echo     - Port: 8000
echo.
echo [*] Starting Chainlit server...
echo.
echo ════════════════════════════════════════════════════════════════
echo   Open your browser to: http://localhost:8000
echo ════════════════════════════════════════════════════════════════
echo.
echo Press Ctrl+C to stop the server
echo.

REM Launch the chatbot
chainlit run src/chatbot_interface_chainlit.py -w --host localhost --port 8000

pause


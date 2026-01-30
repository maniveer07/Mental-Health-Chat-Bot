# ═══════════════════════════════════════════════════════════════
#   Launch Script for Emotion-Aware Mental Health Chatbot
# ═══════════════════════════════════════════════════════════════

Write-Host "`n" -NoNewline
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Starting Emotion-Aware Mental Health Chatbot" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
# Please set your OPENAI_API_KEY environment variable before running this script
# Example: $env:OPENAI_API_KEY = "your-api-key-here"
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY environment variable is not set!" -ForegroundColor Red
    Write-Host "Please set it using: `$env:OPENAI_API_KEY = 'your-api-key-here'" -ForegroundColor Yellow
    exit 1
}
$env:RESPONSE_MODE = "gpt4_api"

Write-Host "[*] Configuration:" -ForegroundColor Yellow
Write-Host "    - Emotion Model: RoBERTa + LoRA (76% accuracy)" -ForegroundColor White
Write-Host "    - Response Model: GPT-4o API" -ForegroundColor White
Write-Host "    - Safety Module: Active" -ForegroundColor White
Write-Host "    - Port: 8000" -ForegroundColor White
Write-Host ""
Write-Host "[*] Starting Chainlit server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Open your browser to: http://localhost:8000" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Launch the chatbot
chainlit run src/chatbot_interface_chainlit.py -w --host localhost --port 8000


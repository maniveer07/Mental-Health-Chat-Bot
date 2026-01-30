#!/usr/bin/env pwsh
# PowerShell script to launch Enhanced Chainlit chatbot with visualizations

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Emotion-Aware Mental Health Chatbot" -ForegroundColor Green
Write-Host "Enhanced Chainlit Interface with Analytics" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in correct directory
if (-not (Test-Path "src/chatbot_interface_chainlit_enhanced.py")) {
    Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Check if API key is set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "Warning: OPENAI_API_KEY not set. Please enter it:" -ForegroundColor Yellow
    $apiKey = Read-Host "OpenAI API Key"
    $env:OPENAI_API_KEY = $apiKey
}

Write-Host "Select Response Mode:" -ForegroundColor Yellow
Write-Host "1. GPT-4o API (Recommended, ~$0.002/message)" -ForegroundColor White
Write-Host "2. Local DialoGPT (Free, uses your GPU)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    $env:RESPONSE_MODE = "gpt4_api"
    Write-Host "Using GPT-4o API mode" -ForegroundColor Green
} else {
    $env:RESPONSE_MODE = "dialogpt_local"
    Write-Host "Using Local DialoGPT mode" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Enhanced Chainlit server with visualizations..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Features:" -ForegroundColor Yellow
Write-Host "  - Real-time emotion tracking" -ForegroundColor White
Write-Host "  - Interactive charts and graphs" -ForegroundColor White
Write-Host "  - Conversation analytics" -ForegroundColor White
Write-Host "  - Export capabilities" -ForegroundColor White
Write-Host ""
Write-Host "Once started, your browser will open automatically" -ForegroundColor White
Write-Host "Or visit: http://localhost:8000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Special Commands:" -ForegroundColor Yellow
Write-Host "  /stats - View detailed statistics" -ForegroundColor White
Write-Host "  /timeline - See emotion timeline" -ForegroundColor White
Write-Host "  /export - Export conversation data" -ForegroundColor White
Write-Host "  /help - Show all commands" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

chainlit run src/chatbot_interface_chainlit_enhanced.py -w --host localhost --port 8000


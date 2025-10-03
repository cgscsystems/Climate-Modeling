# Climate Data Visualization - Auto-Maintenance Setup
# Run this script once to enable automatic documentation updates

Write-Host "🔧 Setting up auto-maintenance for Climate Data Visualization..." -ForegroundColor Cyan

# Check if git hooks are already set up
if (Test-Path ".git/hooks/pre-commit") {
    Write-Host "✅ Pre-commit hook already exists" -ForegroundColor Green
} else {
    Write-Host "❌ Pre-commit hook not found - please run this from the project root" -ForegroundColor Red
    exit 1
}

# Create AI conversations directory if it doesn't exist
if (-not (Test-Path ".ai-conversations")) {
    New-Item -ItemType Directory -Path ".ai-conversations" | Out-Null
    Write-Host "✅ Created .ai-conversations directory" -ForegroundColor Green
}

# Test the hook
Write-Host "🧪 Testing pre-commit hook..." -ForegroundColor Yellow
try {
    powershell.exe -ExecutionPolicy Bypass -File ".git/hooks/pre-commit.ps1"
    Write-Host "✅ Pre-commit hook test successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Pre-commit hook test failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎯 Auto-maintenance setup complete!" -ForegroundColor Yellow
Write-Host ""
Write-Host "What happens now:" -ForegroundColor Cyan
Write-Host "• Every git commit will automatically update documentation timestamps" -ForegroundColor White
Write-Host "• The workspace file will track recent changes" -ForegroundColor White
Write-Host "• AI conversation summaries will be generated" -ForegroundColor White
Write-Host "• All changes will be included in your commit automatically" -ForegroundColor White
Write-Host ""
Write-Host "Files that will be auto-maintained:" -ForegroundColor Cyan
Write-Host "• .github/copilot-instructions.md (timestamp updates)" -ForegroundColor White
Write-Host "• Weather-Modeling.code-workspace (recent changes metadata)" -ForegroundColor White
Write-Host "• .ai-conversations/conversation-summary.md (project status)" -ForegroundColor White
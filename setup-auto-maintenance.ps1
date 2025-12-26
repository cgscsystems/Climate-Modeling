# Climate Data Visualization - Auto-Maintenance Setup
# Run this script once to enable automatic documentation updates

# Check if git hooks are set up
if (-not (Test-Path ".git/hooks/pre-commit")) {
    Write-Error "Pre-commit hook not found - please run this from the project root"
    exit 1
}

# Create AI conversations directory if needed
if (-not (Test-Path ".ai-conversations")) {
    New-Item -ItemType Directory -Path ".ai-conversations" | Out-Null
}

# Test the hook
try {
    powershell.exe -ExecutionPolicy Bypass -File ".git/hooks/pre-commit.ps1"
    Write-Host "Auto-maintenance setup complete - hooks are working"
} catch {
    Write-Error "Pre-commit hook test failed: $_"
    exit 1
}
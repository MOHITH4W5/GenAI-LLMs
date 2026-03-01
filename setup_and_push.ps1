# PowerShell Script to Upload GenAI-LLMs Project to GitHub
# This script initializes Git, sets up the remote, and pushes all files

# Set the project directory
$projectPath = "C:\Users\Lenovo\.gemini\antigravity\scratch\GenAI-LLMs"
$repoURL = "https://github.com/MOHITH4W5/GenAI-LLMs.git"

Write-Host "Starting GenAI-LLMs GitHub upload..." -ForegroundColor Green

# Check if directory exists
if (-not (Test-Path $projectPath)) {
    Write-Host "Error: Project directory not found at $projectPath" -ForegroundColor Red
    exit 1
}

# Change to project directory
Set-Location $projectPath
Write-Host "Changed to project directory: $projectPath" -ForegroundColor Yellow

# Initialize Git if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
} else {
    Write-Host "Git repository already initialized" -ForegroundColor Yellow
}

# Check if remote already exists
$remoteExists = git remote | Select-String -Pattern "^origin$"

if ($null -eq $remoteExists) {
    Write-Host "Adding GitHub remote..." -ForegroundColor Yellow
    git remote add origin $repoURL
} else {
    Write-Host "Remote 'origin' already exists. Updating URL..." -ForegroundColor Yellow
    git remote set-url origin $repoURL
}

# Configure Git user (if not already configured)
Write-Host "Configuring Git user..." -ForegroundColor Yellow
git config user.name "MOHITH4W5"
git config user.email "mohith.ai.ml@gmail.com"

# Add all files
Write-Host "Adding all files..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
$commitMessage = "Upload GenAI-LLMs project files"
git commit -m $commitMessage

# Push to GitHub
Write-Host "Pushing to GitHub (this may require authentication)..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully uploaded project to GitHub!" -ForegroundColor Green
    Write-Host "Repository: https://github.com/MOHITH4W5/GenAI-LLMs" -ForegroundColor Green
} else {
    Write-Host "Error during push. You may need to use Git credentials or SSH." -ForegroundColor Red
    Write-Host "Trying with master branch instead..." -ForegroundColor Yellow
    git push -u origin master
}

Write-Host "Script completed!" -ForegroundColor Green

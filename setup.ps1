# Setup and run FreshScanAI Project
# Windows PowerShell Script

Write-Host "========================================" -ForegroundColor Green
Write-Host "   FreshScanAI - Setup Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Check dataset
Write-Host ""
Write-Host "Checking dataset..." -ForegroundColor Cyan
if (Test-Path "data\raw") {
    $fileCount = (Get-ChildItem -Path "data\raw" -Recurse -File).Count
    if ($fileCount -gt 0) {
        Write-Host "✓ Dataset found: $fileCount files" -ForegroundColor Green
        
        # Ask to preprocess
        Write-Host ""
        $preprocess = Read-Host "Run data preprocessing? (y/n)"
        if ($preprocess -eq "y") {
            Write-Host ""
            Write-Host "Running preprocessing..." -ForegroundColor Cyan
            python preprocessing.py
        }
    } else {
        Write-Host "✗ Dataset folder is empty!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please download dataset from:" -ForegroundColor Yellow
        Write-Host "https://www.kaggle.com/datasets/zlatan599/fruitquality1" -ForegroundColor Yellow
        Write-Host "And extract to: data\raw\" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Dataset not found!" -ForegroundColor Red
    Write-Host "Creating data/raw directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
    Write-Host ""
    Write-Host "Please download dataset from:" -ForegroundColor Yellow
    Write-Host "https://www.kaggle.com/datasets/zlatan599/fruitquality1" -ForegroundColor Yellow
    Write-Host "And extract to: data\raw\" -ForegroundColor Yellow
}

# Check if model exists
Write-Host ""
Write-Host "Checking trained model..." -ForegroundColor Cyan
if (Test-Path "models\freshscan_model.h5") {
    Write-Host "✓ Model found" -ForegroundColor Green
    
    # Ask to run app
    Write-Host ""
    $runApp = Read-Host "Launch web application? (y/n)"
    if ($runApp -eq "y") {
        Write-Host ""
        Write-Host "Starting Streamlit app..." -ForegroundColor Cyan
        Write-Host "App will open in your browser at http://localhost:8501" -ForegroundColor Yellow
        Write-Host ""
        streamlit run app.py
    }
} else {
    Write-Host "✗ Model not found!" -ForegroundColor Red
    Write-Host ""
    $train = Read-Host "Train model now? This will take 30-60 minutes on CPU. (y/n)"
    if ($train -eq "y") {
        Write-Host ""
        Write-Host "Starting model training..." -ForegroundColor Cyan
        python train_model.py
        
        # After training, offer to run app
        Write-Host ""
        $runApp = Read-Host "Launch web application? (y/n)"
        if ($runApp -eq "y") {
            Write-Host ""
            Write-Host "Starting Streamlit app..." -ForegroundColor Cyan
            streamlit run app.py
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Download dataset (if not done)" -ForegroundColor White
Write-Host "2. Run: python preprocessing.py" -ForegroundColor White
Write-Host "3. Run: python train_model.py" -ForegroundColor White
Write-Host "4. Run: streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see QUICKSTART.md" -ForegroundColor Yellow
Write-Host ""

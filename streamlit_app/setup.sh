#!/bin/bash

# Healthcare Fraud Detection App - Setup Script

echo "Setting up the Healthcare Fraud Detection App..."

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Homebrew is already installed."
    fi
    
    # Install libomp (required for XGBoost on macOS)
    echo "Installing OpenMP library (required for XGBoost)..."
    brew install libomp
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source .venv/bin/activate
    else
        # For Windows
        .\.venv\Scripts\activate
    fi
    
    # Install dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "Virtual environment already exists."
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source .venv/bin/activate
    else
        # For Windows
        .\.venv\Scripts\activate
    fi
    
    # Update dependencies
    echo "Updating Python dependencies..."
    pip install -r requirements.txt
fi

echo "Setup complete! Run the app with: streamlit run app2.py"

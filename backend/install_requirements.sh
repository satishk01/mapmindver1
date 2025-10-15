#!/bin/bash

# Installation script for Python 3.11 requirements
# This script will install the optimized requirements with error handling

echo "Installing Python dependencies for Python 3.11..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Detected Python version: $python_version"

if [[ "$python_version" < "3.11" ]]; then
    echo "Warning: Python 3.11+ is recommended. Current version: $python_version"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements with error handling
echo "Installing requirements..."
if pip install -r requirements.txt; then
    echo "✅ All requirements installed successfully!"
else
    echo "❌ Some packages failed to install. Trying individual installation..."
    
    # Try installing core packages first
    echo "Installing core packages..."
    pip install fastapi uvicorn starlette python-multipart
    pip install pymongo boto3 botocore
    pip install langchain langchain-core langchain-community langchain-openai langchain-aws
    pip install chromadb openai tiktoken
    pip install pandas numpy requests beautifulsoup4
    pip install python-dotenv pydantic
    pip install pillow opencv-python
    pip install unstructured pypdfium2 python-docx python-pptx
    
    echo "Core packages installed. Check for any remaining errors above."
fi

# Test imports
echo "Testing critical imports..."
python3 -c "
try:
    import fastapi
    import pymongo
    import boto3
    import langchain
    import openai
    import pandas
    import requests
    print('✅ All critical imports successful!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "Installation complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
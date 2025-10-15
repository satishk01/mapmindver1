#!/bin/bash

echo "==================================================================="
echo "STEP-BY-STEP PYTHON REQUIREMENTS INSTALLATION"
echo "This script installs packages in the correct order to avoid conflicts"
echo "==================================================================="
echo

# Step 1: Upgrade pip and core tools
echo "Step 1: Upgrading pip and core tools..."
pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "❌ Failed to upgrade pip. Continuing anyway..."
fi
echo

# Step 2: Install typing-extensions first (critical for pydantic)
echo "Step 2: Installing typing-extensions (required for pydantic)..."
pip install "typing-extensions>=4.12.2"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install typing-extensions. This is critical!"
    exit 1
fi
echo "✅ typing-extensions installed successfully"
echo

# Step 3: Install pydantic and pydantic-core
echo "Step 3: Installing pydantic..."
pip install "pydantic>=2.10.0,<3.0.0" "pydantic-settings>=2.7.0,<3.0.0"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install pydantic. This is critical!"
    exit 1
fi
echo "✅ pydantic installed successfully"
echo

# Step 4: Install minimal requirements
echo "Step 4: Installing minimal core requirements..."
pip install -r requirements_minimal.txt
if [ $? -eq 0 ]; then
    echo "✅ Minimal requirements installed successfully!"
    echo
    echo "Testing basic FastAPI import..."
    python -c "from fastapi import FastAPI; print('✅ FastAPI import successful')"
    if [ $? -eq 0 ]; then
        echo
        echo "🎉 BASIC INSTALLATION SUCCESSFUL!"
        echo "You can now start with: uvicorn app:app --reload"
        echo
        echo "To install additional features, run:"
        echo "  pip install langchain langchain-openai chromadb  # For AI features"
        echo "  pip install crawl4ai beautifulsoup4              # For web scraping"
        echo "  pip install unstructured pillow opencv-python    # For document processing"
        exit 0
    else
        echo "❌ FastAPI import failed even with minimal requirements"
        exit 1
    fi
else
    echo "❌ Minimal requirements installation failed"
    exit 1
fi
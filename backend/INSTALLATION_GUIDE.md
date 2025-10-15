# Python 3.11 Installation Guide

This guide will help you install the optimized requirements for Python 3.11 without conflicts.

## Prerequisites

- **Python 3.11+** (recommended)
- **pip** (latest version)
- **Virtual environment** (recommended)

## Quick Installation

### For Linux/macOS:
```bash
# Make script executable
chmod +x install_requirements.sh

# Run installation script
./install_requirements.sh
```

### For Windows:
```cmd
# Run installation script
install_requirements.bat
```

## Manual Installation

### 1. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate.bat
```

### 2. Upgrade pip
```bash
pip install --upgrade pip
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

## Troubleshooting Common Issues

### Issue 1: Package Conflicts
If you encounter package conflicts, try installing in this order:

```bash
# Install core packages first
pip install fastapi==0.115.6 uvicorn==0.34.0
pip install pymongo==4.10.1
pip install boto3==1.35.90 botocore==1.35.90

# Install LangChain packages
pip install langchain==0.3.13
pip install langchain-core==0.3.28
pip install langchain-openai==0.2.14
pip install langchain-aws==0.2.6

# Install remaining packages
pip install -r requirements.txt
```

### Issue 2: Camelot Installation Issues
If `camelot-py` fails to install:

```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install python3-tk ghostscript

# Or on macOS
brew install ghostscript

# Then install camelot
pip install camelot-py[cv]
```

### Issue 3: OpenCV Installation Issues
If `opencv-python` fails:

```bash
# Try installing without GUI support
pip install opencv-python-headless==4.11.0.86
```

### Issue 4: Google Cloud Dependencies
If you don't need multimodal features, comment out Google packages:

```bash
# Comment out these lines in requirements.txt:
# google-generativeai==0.8.3
# google-cloud-aiplatform==1.85.0
# google-auth==2.37.0
```

## Minimal Installation (Text-only features)

If you only need text processing and want to minimize dependencies:

```bash
# Core FastAPI
pip install fastapi==0.115.6 uvicorn==0.34.0 python-multipart==0.0.20

# Database
pip install pymongo==4.10.1

# AWS
pip install boto3==1.35.90 langchain-aws==0.2.6

# LangChain
pip install langchain==0.3.13 langchain-openai==0.2.14 langchain-core==0.3.28

# Document processing
pip install python-docx==1.1.2 python-pptx==1.0.2 pypdf==5.1.0

# Utilities
pip install python-dotenv==1.0.1 pydantic==2.9.2 requests==2.32.3
```

## Verification

Test your installation:

```python
# Test script - save as test_installation.py
try:
    import fastapi
    import pymongo
    import boto3
    import langchain
    import openai
    import pandas
    import requests
    print("✅ All critical packages imported successfully!")
    
    # Test FastAPI
    from fastapi import FastAPI
    app = FastAPI()
    print("✅ FastAPI working")
    
    # Test LangChain
    from langchain_aws import ChatBedrock
    print("✅ LangChain AWS working")
    
    print("🎉 Installation successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check the installation guide for troubleshooting.")
```

Run the test:
```bash
python test_installation.py
```

## Package Breakdown

### Essential Packages (Always Required)
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pymongo` - MongoDB driver
- `boto3` - AWS SDK
- `langchain-*` - LLM framework
- `openai` - OpenAI API
- `python-dotenv` - Environment variables

### Optional Packages
- `google-*` - Google Cloud (multimodal features)
- `opencv-python` - Image processing
- `camelot-py` - PDF table extraction
- `crawl4ai` - Web scraping

### Development Packages (Commented Out)
- `pytest` - Testing
- `rich` - Enhanced logging
- `matplotlib` - Plotting

## Environment Variables

After installation, configure your `.env` file:

```env
# Required
mongo_db_url=mongodb://localhost:27017/mindmap
llm_provider=bedrock  # or openai
aws_region=us-east-1

# Optional (for OpenAI)
openai_api_key=your_key_here

# Optional (for Google multimodal)
gemini_api_key=your_key_here
gcp_project_id=your_project_id
```

## Performance Tips

1. **Use virtual environments** to avoid conflicts
2. **Install only needed packages** - comment out unused ones
3. **Use pip cache** for faster reinstalls: `pip install --cache-dir ~/.pip/cache`
4. **Consider using conda** for complex scientific packages

## Support

If you encounter issues:

1. Check Python version: `python --version`
2. Check pip version: `pip --version`
3. Clear pip cache: `pip cache purge`
4. Try installing in a fresh virtual environment
5. Check the specific error messages for missing system dependencies
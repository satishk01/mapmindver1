@echo off
REM Installation script for Python 3.11 requirements on Windows
REM This script will install the optimized requirements with error handling

echo Installing Python dependencies for Python 3.11...

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements with error handling
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Some packages failed to install. Trying individual installation...
    
    REM Try installing core packages first
    echo Installing core packages...
    pip install fastapi uvicorn starlette python-multipart
    pip install pymongo boto3 botocore
    pip install langchain langchain-core langchain-community langchain-openai langchain-aws
    pip install chromadb openai tiktoken
    pip install pandas numpy requests beautifulsoup4
    pip install python-dotenv pydantic
    pip install pillow opencv-python
    pip install unstructured pypdfium2 python-docx python-pptx
    
    echo Core packages installed. Check for any remaining errors above.
)

REM Test imports
echo Testing critical imports...
python -c "
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

echo Installation complete! Activate the virtual environment with:
echo venv\Scripts\activate.bat
pause
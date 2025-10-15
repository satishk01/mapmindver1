#!/bin/bash

echo "==================================================================="
echo "FIXING CURRENT ENVIRONMENT - typing_extensions Sentinel Error"
echo "This will fix the 'cannot import name Sentinel' error"
echo "==================================================================="
echo

echo "Current Python and pip versions:"
python --version
pip --version
echo

echo "Step 1: Uninstalling problematic packages..."
pip uninstall -y typing-extensions pydantic pydantic-core pydantic-settings
echo

echo "Step 2: Installing typing-extensions with correct version..."
pip install "typing-extensions>=4.12.2"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install typing-extensions"
    exit 1
fi
echo "✅ typing-extensions installed"
echo

echo "Step 3: Installing pydantic with correct version..."
pip install "pydantic>=2.10.0,<3.0.0"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install pydantic"
    exit 1
fi
echo "✅ pydantic installed"
echo

echo "Step 4: Installing pydantic-settings..."
pip install "pydantic-settings>=2.7.0,<3.0.0"
if [ $? -ne 0 ]; then
    echo "❌ Failed to install pydantic-settings"
    exit 1
fi
echo "✅ pydantic-settings installed"
echo

echo "Step 5: Testing the fix..."
python -c "
try:
    from typing_extensions import Sentinel
    print('✅ Sentinel import successful')
    from pydantic import BaseModel
    print('✅ Pydantic import successful')
    from fastapi import FastAPI
    print('✅ FastAPI import successful')
    print('🎉 ALL IMPORTS SUCCESSFUL - Environment fixed!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 ENVIRONMENT SUCCESSFULLY FIXED!"
    echo "You can now run: uvicorn app:app --reload"
else
    echo
    echo "❌ Fix failed. You may need to recreate your virtual environment."
    echo "To recreate:"
    echo "  deactivate"
    echo "  rm -rf venv"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  ./install_step_by_step.sh"
fi
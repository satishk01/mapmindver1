@echo off
echo ===================================================================
echo FIXING CURRENT ENVIRONMENT - typing_extensions Sentinel Error
echo This will fix the 'cannot import name Sentinel' error
echo ===================================================================
echo.

echo Current Python and pip versions:
python --version
pip --version
echo.

echo Step 1: Uninstalling problematic packages...
pip uninstall -y typing-extensions pydantic pydantic-core pydantic-settings
echo.

echo Step 2: Installing typing-extensions with correct version...
pip install "typing-extensions>=4.12.2"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install typing-extensions
    pause
    exit /b 1
)
echo ✅ typing-extensions installed
echo.

echo Step 3: Installing pydantic with correct version...
pip install "pydantic>=2.10.0,<3.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install pydantic
    pause
    exit /b 1
)
echo ✅ pydantic installed
echo.

echo Step 4: Installing pydantic-settings...
pip install "pydantic-settings>=2.7.0,<3.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install pydantic-settings
    pause
    exit /b 1
)
echo ✅ pydantic-settings installed
echo.

echo Step 5: Testing the fix...
python -c "try: from typing_extensions import Sentinel; print('✅ Sentinel import successful'); from pydantic import BaseModel; print('✅ Pydantic import successful'); from fastapi import FastAPI; print('✅ FastAPI import successful'); print('🎉 ALL IMPORTS SUCCESSFUL - Environment fixed!'); except ImportError as e: print(f'❌ Import error: {e}'); exit(1)"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo 🎉 ENVIRONMENT SUCCESSFULLY FIXED!
    echo You can now run: uvicorn app:app --reload
    pause
) else (
    echo.
    echo ❌ Fix failed. You may need to recreate your virtual environment.
    echo To recreate:
    echo   deactivate
    echo   rmdir /s /q venv
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   install_step_by_step.bat
    pause
)
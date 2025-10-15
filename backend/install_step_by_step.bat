@echo off
echo ===================================================================
echo STEP-BY-STEP PYTHON REQUIREMENTS INSTALLATION
echo This script installs packages in the correct order to avoid conflicts
echo ===================================================================
echo.

REM Step 1: Upgrade pip and core tools
echo Step 1: Upgrading pip and core tools...
pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to upgrade pip. Continuing anyway...
)
echo.

REM Step 2: Install typing-extensions first (critical for pydantic)
echo Step 2: Installing typing-extensions ^(required for pydantic^)...
pip install "typing-extensions>=4.12.2"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install typing-extensions. This is critical!
    pause
    exit /b 1
)
echo ✅ typing-extensions installed successfully
echo.

REM Step 3: Install pydantic and pydantic-core
echo Step 3: Installing pydantic...
pip install "pydantic>=2.10.0,<3.0.0" "pydantic-settings>=2.7.0,<3.0.0"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install pydantic. This is critical!
    pause
    exit /b 1
)
echo ✅ pydantic installed successfully
echo.

REM Step 4: Install minimal requirements
echo Step 4: Installing minimal core requirements...
pip install -r requirements_minimal.txt
if %ERRORLEVEL% EQU 0 (
    echo ✅ Minimal requirements installed successfully!
    echo.
    echo Testing basic FastAPI import...
    python -c "from fastapi import FastAPI; print('✅ FastAPI import successful')"
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo 🎉 BASIC INSTALLATION SUCCESSFUL!
        echo You can now start with: uvicorn app:app --reload
        echo.
        echo To install additional features, run:
        echo   pip install langchain langchain-openai chromadb  # For AI features
        echo   pip install crawl4ai beautifulsoup4              # For web scraping
        echo   pip install unstructured pillow opencv-python    # For document processing
        pause
        exit /b 0
    ) else (
        echo ❌ FastAPI import failed even with minimal requirements
        pause
        exit /b 1
    )
) else (
    echo ❌ Minimal requirements installation failed
    pause
    exit /b 1
)
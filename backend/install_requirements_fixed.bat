@echo off
echo Installing Python requirements with dependency conflict fixes...
echo.

echo Attempting installation with fixed requirements...
pip install -r requirements_fixed.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Installation successful with fixed requirements!
    echo.
    goto :end
)

echo.
echo ❌ Fixed requirements failed. Trying optimized requirements...
pip install -r requirements_optimized.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Installation successful with optimized requirements!
    echo.
    goto :end
)

echo.
echo ❌ Both attempts failed. Trying original requirements...
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Installation successful with original requirements!
    echo.
) else (
    echo.
    echo ❌ All installation attempts failed.
    echo Please check the error messages above and try manual installation.
    echo.
    echo Suggested manual steps:
    echo 1. pip install --upgrade pip
    echo 2. pip install pydantic^>=2.10.0,^<3.0.0
    echo 3. pip install -r requirements_fixed.txt --no-deps
    echo 4. pip install -r requirements_fixed.txt
)

:end
echo.
echo Installation process completed.
pause
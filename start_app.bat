@echo off
REM Fashion Recommendation System - Windows Launcher
REM This batch file works on any Windows computer

echo Starting Fashion Recommendation System...
echo.

REM Change to the directory where this batch file is located
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Run the application
echo Running setup verification...
python setup_check.py
if %errorlevel% neq 0 (
    echo.
    echo Setup verification failed. Please check the errors above.
    pause
    exit /b 1
)

echo.
echo Starting GUI application...
python run_demo.py

pause
@echo off
REM --- CONFIGURATION ---
set PYTHON="C:\Users\Booma\anaconda3\envs\BinderProject\python.exe"
REM Change this to the python file you want to run immediately after building
set TEST_SCRIPT="preliminary_test.py"

REM --- 1. Load MSVC Compiler ---
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

REM --- 2. Move to current directory ---
cd /d "%~dp0"

echo.
echo ==========================================
echo [SYSTEMS] 1. Compiling C++ Engine...
echo ==========================================
REM --no-deps: Don't check for numpy/pandas updates every time (Faster)
REM --force-reinstall: Ensures it overwrites the old .pyd file
%PYTHON% -m pip install . --no-deps --force-reinstall

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] C++ Build Failed! Check your syntax.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo [SYSTEMS] 2. Running Python Tests...
echo ==========================================
%PYTHON% %TEST_SCRIPT%

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python Script Crashed!
    pause
    exit /b %errorlevel%
)

echo.
echo [SUCCESS] Engine Built And Verified.
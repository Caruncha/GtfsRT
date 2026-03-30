@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Verification de Python...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python nest pas installe !
    pause
    exit /b 1
)

if not exist ".venv\" (
    echo [2/4] Creation de lenvironnement virtuel...
    python -m venv .venv
)

echo [3/4] Installation des composants...
call ".venv\Scripts\activate.bat"
pip install -r requirements.txt --quiet

echo [4/4] Lancement...
streamlit run app.py --server.headless=false
pause
@echo off
title Analyseur GTFS-RT - Lanceur Automatique
setlocal

:: Dossier de l'application
set APP_DIR=%~dp0
cd /d "%APP_DIR%"

echo ======================================================
echo    BIENVENUE DANS L'ANALYSEUR GTFS-RT
echo ======================================================
echo.

:: 1. Vérification de Python
echo [1/4] Verification de Python...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Python n'est pas installe sur cet ordinateur !
    echo ------------------------------------------------------
    echo Pour que l'app fonctionne, vous devez installer Python :
    echo 1. Allez sur https://www.python.org/downloads/
    echo 2. Cliquez sur "Download Python 3.xx"
    echo 3. IMPORTANT: Cochez la case "Add Python to PATH" lors de l'installation.
    echo ------------------------------------------------------
    pause
    exit /b 1
)
echo OK: Python est present.

:: 2. Creation de l'environnement (si besoin)
if not exist ".venv" (
    echo [2/4] Preparation de l'environnement (cela peut prendre 1-2 min)...
    python -m venv .venv
) else (
    echo [2/4] Environnement deja pret.
)

:: 3. Activation et Installation des dependances
echo [3/4] Mise a jour des composants...
call .venv\Scripts\activate
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo OK: Composants a jour.

:: 4. Fichier de configuration (.env)
if not exist ".env" (
    if exist ".env.example" (
        echo [!] Creation du fichier de configuration par defaut...
        copy ".env.example" ".env" >nul
    )
)

:: 5. Lancement
echo [4/4] Lancement de l'Analyseur...
echo.
echo ======================================================
echo    L'APPLICATION VA S'OUVRIR DANS VOTRE NAVIGATEUR
echo    (Veuillez ne pas fermer cette fenetre noire)
echo ======================================================
echo.

streamlit run app.py --server.headless=false

pause

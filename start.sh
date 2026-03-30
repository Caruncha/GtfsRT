#!/bin/bash

# Configuration
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
APP_FILE="app.py"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Démarrage de l'Analyseur GTFS-RT...${NC}"

# 1. Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}❌ Python 3 n'est pas installé. Veuillez l'installer.${NC}"
    exit 1
fi

# 2. Créer l'environnement virtuel si nécessaire
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}📦 Création de l'environnement virtuel (première fois)...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# 3. Activer l'environnement virtuel
source "$VENV_DIR/bin/activate"

# 4. Installer/Mettre à jour les dépendances
echo -e "${BLUE}🔧 Vérification des dépendances...${NC}"
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS_FILE" --quiet

# 5. Gérer le fichier .env
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        echo -e "${YELLOW}⚠️  Fichier .env manquant. Création à partir de l'exemple...${NC}"
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo -e "${YELLOW}👉 Pensez à éditer le fichier .env pour ajouter vos clés API si nécessaire.${NC}"
    fi
fi

# 6. Lancer l'application
echo -e "${GREEN}✨ Lancement de l'application !${NC}"
echo -e "${BLUE}Le navigateur va s'ouvrir automatiquement...${NC}"
streamlit run "$APP_FILE" --server.headless=false

# 🚌 Analyseur GTFS‑realtime : TripUpdates

Outil Streamlit pour analyser un fichier **GTFS‑rt TripUpdates (.pb)** et, optionnellement, un **GTFS statique** :
- volumes, annulations complètes/partielles
- qualité des timestamps (ms→s, monotonicité, arrival ≤ departure)
- incohérences (stop_id / trip_id inconnus…)
- **comparaison prédiction vs horaire planifié** (si `gtfs.zip` fourni)
- exports : CSV/JSON/MD/HTML, graphiques interactifs

## 🚀 Déploiement (Streamlit Community Cloud)

1. Fork/cloner ce repo sur **GitHub**.  
2. Aller sur **https://streamlit.io/cloud** → *Sign in with GitHub* → *New app*  
3. Sélectionner ce repo, la branche, et le fichier d’entrée `app.py`, puis **Deploy**.

> Docs : [Streamlit Community Cloud](https://streamlit.io/cloud) · [Guide de déploiement](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## ▶️ Utilisation locale

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py

# app.py (optimisé mémoire)
# -*- coding: utf-8 -*-
import os
import io
import json
import tempfile
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from gtfsrt_tripupdates_report import (
    analyze_tripupdates,
    load_static_gtfs,
    extract_ids_from_tripupdates,
    write_reports,
)

st.set_page_config(page_title="🚌 Analyseur TripUpdates (optimisé)", layout="wide")
st.title("🚌 Analyseur GTFS‑realtime : TripUpdates (optimisé mémoire)")

with st.expander("Mode d'emploi", expanded=False):
    st.markdown(
        """
        **Étapes**
        1) Uploadez un **TripUpdates (.pb)** et, en option, un **GTFS statique (.zip)**.
        2) L'app extrait d'abord **les `trip_id` et `stop_id` présents dans le TripUpdates**.
        3) Elle charge **uniquement les parties utiles du GTFS**, **par morceaux (chunks)**.
        4) Les **résultats complets** sont fournis **dans un ZIP téléchargeable** (aucune table massive affichée).

        **Pourquoi c'est plus robuste ?**
        - Lecture **streamée** (`chunksize`) des CSV du GTFS.
        - **Filtrage ciblé** (trips/stop_times/stops/routes) à partir des ids du RT.
        - L’UI affiche une **synthèse légère**; tout le détail est dans un **fichier ZIP**.
        """
    )

with st.sidebar:
    st.header("Fichiers")
    tu_file = st.file_uploader("TripUpdates (protobuf .pb)", type=["pb"])  # obligé .pb pour rester léger
    gtfs_zip = st.file_uploader("GTFS statique (.zip) (optionnel)", type=["zip"])
    st.divider()
    st.header("Options")
    chunksize = st.number_input("Taille de chunk (lignes)", min_value=50_000, max_value=1_000_000, value=200_000, step=50_000)
    fast_mode = st.checkbox("Charger uniquement les voyages présents dans le TripUpdates", value=True)
    run_btn = st.button("Analyser", type="primary")

@st.cache_data(show_spinner=False)
def _run_analysis_cached(tu_bytes: bytes, gtfs_bytes: Optional[bytes], fast: bool, chunksize: int):
    """Effectue toute l'analyse en FS temporaire puis renvoie (analysis_dict, outputs_zip_bytes)."""
    with tempfile.TemporaryDirectory() as tmp:
        tu_path = os.path.join(tmp, 'tripupdates.pb')
        with open(tu_path, 'wb') as f:
            f.write(tu_bytes)

        trip_ids, stop_ids = extract_ids_from_tripupdates(tu_path)

        static_gtfs = {"stops": pd.DataFrame(), "trips": pd.DataFrame(), "stop_times": pd.DataFrame(), "routes": pd.DataFrame(), "agency": pd.DataFrame()}
        gtfs_path = None
        if gtfs_bytes is not None:
            gtfs_path = os.path.join(tmp, 'gtfs.zip')
            with open(gtfs_path, 'wb') as f:
                f.write(gtfs_bytes)
            static_gtfs = load_static_gtfs(gtfs_path, trip_ids=trip_ids if fast else None, stop_ids=stop_ids if fast else None, fast=fast, chunksize=chunksize)

        analysis = analyze_tripupdates(tu_path, static_gtfs)

        # Génère un répertoire de sortie et un zip
        out_dir = os.path.join(tmp, 'report')
        os.makedirs(out_dir, exist_ok=True)
        outputs = write_reports(analysis, out_dir, tu_path, gtfs_path)
        zip_path = os.path.join(tmp, 'gtfsrt_report.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in os.listdir(out_dir):
                z.write(os.path.join(out_dir, p), arcname=p)
        with open(zip_path, 'rb') as f:
            zip_bytes = f.read()
    return analysis, zip_bytes

if run_btn:
    if not tu_file:
        st.error("Merci de fournir un fichier TripUpdates (.pb)")
        st.stop()
    with st.spinner("Analyse en cours…"):
        try:
            analysis, zip_bytes = _run_analysis_cached(tu_file.getvalue(), gtfs_zip.getvalue() if gtfs_zip else None, fast_mode, int(chunksize))
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("Analyse terminée ✅")

    # Courte synthèse uniquement (pas de gros DataFrames)
    meta = analysis.get('meta', {})
    summary = analysis.get('summary', {})
    tsq = analysis.get('ts_quality', {})
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trips (total)", f"{summary.get('total_trips',0):,}".replace(',', ' '))
    c2.metric("Trips annulés", f"{summary.get('canceled_trips',0):,}".replace(',', ' '))
    c3.metric("STU (lignes)", f"{int(analysis.get('stu_df', pd.DataFrame()).shape[0]):,}".replace(',', ' '))
    c4.metric("% STU avec heures", f"{tsq.get('stus_with_any_time_pct',0):.1f}%")

    # Téléchargement unique
    st.download_button("📦 Télécharger le rapport (ZIP)", data=zip_bytes, file_name="gtfsrt_report.zip", mime="application/zip")

    with st.expander("Détails (JSON)"):
        st.json({
            "meta": meta,
            "summary": summary,
            "ts_quality": tsq,
            "counts": {
                "trips_rows": int(analysis.get('trips_df', pd.DataFrame()).shape[0]),
                "stop_time_updates": int(analysis.get('stu_df', pd.DataFrame()).shape[0]),
                "anomalies": int(analysis.get('anomalies', pd.DataFrame()).shape[0]),
                "schedule_compare_rows": int(analysis.get('schedule_compare_df', pd.DataFrame()).shape[0])
            }
        })
else:
    st.info("Charge un **TripUpdates (.pb)** puis clique **Analyser**. Optionnellement, ajoute un **GTFS (.zip)**.")

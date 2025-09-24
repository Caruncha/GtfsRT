# -*- coding: utf-8 -*-
import os
import io
import json
import tempfile
from typing import Optional, Dict
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from gtfsrt_tripupdates_report import (
    analyze_tripupdates,
    load_static_gtfs,
    validate_cancellations_against_tripupdates, # <-- nécessite la mise à jour du module lib
)

# Altair : retire la limite 5k lignes pour éviter MaxRowsError
alt.data_transformers.enable('default', max_rows=None)

st.set_page_config(page_title="🚌 Analyseur TripUpdates", layout="wide")
st.title("🚌 Analyseur GTFS‑realtime : TripUpdates")
st.write(
    "Charge un fichier **TripUpdates (Protocol Buffer)** (extension libre) et, optionnellement, un **GTFS statique** "
    "pour des validations avancées et la comparaison au planifié. Utilise les filtres pour explorer, et télécharge les résultats."
)

# ----------------------------- Uploaders & options (sidebar) ------------------
with st.sidebar:
    st.header("Fichiers")
    tu_file = st.file_uploader(
        "Fichier TripUpdates (Protocol Buffer GTFS‑rt – extension quelconque)",
        type=None # accepte tout
    )
    gtfs_file = st.file_uploader("GTFS statique (zip) (optionnel)", type=["zip"])
    st.divider()
    st.header("Options")
    st.caption("Astuce : pour de gros fichiers, augmentez la taille via `.streamlit/config.toml` → [server] maxUploadSize = 200")
    run_button = st.button("Analyser", type="primary")

# -------------------------------- Cache d'analyse -----------------------------
@st.cache_data(show_spinner=False)
def run_analysis_cached(tu_bytes: bytes, gtfs_bytes: Optional[bytes]):
    """
    Exécute l'analyse en environnement temporaire et met en cache le résultat pour
    ne pas recalculer à chaque rechargement de l'app.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tu_path = os.path.join(tmpdir, "tripupdates.any")  # extension libre
        with open(tu_path, "wb") as f:
            f.write(tu_bytes)

        gtfs_path = None
        static_gtfs = {
            "stops": pd.DataFrame(),
            "trips": pd.DataFrame(),
            "stop_times": pd.DataFrame(),
            "routes": pd.DataFrame(),
            "agency": pd.DataFrame(),
        }
        if gtfs_bytes:
            gtfs_path = os.path.join(tmpdir, "gtfs.zip")
            with open(gtfs_path, "wb") as f:
                f.write(gtfs_bytes)
            static_gtfs = load_static_gtfs(gtfs_path)

        analysis = analyze_tripupdates(tu_path, static_gtfs)
        return analysis, static_gtfs

# ----------------------------- Helpers UI / Data ------------------------------
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _sr_label_map_stu():
    # STU schedule_relationship: 0=SCHEDULED, 1=SKIPPED, 2=NO_DATA
    return {0: "SCHEDULED", 1: "SKIPPED", 2: "NO_DATA"}


def _sr_label_map_trip():
    # TripDescriptor schedule_relationship: 0=SCHEDULED, 1=ADDED, 2=UNSCHEDULED, 3=CANCELED
    return {0: "SCHEDULED", 1: "ADDED", 2: "UNSCHEDULED", 3: "CANCELED"}


def _event_epoch_series(stu_df: pd.DataFrame) -> pd.Series:
    """
    Horodatage d’événement pour l’axe temps : priorise departure_time, sinon arrival_time.
    """
    if stu_df.empty:
        return pd.Series(dtype="float")
    t = stu_df["departure_time"].where(stu_df["departure_time"].notna(), stu_df["arrival_time"])
    return pd.to_numeric(t, errors="coerce")


def _add_local_bin10(df: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Ajoute :
    - bin10_minute (Int64) : minute depuis minuit locale arrondie à 10 min (0..1430)
    - bin10_label (string): libellé 'HH:MM' de la tranche
    """
    s = _event_epoch_series(df)
    out = df.copy()
    if s.empty:
        out["bin10_minute"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["bin10_label"] = pd.Series([pd.NA] * len(out), dtype="string")
        return out
    try:
        dt_local = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(ZoneInfo(tz_str))
    except Exception:
        dt_local = pd.to_datetime(s, unit="s", utc=True)
    dt10 = dt_local.dt.floor("10min")
    minute_of_day = dt10.dt.hour * 60 + dt10.dt.minute
    out["bin10_minute"] = pd.to_numeric(minute_of_day, errors="coerce").astype("Int64")
    out["bin10_label"] = dt10.dt.strftime("%H:%M").astype("string")
    return out

# --- Helpers annulations (10 min) ---
def _hms_to_seconds(hms: Optional[str]) -> Optional[int]:
    """Accepte HH:MM ou HH:MM:SS (seconds facultatives)."""
    if hms is None or pd.isna(hms):
        return None
    try:
        parts = [int(x) for x in str(hms).split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m = parts
            s = 0
        else:
            return None
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def _service_midnight_epoch(start_date: str, tz_str: str) -> Optional[int]:
    """start_date (YYYYMMDD) → epoch UTC de minuit local."""
    try:
        y, m, d = int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8])
        return int(pd.Timestamp(y, m, d, 0, 0, 0, tz=ZoneInfo(tz_str)).timestamp())
    except Exception:
        return None


def _trips_binning_for_cancellations(trips_view: pd.DataFrame, stu_view: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Retourne colonnes: bin10_minute (Int64), bin10_label (string), kind {"Annulations complètes","Annulations partielles"}, compte
    Binning 10 min basé sur : min(event_time) par trip (dep/arr STU), sinon start_date+start_time.
    """
    if trips_view.empty:
        return pd.DataFrame(columns=["bin10_minute", "bin10_label", "kind", "compte"])

    # min(event_time) par trip_key
    if not stu_view.empty:
        sv = stu_view.copy()
        sv["event_time"] = sv["departure_time"].where(sv["departure_time"].notna(), sv["arrival_time"])
        per_trip_event = (
            sv.dropna(subset=["event_time"])
            .groupby("trip_key")["event_time"]
            .min()
            .rename("trip_event_epoch")
        )
    else:
        per_trip_event = pd.Series(dtype="float", name="trip_event_epoch")

    tv = trips_view.copy().merge(per_trip_event, left_on="trip_key", right_index=True, how="left")
    # fallback start_date + start_time
    missing_event = tv["trip_event_epoch"].isna()
    if "start_date" in tv.columns and "start_time" in tv.columns:
        midnight_epoch = tv.loc[missing_event, "start_date"].apply(
            lambda sd: _service_midnight_epoch(sd, tz_str) if isinstance(sd, str) and len(sd) == 8 else None
        )
        start_sec = tv.loc[missing_event, "start_time"].apply(_hms_to_seconds)
        fallback_epoch = []
        for me, ss in zip(midnight_epoch.tolist(), start_sec.tolist()):
            fallback_epoch.append(None if me is None or ss is None else me + ss)
        tv.loc[missing_event, "trip_event_epoch"] = pd.to_numeric(pd.Series(fallback_epoch), errors="coerce")

    # epoch → local → bin 10 min
    dt_local = pd.to_datetime(tv["trip_event_epoch"], unit="s", utc=True, errors="coerce")
    try:
        dt_local = dt_local.dt.tz_convert(ZoneInfo(tz_str))
    except Exception:
        pass
    dt10 = dt_local.dt.floor("10min")
    tv["bin10_minute"] = pd.to_numeric(dt10.dt.hour * 60 + dt10.dt.minute, errors="coerce").astype("Int64")
    tv["bin10_label"] = dt10.dt.strftime("%H:%M").astype("string")

    # jeux d'annulation
    fully_canceled_keys = set(tv.loc[tv["trip_schedule_relationship"] == 3, "trip_key"])
    if not stu_view.empty:
        skipped_keys = set(stu_view.loc[stu_view["stu_schedule_relationship"] == 1, "trip_key"])
    else:
        skipped_keys = set()
    partial_keys = skipped_keys - fully_canceled_keys

    tv_non_na = tv.dropna(subset=["bin10_minute"])
    series_full = (
        tv_non_na.loc[tv_non_na["trip_key"].isin(fully_canceled_keys)]
        .groupby(["bin10_minute", "bin10_label"]).size().rename("compte")
    )
    series_part = (
        tv_non_na.loc[tv_non_na["trip_key"].isin(partial_keys)]
        .groupby(["bin10_minute", "bin10_label"]).size().rename("compte")
    )

    df_full = series_full.reset_index(); df_full["kind"] = "Annulations complètes"
    df_part = series_part.reset_index(); df_part["kind"] = "Annulations partielles"
    if not df_full.empty or not df_part.empty:
        out = pd.concat([df_full, df_part], ignore_index=True, sort=False)
    else:
        out = pd.DataFrame(columns=["bin10_minute", "bin10_label", "compte", "kind"])
    return out.sort_values(["bin10_minute", "kind"])[["bin10_minute", "bin10_label", "kind", "compte"]]


# --------------- Histogrammes robustes (anti-overflow) -----------------------
def _safe_histogram(series: pd.Series, bins: int = 60, clip_abs_minutes: Optional[int] = 1440) -> pd.DataFrame:
    """
    Convertit en float64, retire NaN/Inf, clippe les extrêmes (±clip_abs_minutes),
    puis calcule un histogramme. Retourne {bin_center, count}.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    s = s[np.isfinite(s)]
    if s.size == 0:
        return pd.DataFrame(columns=["bin_center", "count"])
    if clip_abs_minutes is not None:
        s = np.clip(s, -abs(clip_abs_minutes), abs(clip_abs_minutes))
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return pd.DataFrame({"bin_center": centers, "count": counts})


# ------ Helpers schéma minimal pour éviter les KeyError sur merge/select ------
def _ensure_min_schema(df: pd.DataFrame, required: Dict[str, str]) -> pd.DataFrame:
    """
    Ajoute les colonnes manquantes avec le dtype indiqué si nécessaire.
    Exemple required: {"trip_key": "string", "route_id": "string", "trip_id": "string"}
    """
    out = df.copy()
    for col, dtype in required.items():
        if col not in out.columns:
            try:
                out[col] = pd.Series([], dtype=dtype)
            except Exception:
                out[col] = pd.Series([], dtype="object")
    return out


# --------------------------------- Analyse -----------------------------------
if run_button and tu_file is not None:
    try:
        with st.spinner("Analyse en cours…"):
            analysis, static_gtfs = run_analysis_cached(
                tu_file.getvalue(),
                gtfs_file.getvalue() if gtfs_file else None
            )
    except Exception as e:
        st.error("❌ Le fichier fourni ne semble pas être un **GTFS‑rt TripUpdates** valide (Protocol Buffer).")
        st.caption(f"Détail technique : {type(e).__name__}: {e}")
        st.stop()

    st.success("Analyse terminée ✅")
    st.write(f"Fichier chargé : **{tu_file.name}**")

    # Détecte tz par défaut à partir du GTFS s'il existe
    tz_input = "America/Montreal"
    if static_gtfs and notPar

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
    validate_cancellations_against_tripupdates,  # <-- nécessite la mise à jour du module lib
)

# Altair : retire la limite 5k lignes pour éviter MaxRowsError
alt.data_transformers.enable('default', max_rows=None)

st.set_page_config(page_title="🚌 Analyseur TripUpdates", layout="wide")
st.title("🚌 Analyseur GTFS‑realtime : TripUpdates")
st.write(
    "Charge un fichier **TripUpdates (Protocol Buffer)** (extension libre) et, optionnellement, un **GTFS statique** "
    "pour des validations avancées et la comparaison au planifié. Utilise les filtres pour explorer, et télécharge les résultats."
)

# -----------------------------------------------------------------------------
# Uploaders & options (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Fichiers")
    tu_file = st.file_uploader(
        "Fichier TripUpdates (Protocol Buffer GTFS‑rt – extension quelconque)",
        type=None  # accepte tout
    )
    gtfs_file = st.file_uploader("GTFS statique (zip) (optionnel)", type=["zip"])

    st.divider()
    st.header("Options")
    st.caption("Astuce : pour de gros fichiers, augmentez la taille via `.streamlit/config.toml` → [server] maxUploadSize = 200")
    run_button = st.button("Analyser", type="primary")

# -----------------------------------------------------------------------------
# Cache d'analyse
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helpers UI / Data
# -----------------------------------------------------------------------------
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
    if hms is None or pd.isna(hms):
        return None
    try:
        h, m, s = [int(x) for x in str(hms).split(":")]
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

# --- Histogrammes robustes (anti-overflow) ---
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

# --- Helpers schéma minimal pour éviter les KeyError sur merge/select ---
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

# --- Helper couleurs (carte) ---
def _normalize_hex_color(s: str, default: str = "#888888") -> str:
    """
    Normalise une couleur hex GTFS en '#RRGGBB'.
    - Accepte 'RRGGBB', '#RRGGBB', 'RGB' ; sinon fallback default.
    """
    if s is None or pd.isna(s):
        return default
    t = str(s).strip().lstrip("#")
    if not t:
        return default
    if len(t) == 3 and all(c in "0123456789abcdefABCDEF" for c in t):
        t = "".join(ch * 2 for ch in t)
    t = t.upper()
    if len(t) >= 6 and all(c in "0123456789ABCDEF" for c in t[:6]):
        return "#" + t[:6]
    return default

# -----------------------------------------------------------------------------
# Analyse
# -----------------------------------------------------------------------------
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
    if static_gtfs and not static_gtfs.get("agency", pd.DataFrame()).empty:
        try:
            tz_detected = str(static_gtfs["agency"].iloc[0].get("agency_timezone")) or tz_input
            tz_input = tz_detected
        except Exception:
            pass

    trips_df = analysis["trips_df"].copy()
    stu_df = analysis["stu_df"].copy()
    anomalies_df = analysis["anomalies"].copy()
    sched_df = analysis.get("schedule_compare_df", pd.DataFrame())
    schedule_stats = analysis.get("schedule_stats", {})

    # Garantit le schéma minimal (même si DataFrame vide)
    trips_df = _ensure_min_schema(trips_df, {
        "trip_key": "string",
        "route_id": "string",
        "trip_id": "string",
        "start_date": "string",
        "start_time": "string",
        "trip_schedule_relationship": "Int64",
    })
    stu_df = _ensure_min_schema(stu_df, {
        "trip_key": "string",
        "route_id": "string",
        "trip_id": "string",
        "stop_id": "string",
        "stop_sequence": "Int64",
        "stu_schedule_relationship": "Int64",
        "arrival_time": "float",
        "departure_time": "float",
    })

    # ----------------------------- Filtres
    st.sidebar.header("Filtres")
    route_opts = sorted([r for r in trips_df["route_id"].dropna().unique().tolist() if r != ""])
    route_sel = st.sidebar.multiselect("Filtrer par route_id", options=route_opts, default=[])
    map_trip = _sr_label_map_trip()
    trip_types = [(0, "SCHEDULED"), (1, "ADDED"), (2, "UNSCHEDULED"), (3, "CANCELED")]
    trip_sel_labels = st.sidebar.multiselect(
        "Types de voyages", options=[lbl for _, lbl in trip_types], default=[lbl for _, lbl in trip_types]
    )
    trip_sel_ids = {k for k, v in trip_types if v in set(trip_sel_labels)}
    map_stu = _sr_label_map_stu()
    stu_types = [(0, "SCHEDULED"), (1, "SKIPPED"), (2, "NO_DATA")]
    stu_sel_labels = st.sidebar.multiselect(
        "Types d'arrêts (STU)", options=[lbl for _, lbl in stu_types], default=[lbl for _, lbl in stu_types]
    )
    stu_sel_ids = {k for k, v in stu_types if v in set(stu_sel_labels)}
    trip_id_query = st.sidebar.text_input("Recherche trip_id (contient)", value="")

    trips_view = trips_df.copy()
    if route_sel:
        trips_view = trips_view[trips_view["route_id"].isin(route_sel)]
    if trip_sel_ids:
        trips_view = trips_view[trips_view["trip_schedule_relationship"].isin(trip_sel_ids)]
    if trip_id_query:
        trips_view = trips_view[trips_view["trip_id"].fillna("").str.contains(trip_id_query, case=False)]

    # Merge défensif pour enrichir STU (route_id, trip_id)
    merge_cols = [c for c in ["trip_key", "route_id", "trip_id"] if c in trips_df.columns]
    if "trip_key" in stu_df.columns and "trip_key" in merge_cols:
        stu_view = stu_df.merge(trips_df[merge_cols], on="trip_key", how="left", suffixes=("", "_t"))
    else:
        stu_view = stu_df.copy()
    if route_sel:
        stu_view = stu_view[stu_view["route_id"].isin(route_sel)]
    if trip_id_query:
        # après merge, le trip_id original peut être suffixé _t ; on filtre sur les deux si existants
        col_candidates = [c for c in ["trip_id", "trip_id_t"] if c in stu_view.columns]
        if col_candidates:
            mask = False
            for c in col_candidates:
                mask = (mask | stu_view[c].fillna("").str.contains(trip_id_query, case=False))
            stu_view = stu_view[mask]
    if stu_sel_ids:
        stu_view = stu_view[stu_view["stu_schedule_relationship"].isin(stu_sel_ids)]

    # ----------------------------- KPIs
    st.subheader("Résumé")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_trips = int(trips_view["trip_key"].nunique())
    canceled_trips = int((trips_view["trip_schedule_relationship"] == 3).sum())
    added_trips = int((trips_view["trip_schedule_relationship"] == 1).sum())
    unscheduled_trips = int((trips_view["trip_schedule_relationship"] == 2).sum())

    sk = stu_view[stu_view["stu_schedule_relationship"] == 1]  # SKIPPED
    trips_with_sk = set(sk["trip_key"]) if not sk.empty else set()
    fully_canceled = set(trips_view.loc[trips_view["trip_schedule_relationship"] == 3, "trip_key"])
    partial_canceled_trips = len(trips_with_sk - fully_canceled)
    canceled_stops_total = int(len(sk))

    col1.metric("Voyages (total)", f"{total_trips:,}".replace(",", " "))
    col2.metric("Voyages annulés", f"{canceled_trips:,}".replace(",", " "))
    col3.metric("Voyages ajoutés", f"{added_trips:,}".replace(",", " "))
    col4.metric("Voyages non planifiés", f"{unscheduled_trips:,}".replace(",", " "))
    col5.metric("Voyages partiellement annulés", f"{partial_canceled_trips:,}".replace(",", " "))

    # ----------------------------- Graphiques récapitulatifs
    st.markdown("### ### Graphiques récapitulatifs")
    charts_col1, charts_col2 = st.columns(2)
    with charts_col1:
        summary_counts = pd.DataFrame([
            {"type": "CANCELED (trips)", "val": canceled_trips},
            {"type": "ADDED (trips)", "val": added_trips},
            {"type": "UNSCHEDULED (trips)", "val": unscheduled_trips},
            {"type": "PARTIAL CANCELED (trips)", "val": partial_canceled_trips},
            {"type": "SKIPPED (stops)", "val": canceled_stops_total},
        ])
        chart_summary = (
            alt.Chart(summary_counts)
               .mark_bar()
               .encode(
                   x=alt.X("type:N", sort="-y", title="Catégorie"),
                   y=alt.Y("val:Q", title="Nombre"),
                   tooltip=["type", "val"]
               )
               .properties(height=300)
        )
        st.altair_chart(chart_summary, use_container_width=True)

    with charts_col2:
        if not stu_view.empty:
            dist = (
                stu_view
                .assign(type=lambda d: d["stu_schedule_relationship"].map(_sr_label_map_stu()).fillna("AUTRE"))
                .groupby("type").size().reset_index(name="compte")
            )
            chart_dist = (
                alt.Chart(dist)
                   .mark_arc(innerRadius=60)
                   .encode(theta="compte:Q", color="type:N", tooltip=["type", "compte"])
                   .properties(height=300)
            )
            st.altair_chart(chart_dist, use_container_width=True)
        else:
            st.info("Aucun StopTimeUpdate après filtres.")

    # ----------------------------- Carte des trajets (Altair)
    st.markdown("### 🗺️ Carte des trajets (Altair)")

    stops_static = static_gtfs.get("stops", pd.DataFrame())
    stop_times_static = static_gtfs.get("stop_times", pd.DataFrame())
    trips_static = static_gtfs.get("trips", pd.DataFrame())
    routes_static = static_gtfs.get("routes", pd.DataFrame())

    missing_bits = []
    if stops_static.empty or not {"stop_id", "stop_lat", "stop_lon"}.issubset(stops_static.columns):
        missing_bits.append("`stops.txt` avec `stop_lat/stop_lon`")
    if stop_times_static.empty or not {"trip_id", "stop_id", "stop_sequence"}.issubset(stop_times_static.columns):
        missing_bits.append("`stop_times.txt` (trip_id, stop_id, stop_sequence)")
    if trips_static.empty or "route_id" not in trips_static.columns:
        missing_bits.append("`trips.txt` (trip_id, route_id)")
    if routes_static.empty or "route_id" not in routes_static.columns:
        missing_bits.append("`routes.txt`")

    if missing_bits:
        st.info("Carte indisponible : " + " ; ".join(missing_bits))
    else:
        trip_opts = sorted([t for t in trips_view["trip_id"].dropna().unique().tolist() if t != ""])
        default_selection = trip_opts[:5]
        trip_sel = st.multiselect(
            "Trip_id à cartographier",
            options=trip_opts,
            default=default_selection,
            help="Sélectionne un ou plusieurs voyages à tracer sur la carte."
        )
        max_trips = st.slider("Nombre maximal de trips représentés", 1, 50, max(1, len(trip_sel) or 5))
        if trip_sel:
            trip_sel = trip_sel[:max_trips]

        if not trip_sel:
            st.info("Sélectionne au moins un `trip_id` pour afficher la carte.")
        else:
            paths = (
                stop_times_static[stop_times_static["trip_id"].isin(trip_sel)]
                .merge(stops_static[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
                .merge(trips_static[["trip_id", "route_id"]], on="trip_id", how="left")
                .merge(routes_static[["route_id", "route_color"]], on="route_id", how="left")
            )

            paths = paths.dropna(subset=["stop_lat", "stop_lon"])
            if "stop_sequence" in paths.columns:
                # s'assure que l'ordre est numérique pour le tracé
                paths["stop_sequence"] = pd.to_numeric(paths["stop_sequence"], errors="coerce")
            paths["route_color"] = paths.get("route_color", pd.Series([], dtype="object")).apply(
                lambda v: _normalize_hex_color(v, default="#4C78A8")
            )

            # Points annulés (SKIPPED) depuis le RT
            sk_points = pd.DataFrame(columns=["trip_id", "stop_id", "stop_lat", "stop_lon", "stop_sequence"])
            if not stu_view.empty and "stu_schedule_relationship" in stu_view.columns:
                sk = stu_view[
                    (stu_view["trip_id"].isin(trip_sel)) &
                    (stu_view["stu_schedule_relationship"] == 1)  # 1 == SKIPPED
                ].copy()
                if not sk.empty:
                    sk_points = sk.merge(stops_static[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
                    if "stop_sequence" not in sk_points.columns or sk_points["stop_sequence"].isna().all():
                        sk_points = sk_points.merge(
                            stop_times_static[["trip_id", "stop_id", "stop_sequence"]],
                            on=["trip_id", "stop_id"],
                            how="left",
                            suffixes=("", "_sched")
                        )
                    if "stop_sequence" in sk_points.columns:
                        sk_points["stop_sequence"] = pd.to_numeric(sk_points["stop_sequence"], errors="coerce")

            base = alt.Chart(paths).project("mercator").properties(height=520)

            lines = (
                base
                .mark_line(strokeWidth=2, opacity=0.85)
                .encode(
                    longitude="stop_lon:Q",
                    latitude="stop_lat:Q",
                    detail="trip_id:N",
                    order="stop_sequence:Q",
                    color=alt.Color("route_color:N", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip("trip_id:N", title="Trip"),
                        alt.Tooltip("route_id:N", title="Route"),
                        alt.Tooltip("stop_id:N", title="Arrêt"),
                        alt.Tooltip("stop_sequence:Q", title="Ordre")
                    ]
                )
            )

            points_all = (
                base
                .mark_point(filled=True, color="#999999", size=20, opacity=0.6)
                .encode(
                    longitude="stop_lon:Q",
                    latitude="stop_lat:Q",
                    tooltip=[
                        alt.Tooltip("trip_id:N", title="Trip"),
                        alt.Tooltip("stop_id:N", title="Arrêt"),
                        alt.Tooltip("stop_sequence:Q", title="Ordre")
                    ]
                )
            )

            layer_sk = None
            if not sk_points.empty:
                layer_sk = (
                    alt.Chart(sk_points)
                    .project("mercator")
                    .mark_point(filled=True, color="#E45756", size=90, opacity=0.95)
                    .encode(
                        longitude="stop_lon:Q",
                        latitude="stop_lat:Q",
                        tooltip=[
                            alt.Tooltip("trip_id:N", title="Trip"),
                            alt.Tooltip("stop_id:N", title="Arrêt (SKIPPED)"),
                            alt.Tooltip("stop_sequence:Q", title="Ordre")
                        ]
                    )
                )

            chart_map = lines + points_all
            if layer_sk is not None:
                chart_map = chart_map + layer_sk

            st.altair_chart(chart_map.interactive(), use_container_width=True)

    # ----------------------------- Courbe des passages (10 min)
    st.markdown("### ### Courbe — Passages par tranches de 10 minutes (par type d'arrêt)")
    if not stu_view.empty:
        try:
            stu_10 = _add_local_bin10(stu_view, tz_input)
        except Exception:
            stu_10 = _add_local_bin10(stu_view, "UTC")
        series_passages = (
            stu_10
            .assign(type=lambda d: d["stu_schedule_relationship"].map(_sr_label_map_stu()).fillna("AUTRE"))
            .dropna(subset=["bin10_minute"])
            .groupby(["bin10_minute", "bin10_label", "type"])
            .size()
            .reset_index(name="compte")
            .sort_values(["bin10_minute", "type"])
        )
        if not series_passages.empty:
            line_passages = (
                alt.Chart(series_passages)
                   .mark_line(point=True)
                   .encode(
                       x=alt.X(
                           "bin10_label:N",
                           sort=alt.SortField(field="bin10_minute", order="ascending"),
                           title="Heure locale (10 min)"),
                       y=alt.Y("compte:Q", title="Nombre"),
                       color=alt.Color("type:N", title="Type d'arrêt"),
                       tooltip=["bin10_label", "type", "compte"]
                   )
                   .properties(height=320)
            )
            st.altair_chart(line_passages, use_container_width=True)
        else:
            st.info("Pas de données suffisantes pour la courbe des passages.")
    else:
        st.info("Aucun STU à représenter pour la courbe des passages.")

    # ----------------------------- Courbes des annulations de voyages (10 min)
    st.markdown("### ### Courbes — Annulations de voyages (complètes vs partielles, 10 min)")
    series_cancel = _trips_binning_for_cancellations(trips_view, stu_view, tz_input)
    if not series_cancel.empty:
        line_cancel = (
            alt.Chart(series_cancel)
               .mark_line(point=True)
               .encode(
                   x=alt.X(
                       "bin10_label:N",
                       sort=alt.SortField(field="bin10_minute", order="ascending"),
                       title="Heure locale (10 min)"),
                   y=alt.Y("compte:Q", title="Nombre de voyages"),
                   color=alt.Color("kind:N", title="Type d'annulation"),
                   tooltip=["bin10_label", "kind", "compte"]
               )
               .properties(height=320)
        )
        st.altair_chart(line_cancel, use_container_width=True)
    else:
        st.info("Pas de voyages annulés (complets ou partiels) dans la sélection.")

    # ----------------------------- Top 20 arrêts annulés
    st.markdown("### ### Top 20 des arrêts annulés (SKIPPED)")
    sk_only = stu_view[stu_view["stu_schedule_relationship"] == 1]
    if not sk_only.empty:
        top_sk = (
            sk_only.groupby("stop_id").size().reset_index(name="compte")
            .sort_values("compte", ascending=False)
            .head(20)
        )
        stops_static = static_gtfs.get("stops", pd.DataFrame())
        if not stops_static.empty and "stop_name" in stops_static.columns:
            top_sk = top_sk.merge(stops_static[["stop_id", "stop_name"]], on="stop_id", how="left")
        chart_top = (
            alt.Chart(top_sk)
               .mark_bar()
               .encode(
                   x=alt.X("compte:Q", title="Nombre de SKIPPED"),
                   y=(
                       alt.Y("stop_name:N", sort="-x", title="Arrêt").axis(labelLimit=250)
                       if "stop_name" in top_sk.columns else
                       alt.Y("stop_id:N", sort="-x", title="stop_id")
                   ),
                   tooltip=top_sk.columns.tolist()
               )
               .properties(height=400)
        )
        st.altair_chart(chart_top, use_container_width=True)
    else:
        st.info("Aucun arrêt annulé dans la sélection.")

    # ----------------------------- Comparaison au planifié (si GTFS fourni)
    st.markdown("### ### Écart vs horaire planifié (si GTFS fourni)")
    if not sched_df.empty and schedule_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes comparées", f"{schedule_stats.get('rows_compared', 0):,}".replace(",", " "))
        arr = schedule_stats.get("arrival", {})
        dep = schedule_stats.get("departure", {})
        c2.metric("Arrival — médiane (s)", f"{arr.get('median_signed_sec', 0):.0f}")
        c3.metric("Departure — médiane (s)", f"{dep.get('median_signed_sec', 0):.0f}")
        c4.metric("≤ 5 min (arrival)", f"{arr.get('within_5_min_pct', 0):.1f}%")

        # Histogrammes pré-agrégés (60 bacs) — robustes
        sched_plot = sched_df.copy()
        sched_plot["arr_delta_min"] = pd.to_numeric(sched_plot["arr_delta_sec"], errors="coerce") / 60.0
        sched_plot["dep_delta_min"] = pd.to_numeric(sched_plot["dep_delta_sec"], errors="coerce") / 60.0

        charts = st.tabs(["Histogramme arrival", "Histogramme departure"])
        with charts[0]:
            s = sched_plot["arr_delta_min"].dropna()
            if not s.empty:
                df_arr_hist = _safe_histogram(s, bins=60, clip_abs_minutes=1440)
                if not df_arr_hist.empty:
                    chart_arr = (
                        alt.Chart(df_arr_hist)
                           .mark_bar()
                           .encode(
                               x=alt.X("bin_center:Q", title="Écart (minutes, +retard / -avance)"),
                               y=alt.Y("count:Q", title="Nombre")
                           )
                           .properties(height=300)
                    )
                    st.altair_chart(chart_arr, use_container_width=True)
                else:
                    st.info("Histogramme arrival : aucune donnée exploitable après nettoyage.")
            else:
                st.info("Aucun delta arrival disponible.")
        with charts[1]:
            s = sched_plot["dep_delta_min"].dropna()
            if not s.empty:
                df_dep_hist = _safe_histogram(s, bins=60, clip_abs_minutes=1440)
                if not df_dep_hist.empty:
                    chart_dep = (
                        alt.Chart(df_dep_hist)
                           .mark_bar()
                           .encode(
                               x=alt.X("bin_center:Q", title="Écart (minutes)"),
                               y=alt.Y("count:Q", title="Nombre")
                           )
                           .properties(height=300)
                    )
                    st.altair_chart(chart_dep, use_container_width=True)
                else:
                    st.info("Histogramme departure : aucune donnée exploitable après nettoyage.")
            else:
                st.info("Aucun delta departure disponible.")
    else:
        st.info("Aucune comparaison planifié vs prédiction (ajoute un GTFS statique pour activer).")

    st.divider()

    # ----------------------------- Validation CSV d’annulations (fenêtre 2 h)
    st.markdown("### ### Valider un fichier d’annulations (fenêtre 2 h)")
    with st.expander("Ajouter un CSV d’annulations et valider sur une fenêtre temporelle"):
        canc_file = st.file_uploader(
            "Fichier d’annulations (CSV) — colonnes: Trip_id,Start_date,Route_id,Stop_id,Stop_seq",
            type=["csv"], key="cancellations_csv"
        )
        # t0 par défaut: feed.header.timestamp ISO fourni par analyze_tripupdates()
        t0_default = analysis["meta"].get("feed_timestamp_iso") or ""
        win_start_str = st.text_input("Début de fenêtre (ISO local ou epoch s)", value=t0_default)
        win_hours = st.number_input("Durée (heures)", min_value=1, max_value=12, value=2, step=1)
        tz_opt = st.text_input(
            "Fuseau horaire",
            value=(static_gtfs.get("agency", pd.DataFrame()).iloc[0].get("agency_timezone", tz_input)
                   if static_gtfs.get("agency", pd.DataFrame()).shape[0] else tz_input)
        )
        run_val = st.button("Valider les annulations", type="primary")
        if run_val and canc_file is not None:
            try:
                canc_df = pd.read_csv(canc_file, dtype={"Trip_id": str, "Route_id": str, "Stop_id": str})
                val = validate_cancellations_against_tripupdates(
                    cancel_df=canc_df,
                    analysis=analysis,
                    window_start=win_start_str if win_start_str else None,
                    window_hours=int(win_hours),
                    tz=tz_opt or tz_input,
                )
                st.success("Validation terminée ✅")
                # KPIs
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total (lignes CSV)", f"{val['summary']['n']:,}".replace(",", " "))
                c2.metric("OK_FULL", f"{val['summary']['ok_full']:,}".replace(",", " "))
                c3.metric("OK_PARTIAL", f"{val['summary']['ok_partial']:,}".replace(",", " "))
                c4.metric("MISSING_FULL", f"{val['summary']['miss_full']:,}".replace(",", " "))
                c5.metric("MISSING_PARTIAL", f"{val['summary']['miss_partial']:,}".replace(",", " "))

                # Table + Download
                st.dataframe(val["results"], use_container_width=True)
                st.download_button(
                    "⬇️ cancellations_validation.csv",
                    _to_csv_bytes(val["results"]),
                    "cancellations_validation.csv",
                    mime="text/csv",
                )

                # Détails fenêtre
                ws = val.get("window_start_epoch"); we = val.get("window_end_epoch")
                if ws is not None and we is not None:
                    try:
                        ws_iso = pd.to_datetime(ws, unit="s", utc=True).tz_convert(ZoneInfo(tz_opt or tz_input)).isoformat()
                        we_iso = pd.to_datetime(we, unit="s", utc=True).tz_convert(ZoneInfo(tz_opt or tz_input)).isoformat()
                    except Exception:
                        ws_iso = pd.to_datetime(ws, unit="s", utc=True).isoformat()
                        we_iso = pd.to_datetime(we, unit="s", utc=True).isoformat()
                    st.caption(f"Fenêtre: {ws_iso} → {we_iso} ({tz_opt})")
            except Exception as e:
                st.error("❌ Échec de lecture/validation du CSV d’annulations.")
                st.caption(f"Détail: {type(e).__name__}: {e}")
        elif run_val and canc_file is None:
            st.warning("Ajoute d’abord un fichier CSV d’annulations.")

    st.divider()

    # ----------------------------- Détails JSON
    st.markdown("### ### Détails (JSON)")
    st.json({
        "meta": analysis["meta"],
        "summary": analysis["summary"],
        "timestamp_quality": analysis["ts_quality"],
        "schedule_compare": schedule_stats if schedule_stats else "(non disponible)",
        "counts": {
            "trip_rows": int(len(trips_view)),
            "stop_time_updates": int(len(stu_view)),
            "anomalies": int(len(anomalies_df))
        }
    })

    # ----------------------------- Téléchargements
    st.markdown("### ### Téléchargements")
    cdl1, cdl2, cdl3, cdl4 = st.columns(4)
    with cdl1:
        st.download_button("⬇️ trips.csv", _to_csv_bytes(trips_df), "trips.csv", mime="text/csv")
    with cdl2:
        st.download_button("⬇️ stop_updates.csv", _to_csv_bytes(stu_df), "stop_updates.csv", mime="text/csv")
    with cdl3:
        st.download_button("⬇️ anomalies.csv", _to_csv_bytes(anomalies_df), "anomalies.csv", mime="text/csv")
    with cdl4:
        summary_payload = {
            "meta": analysis["meta"],
            "summary": analysis["summary"],
            "timestamp_quality": analysis["ts_quality"],
            "schedule_compare": schedule_stats,
            "counts": {
                "trip_rows": int(len(trips_df)),
                "stop_time_updates": int(len(stu_df)),
                "anomalies": int(len(anomalies_df)),
                "schedule_compare_rows": int(len(sched_df)),
            },
        }
        st.download_button(
            "⬇️ summary.json",
            json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            "summary.json",
            mime="application/json"
        )

    # ----------------------------- Tables
    st.markdown("### ### Tables")
    show_trips, show_stu, show_anom = st.columns(3)
    with show_trips:
        show_trips_flag = st.checkbox("Afficher trips.csv", value=True)
    with show_stu:
        show_stu_flag = st.checkbox("Afficher stop_updates.csv", value=False)
    with show_anom:
        show_anom_flag = st.checkbox("Afficher anomalies.csv", value=True)

    if show_trips_flag:
        st.subheader("Trips")
        tv = trips_view.copy()
        tv["trip_type"] = tv["trip_schedule_relationship"].map(_sr_label_map_trip()).fillna("SCHEDULED")
        st.dataframe(tv, use_container_width=True)

    if show_stu_flag:
        st.subheader("Stop Time Updates (STU)")
        sv = stu_view.copy()
        # selon le merge, la vraie colonne trip_id peut être 'trip_id' ou 'trip_id_t'
        if "trip_id_t" in sv.columns and sv["trip_id"].isna().all():
            sv["trip_id"] = sv["trip_id_t"]
        sv["stu_type"] = sv["stu_schedule_relationship"].map(_sr_label_map_stu()).fillna("SCHEDULED")
        st.dataframe(sv, use_container_width=True)

    if show_anom_flag:
        st.subheader("Anomalies")
        st.dataframe(anomalies_df, use_container_width=True)

    # Message d'info si aucun TripUpdate
    if trips_df.empty and stu_df.empty:
        st.info("Le fichier TripUpdates chargé ne contient aucun TripUpdate. Aucune donnée à afficher.")

else:
    st.info("Charge au moins un fichier **TripUpdates (Protocol Buffer)** puis clique **Analyser** dans la barre latérale.")
``

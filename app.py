# app.py
# ---------------------------------------------------------
# Streamlit - Analyse GTFS-RT TripUpdates vs GTFS statique
# Optimisé pour gros GTFS (filtrage par trips RT), avec:
# - Détection voyages: ADDED / CANCELED / UNSCHEDULED / DELETED
# - Détection arrêts: SKIPPED / NO_DATA
# - Représentations graphiques Altair (délai par stop_sequence,
#   histogrammes, compteurs par route)
# - Téléchargements CSV
# ---------------------------------------------------------
# Dépendances:
#   pip install streamlit pandas altair gtfs-realtime-bindings protobuf
# ---------------------------------------------------------

from __future__ import annotations

import io
import gc
import zipfile
import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# Altair (graphiques)
try:
    import altair as alt
    _HAS_ALTAIR = True
except Exception:
    _HAS_ALTAIR = False

# GTFS-RT protobuf (optionnel, fallback JSON si absent)
try:
    from google.transit import gtfs_realtime_pb2  # type: ignore
    _HAS_GTFS_RT = True
except Exception:
    _HAS_GTFS_RT = False


# ---------------------------------------------
# Configuration Streamlit
# ---------------------------------------------
st.set_page_config(
    page_title="Analyse GTFS-RT TripUpdates",
    page_icon="🚌",
    layout="wide"
)

# ---------------------------------------------
# Constantes / utilitaires
# ---------------------------------------------

TRIP_SCHED_REL = {
    0: "SCHEDULED",
    1: "ADDED",
    2: "UNSCHEDULED",
    3: "CANCELED",
}
STOP_SCHED_REL = {
    0: "SCHEDULED",
    1: "SKIPPED",
    2: "NO_DATA",
}

def _bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    return df

def _hms_to_seconds(hms: str) -> Optional[int]:
    try:
        parts = hms.split(":")
        if len(parts) != 3:
            return None
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None

def _safe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)

def _get_zip_member_case_insensitive(zf: zipfile.ZipFile, target: str) -> Optional[str]:
    lower_target = target.lower()
    for name in zf.namelist():
        if name.lower().endswith("/" + lower_target) or name.lower() == lower_target:
            return name
    return None

def _read_csv_from_zip_filtered(
    zip_bytes: bytes,
    filename: str,
    usecols: List[str],
    dtypes: Dict[str, str],
    filter_col: Optional[str] = None,
    filter_values: Optional[Set[str]] = None,
    chunksize: int = 200_000,
    keep_order_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lecture mémoire-optimisée d'un CSV dans un ZIP, avec filtrage par chunk.
    """
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member = _get_zip_member_case_insensitive(zf, filename)
    if not member:
        return pd.DataFrame(columns=usecols)

    frames: List[pd.DataFrame] = []
    with zf.open(member, "r") as fb:
        text = io.TextIOWrapper(fb, encoding="utf-8", newline="")
        reader = pd.read_csv(
            text,
            usecols=usecols,
            dtype=dtypes,
            chunksize=chunksize
        )
        for chunk in reader:
            chunk = _normalize_cols(chunk)
            if filter_col and filter_values is not None and filter_col in chunk.columns:
                chunk = chunk[chunk[filter_col].isin(filter_values)]
            if keep_order_cols:
                missing = [c for c in keep_order_cols if c not in chunk.columns]
                for m in missing:
                    chunk[m] = pd.NA
                chunk = chunk[keep_order_cols]
            frames.append(chunk)

    df = _safe_concat(frames)
    return df


# ---------------------------------------------
# Parsing GTFS-RT TripUpdates
# ---------------------------------------------

@st.cache_data(show_spinner=False)
def parse_tripupdates_rt(rt_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Parse TripUpdates (protobuf ou JSON) et retourne:
      - rt_trips: une ligne par trip (trip_id, route_id, start_date, start_time, timestamp, trip_status, is_deleted)
      - rt_stop_updates: une ligne par arrêt (trip_id, stop_id, stop_sequence, arrival_time, departure_time,
                          arrival_delay, departure_delay, stop_status)
      - rt_trip_ids: set de trip_id
    """
    rt_trips_records: List[Dict] = []
    rt_su_records: List[Dict] = []
    rt_trip_ids: Set[str] = set()

    # --- Protobuf ---
    if _HAS_GTFS_RT:
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(rt_bytes)
            for entity in feed.entity:
                if not entity.HasField("trip_update"):
                    continue
                tu = entity.trip_update
                trip = tu.trip

                trip_id = (trip.trip_id or "").strip()
                route_id = (trip.route_id or "").strip()
                start_date = (trip.start_date or "").strip()
                start_time = (trip.start_time or "").strip()
                ts = getattr(tu, "timestamp", 0)
                is_deleted = getattr(entity, "is_deleted", False)

                # Trip schedule_relationship
                trip_sr_num = getattr(trip, "schedule_relationship", 0)
                trip_status = TRIP_SCHED_REL.get(int(trip_sr_num), str(trip_sr_num))

                if trip_id:
                    rt_trip_ids.add(trip_id)

                rt_trips_records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "start_time": start_time,
                    "rt_timestamp": int(ts) if ts else pd.NA,
                    "trip_status": trip_status,
                    "is_deleted": bool(is_deleted),
                })

                for stu in tu.stop_time_update:
                    arr = stu.arrival if stu.HasField("arrival") else None
                    dep = stu.departure if stu.HasField("departure") else None
                    stop_sr_num = getattr(stu, "schedule_relationship", 0)
                    stop_status = STOP_SCHED_REL.get(int(stop_sr_num), str(stop_sr_num))

                    su = {
                        "trip_id": trip_id,
                        "stop_id": (stu.stop_id or "").strip(),
                        "stop_sequence": getattr(stu, "stop_sequence", pd.NA),
                        "arrival_time": getattr(arr, "time", pd.NA) if arr else pd.NA,
                        "departure_time": getattr(dep, "time", pd.NA) if dep else pd.NA,
                        "arrival_delay": getattr(arr, "delay", pd.NA) if arr else pd.NA,
                        "departure_delay": getattr(dep, "delay", pd.NA) if dep else pd.NA,
                        "stop_status": stop_status,
                    }
                    rt_su_records.append(su)
        except Exception:
            # On essaiera JSON
            pass

    # --- JSON fallback ---
    if not rt_trips_records and not rt_su_records:
        import json
        try:
            data = json.loads(rt_bytes.decode("utf-8"))
            for entity in data.get("entity", []):
                tu = entity.get("trip_update")
                if not tu:
                    continue
                trip = tu.get("trip", {}) or {}
                trip_id = (trip.get("trip_id") or "").strip()
                route_id = (trip.get("route_id") or "").strip()
                start_date = (trip.get("start_date") or "").strip()
                start_time = (trip.get("start_time") or "").strip()
                ts = tu.get("timestamp")
                is_deleted = bool(entity.get("is_deleted", False))

                # schedule_relationship num ou string
                sr = trip.get("schedule_relationship", 0)
                try:
                    srnum = int(sr)
                    trip_status = TRIP_SCHED_REL.get(srnum, str(sr))
                except Exception:
                    trip_status = str(sr)

                if trip_id:
                    rt_trip_ids.add(trip_id)

                rt_trips_records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "start_time": start_time,
                    "rt_timestamp": int(ts) if ts else pd.NA,
                    "trip_status": trip_status,
                    "is_deleted": is_deleted,
                })

                for stu in tu.get("stop_time_update", []):
                    arr = stu.get("arrival", {}) or {}
                    dep = stu.get("departure", {}) or {}
                    sr = stu.get("schedule_relationship", 0)
                    try:
                        srnum = int(sr)
                        stop_status = STOP_SCHED_REL.get(srnum, str(sr))
                    except Exception:
                        stop_status = str(sr)

                    su = {
                        "trip_id": trip_id,
                        "stop_id": (stu.get("stop_id") or "").strip(),
                        "stop_sequence": stu.get("stop_sequence"),
                        "arrival_time": arr.get("time"),
                        "departure_time": dep.get("time"),
                        "arrival_delay": arr.get("delay"),
                        "departure_delay": dep.get("delay"),
                        "stop_status": stop_status,
                    }
                    rt_su_records.append(su)
        except Exception as e:
            raise ValueError("Impossible de parser le TripUpdate (ni protobuf ni JSON).") from e

    rt_trips = pd.DataFrame(rt_trips_records)
    rt_su = pd.DataFrame(rt_su_records)

    # Types
    if not rt_trips.empty:
        for c in ["trip_id", "route_id", "start_date", "start_time", "trip_status"]:
            if c in rt_trips:
                rt_trips[c] = rt_trips[c].astype("string")
        for c in ["rt_timestamp"]:
            if c in rt_trips:
                rt_trips[c] = pd.to_numeric(rt_trips[c], errors="coerce").astype("Int64")
        if "is_deleted" in rt_trips:
            rt_trips["is_deleted"] = rt_trips["is_deleted"].astype("boolean")

    if not rt_su.empty:
        for c in ["trip_id", "stop_id", "stop_status"]:
            if c in rt_su:
                rt_su[c] = rt_su[c].astype("string")
        for c in ["stop_sequence", "arrival_time", "departure_time", "arrival_delay", "departure_delay"]:
            if c in rt_su:
                rt_su[c] = pd.to_numeric(rt_su[c], errors="coerce").astype("Int64")

    return rt_trips, rt_su, rt_trip_ids


# ---------------------------------------------
# Chargement GTFS statique filtré
# ---------------------------------------------

@st.cache_data(show_spinner=False)
def load_trips_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    dtypes = {
        "route_id": "string", "service_id": "string", "trip_id": "string",
        "trip_headsign": "string", "direction_id": "Int64", "shape_id": "string",
    }
    if not keep_trip_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "trips.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=200_000, keep_order_cols=usecols
    )

@st.cache_data(show_spinner=False)
def load_stop_times_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    dtypes = {
        "trip_id": "string", "arrival_time": "string", "departure_time": "string",
        "stop_id": "string", "stop_sequence": "Int64",
    }
    if not keep_trip_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stop_times.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=400_000, keep_order_cols=usecols
    )

@st.cache_data(show_spinner=False)
def load_stops_filtered(gtfs_zip_bytes: bytes, keep_stop_ids: Set[str]) -> pd.DataFrame:
    usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    dtypes = {"stop_id": "string", "stop_name": "string", "stop_lat": "float64", "stop_lon": "float64"}
    if not keep_stop_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stops.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="stop_id", filter_values=keep_stop_ids,
        chunksize=200_000, keep_order_cols=usecols
    )

@st.cache_data(show_spinner=False)
def load_routes_filtered(gtfs_zip_bytes: bytes, keep_route_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "route_short_name", "route_long_name"]
    dtypes = {"route_id": "string", "route_short_name": "string", "route_long_name": "string"}
    if not keep_route_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "routes.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="route_id", filter_values=keep_route_ids,
        chunksize=200_000, keep_order_cols=usecols
    )

@st.cache_data(show_spinner=False)
def load_calendar(gtfs_zip_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal_cols = ["service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"]
    cal_dtypes = {c:"Int64" for c in cal_cols if c not in ["service_id","start_date","end_date"]}
    cal_dtypes.update({"service_id":"string","start_date":"string","end_date":"string"})
    cd_cols = ["service_id","date","exception_type"]
    cd_dtypes = {"service_id":"string","date":"string","exception_type":"Int64"}

    zf = zipfile.ZipFile(io.BytesIO(gtfs_zip_bytes))

    cal_df = pd.DataFrame(columns=cal_cols)
    member = _get_zip_member_case_insensitive(zf, "calendar.txt")
    if member:
        with zf.open(member, "r") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8", newline="")
            cal_df = pd.read_csv(text, usecols=[c for c in cal_cols if c], dtype=cal_dtypes)
            cal_df = _normalize_cols(cal_df)

    cd_df = pd.DataFrame(columns=cd_cols)
    member = _get_zip_member_case_insensitive(zf, "calendar_dates.txt")
    if member:
        with zf.open(member, "r") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8", newline="")
            cd_df = pd.read_csv(text, usecols=cd_cols, dtype=cd_dtypes)
            cd_df = _normalize_cols(cd_df)

    return cal_df, cd_df


# ---------------------------------------------
# Analyse / Jointures
# ---------------------------------------------

def compute_schedule_vs_rt(
    rt_stop_updates: pd.DataFrame,
    stop_times_filtered: pd.DataFrame,
    stops_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Jointure RT vs schedule par arrêt. Clés: (trip_id, stop_id) sinon fallback (trip_id, stop_sequence).
    Ajoute seconds planifiées et libellé d'arrêt.
    """
    if rt_stop_updates.empty or stop_times_filtered.empty:
        return pd.DataFrame()

    sched = stop_times_filtered.copy()
    for col in ["arrival_time", "departure_time"]:
        if col in sched.columns:
            sched[col + "_sec"] = sched[col].map(_hms_to_seconds).astype("Int64")

    left = rt_stop_updates.copy()

    if "stop_id" in left.columns and "stop_id" in sched.columns:
        join_cols = ["trip_id", "stop_id"]
    elif "stop_sequence" in left.columns and "stop_sequence" in sched.columns:
        join_cols = ["trip_id", "stop_sequence"]
    else:
        join_cols = ["trip_id"]

    merged = pd.merge(
        left, sched, how="left", on=join_cols, suffixes=("_rt", "_sched"), copy=False
    )

    if not stops_df.empty and "stop_id" in merged.columns:
        merged = pd.merge(merged, stops_df[["stop_id", "stop_name"]], how="left", on="stop_id", copy=False)

    # Types propres
    for c in ["arrival_time_sec", "departure_time_sec"]:
        if c in merged.columns:
            merged[c] = merged[c].astype("Int64")

    # Délai "best" (priorise arrival puis departure)
    if "arrival_delay" in merged.columns and "departure_delay" in merged.columns:
        merged["delay_best"] = merged[["arrival_delay", "departure_delay"]].bfill(axis=1).iloc[:, 0]
        merged["delay_best"] = pd.to_numeric(merged["delay_best"], errors="coerce").astype("Int64")

    return merged


def summarize_trips(
    rt_trips: pd.DataFrame,
    rt_stop_updates: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Résumé par trip: statut (ADDED/CANCELED/...), nb updates, nb SKIPPED / NO_DATA,
    stats de délai (moyenne, max, min).
    """
    if rt_trips.empty:
        return pd.DataFrame()

    # Agrégations stop-level
    agg = pd.DataFrame({"trip_id": rt_trips["trip_id"].unique()})
    if not rt_stop_updates.empty:
        su = rt_stop_updates.copy()
        su["delay_any"] = su[["arrival_delay", "departure_delay"]].mean(axis=1, skipna=True)
        grp = su.groupby("trip_id", as_index=False).agg(
            updates_count=("stop_id", "count"),
            avg_delay_sec=("delay_any", "mean"),
            max_delay_sec=("delay_any", "max"),
            min_delay_sec=("delay_any", "min"),
            skipped_stops=("stop_status", lambda s: (s == "SKIPPED").sum()),
            nodata_stops=("stop_status", lambda s: (s == "NO_DATA").sum()),
        )
        agg = pd.merge(agg, grp, on="trip_id", how="left")

    out = pd.merge(rt_trips, agg, on="trip_id", how="left")
    if not trips_df.empty:
        out = pd.merge(
            out,
            trips_df[["trip_id", "route_id", "trip_headsign", "direction_id"]],
            on="trip_id", how="left", suffixes=("", "_sched")
        )
    # route info
    if not routes_df.empty and "route_id" in out.columns:
        out = pd.merge(out, routes_df, on="route_id", how="left")

    # Nettoyage types
    for c in ["avg_delay_sec", "max_delay_sec", "min_delay_sec"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    # Colonnes utiles en tête
    prefer = [
        "trip_id","route_id","route_short_name","route_long_name","trip_headsign","direction_id",
        "trip_status","is_deleted","updates_count","skipped_stops","nodata_stops",
        "avg_delay_sec","max_delay_sec","min_delay_sec","start_date","start_time","rt_timestamp"
    ]
    cols = [c for c in prefer if c in out.columns] + [c for c in out.columns if c not in prefer]
    out = out[cols]

    return out


# ---------------------------------------------
# Graphiques Altair
# ---------------------------------------------

def chart_trip_delays(det: pd.DataFrame, trip_id: str):
    """
    Courbe des délais (arrival/departure) par stop_sequence pour un trip.
    Points: status (SKIPPED/NO_DATA).
    """
    if not _HAS_ALTAIR or det.empty:
        return None

    df = det.copy()
    # axis x: stop_sequence (si manquant, fallback index)
    if "stop_sequence" not in df.columns or df["stop_sequence"].isna().all():
        df = df.reset_index().rename(columns={"index": "stop_sequence"})
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")

    # delay_best (préféré)
    if "delay_best" not in df.columns:
        df["delay_best"] = df[["arrival_delay","departure_delay"]].bfill(axis=1).iloc[:,0]

    base = alt.Chart(df).encode(
        x=alt.X("stop_sequence:Q", title="Stop sequence"),
        tooltip=[
            alt.Tooltip("stop_sequence:Q"),
            alt.Tooltip("stop_id:N"),
            alt.Tooltip("stop_name:N"),
            alt.Tooltip("stop_status:N", title="Stop status"),
            alt.Tooltip("arrival_delay:Q", title="Arr delay (s)"),
            alt.Tooltip("departure_delay:Q", title="Dep delay (s)"),
        ],
    )

    line = base.mark_line(point=True).encode(
        y=alt.Y("delay_best:Q", title="Délai (s)"),
        color=alt.condition(
            alt.datum.delay_best > 0,
            alt.value("#d62728"),  # rouge retard
            alt.value("#2ca02c")   # vert avance/à l'heure
        )
    ).properties(
        title=f"Delais par arrêt — Trip {trip_id}"
    )

    # Points pour SKIPPED/NO_DATA
    pts = base.mark_point(filled=True, size=90, shape="triangle-up").encode(
        y="delay_best:Q",
        color=alt.condition(
            alt.FieldEqualPredicate("stop_status", "SKIPPED"),
            alt.value("#1f77b4"),
            alt.value("#9467bd")
        ),
        shape=alt.Shape("stop_status:N", scale=alt.Scale(domain=["SKIPPED","NO_DATA"], range=["triangle-up","diamond"])),
        opacity=alt.condition(alt.FieldOneOfPredicate("stop_status", ["SKIPPED","NO_DATA"]), alt.value(1), alt.value(0))
    )

    return (line + pts).interactive()

def chart_avg_delay_by_route(summary_df: pd.DataFrame):
    if not _HAS_ALTAIR or summary_df.empty:
        return None
    df = summary_df.copy()
    # Label route
    if "route_short_name" in df.columns:
        df["route_label"] = df["route_short_name"].fillna(df.get("route_long_name"))
    else:
        df["route_label"] = df.get("route_long_name", pd.Series([""]*len(df)))
    df["route_label"] = df["route_label"].fillna(df["route_id"]).astype(str)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("route_label:N", title="Route", sort="-y"),
        y=alt.Y("mean(avg_delay_sec):Q", title="Retard moyen (s)"),
        tooltip=[
            alt.Tooltip("route_label:N", title="Route"),
            alt.Tooltip("mean(avg_delay_sec):Q", title="Retard moyen (s)", format=".1f"),
            alt.Tooltip("count():Q", title="Nb trips"),
        ],
        color=alt.Color("mean(avg_delay_sec):Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
    ).properties(
        title="Retard moyen par route"
    )
    return chart.interactive()

def chart_trip_status_counts(summary_df: pd.DataFrame):
    if not _HAS_ALTAIR or summary_df.empty:
        return None
    df = summary_df.copy()
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("trip_status:N", title="Statut du voyage"),
        y=alt.Y("count():Q", title="Nombre de voyages"),
        color="trip_status:N",
        tooltip=[alt.Tooltip("trip_status:N", title="Statut"), alt.Tooltip("count():Q", title="Nb")]
    ).properties(
        title="Comptes de voyages par statut (SCHEDULED/ADDED/CANCELED/...)"
    )
    return chart.interactive()


# ---------------------------------------------
# UI Streamlit
# ---------------------------------------------

def main():
    st.title("🚌 Analyse GTFS-RT TripUpdates vs GTFS (optimisée)")
    st.caption("Inclut voyages ADDED/CANCELED, arrêts SKIPPED/NO_DATA et graphiques Altair. Lecture ciblée sur les trips RT pour gros GTFS (~70 Mo+).")

    with st.sidebar:
        st.header("Fichiers d'entrée")
        gtfs_zip = st.file_uploader("GTFS statique (.zip)", type=["zip"])
        rt_file = st.file_uploader("TripUpdates GTFS-RT (.pb/.bin ou .json)", type=["pb", "bin", "json"])

        st.divider()
        st.subheader("Options")
        focus_only_rt_trips = st.checkbox("Se focaliser sur les voyages du TripUpdate (recommandé)", value=True)
        limit_trips = st.number_input("Limiter à N trips (optionnel)", min_value=0, value=0, step=100)
        show_raw_tables = st.checkbox("Afficher tables brutes (debug)", value=False)
        enable_charts = st.checkbox("Afficher graphiques Altair", value=True)

        st.divider()
        run_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if not run_btn:
        st.info("💡 Charge les fichiers puis clique sur **Lancer l'analyse**.")
        return

    if not gtfs_zip or not rt_file:
        st.error("Merci de fournir le **GTFS statique (.zip)** et le **TripUpdate (.pb/.bin/.json)**.")
        return

    try:
        gtfs_bytes = gtfs_zip.getvalue()
    except Exception:
        gtfs_bytes = gtfs_zip.read()

    try:
        rt_bytes = rt_file.getvalue()
    except Exception:
        rt_bytes = rt_file.read()

    with st.status("📥 Lecture du TripUpdate...", expanded=False) as status:
        try:
            rt_trips, rt_su, rt_trip_ids = parse_tripupdates_rt(rt_bytes)
        except Exception as e:
            st.exception(e)
            status.update(label="Erreur parsing TripUpdate.", state="error")
            return
        status.update(label=f"TripUpdate: {len(rt_trips)} trips, {len(rt_su)} stop updates.", state="complete")

    if focus_only_rt_trips:
        keep_trip_ids = {tid for tid in rt_trip_ids if isinstance(tid, str) and tid}
        if limit_trips and limit_trips > 0 and keep_trip_ids:
            keep_trip_ids = set(list(keep_trip_ids)[: int(limit_trips)])
    else:
        st.warning("⚠️ Filtrage par trips RT désactivé — utilisation mémoire potentiellement élevée.")
        keep_trip_ids = set()

    # Chargements filtrés
    progress = st.progress(0, text="Chargement trips...")
    trips_df = load_trips_filtered(gtfs_bytes, keep_trip_ids)
    progress.progress(25, text="Chargement stop_times...")
    stop_times_df = load_stop_times_filtered(gtfs_bytes, keep_trip_ids)

    # Dérive stops et routes à charger
    keep_stop_ids: Set[str] = set()
    keep_route_ids: Set[str] = set()
    if not stop_times_df.empty and "stop_id" in stop_times_df:
        keep_stop_ids = set(stop_times_df["stop_id"].dropna().astype("string").unique().tolist())
    if not trips_df.empty and "route_id" in trips_df:
        keep_route_ids = set(trips_df["route_id"].dropna().astype("string").unique().tolist())

    progress.progress(55, text="Chargement stops...")
    stops_df = load_stops_filtered(gtfs_bytes, keep_stop_ids)
    progress.progress(75, text="Chargement routes...")
    routes_df = load_routes_filtered(gtfs_bytes, keep_route_ids)
    progress.progress(88, text="Chargement calendar (optionnel)...")
    cal_df, cd_df = load_calendar(gtfs_bytes)

    progress.progress(94, text="Jointure Schedule vs RT...")
    sched_vs_rt = compute_schedule_vs_rt(rt_su, stop_times_df, stops_df)

    progress.progress(100, text="Synthèse...")
    summary_df = summarize_trips(rt_trips, rt_su, trips_df, routes_df)
    st.success("Analyse terminée ✅")

    # -----------------------------
    # KPIs
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trips dans RT", f"{rt_trips['trip_id'].nunique() if not rt_trips.empty else 0:,}")
    with c2:
        canceled = 0
        if "trip_status" in rt_trips.columns:
            canceled = int((rt_trips["trip_status"] == "CANCELED").sum())
        st.metric("Voyages annulés", f"{canceled:,}")
    with c3:
        added = 0
        if "trip_status" in rt_trips.columns:
            added = int((rt_trips["trip_status"] == "ADDED").sum())
        st.metric("Voyages ajoutés", f"{added:,}")
    with c4:
        skipped = 0
        if not rt_su.empty and "stop_status" in rt_su.columns:
            skipped = int((rt_su["stop_status"] == "SKIPPED").sum())
        st.metric("Arrêts SKIPPED", f"{skipped:,}")

    # -----------------------------
    # Résumé voyages
    # -----------------------------
    st.subheader("Résumé par voyage (statut, retards, compteurs)")
    if summary_df.empty:
        st.info("Aucun résumé disponible.")
    else:
        order_cols = [c for c in [
            "trip_id","route_id","route_short_name","route_long_name","trip_headsign","direction_id",
            "trip_status","is_deleted","updates_count","skipped_stops","nodata_stops",
            "avg_delay_sec","max_delay_sec","min_delay_sec","start_date","start_time","rt_timestamp"
        ] if c in summary_df.columns]
        st.dataframe(
            summary_df[order_cols].sort_values(
                ["trip_status","route_id","trip_id"], na_position="last"
            ),
            use_container_width=True, height=360
        )
        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le résumé CSV", data=csv, file_name="resume_trips_rt.csv", mime="text/csv")

    # -----------------------------
    # Détails arrêt par arrêt
    # -----------------------------
    st.subheader("Détails arrêt par arrêt (Schedule vs RT)")
    if sched_vs_rt.empty:
        st.info("Aucun détail d'arrêt joint (clés manquantes ou TripUpdate vide).")
    else:
        # Filtre trip
        trip_choices = sorted(set(sched_vs_rt["trip_id"].dropna().astype(str).unique()))
        selected_trip = st.selectbox("Sélectionne un trip", options=trip_choices)
        det = sched_vs_rt[sched_vs_rt["trip_id"] == selected_trip].copy()

        # Tri par stop_sequence ou stop_id
        if "stop_sequence" in det.columns and not det["stop_sequence"].isna().all():
            det = det.sort_values("stop_sequence")
        elif "stop_id" in det.columns:
            det = det.sort_values("stop_id")

        cols = [c for c in [
            "trip_id","stop_sequence","stop_id","stop_name",
            "stop_status","arrival_time_sec","departure_time_sec",
            "arrival_delay","departure_delay","delay_best",
            "arrival_time","departure_time"
        ] if c in det.columns]
        st.dataframe(det[cols], use_container_width=True, height=420)

        csvd = det.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les détails (trip sélectionné)", data=csvd, file_name=f"details_trip_{selected_trip}.csv", mime="text/csv")

        # -------------------------
        # Graphiques Altair (trip)
        # -------------------------
        if enable_charts:
            if not _HAS_ALTAIR:
                st.warning("Altair n'est pas installé — `pip install altair` pour activer les graphiques.")
            else:
                st.subheader("Visualisations")
                ch1 = chart_trip_delays(det, selected_trip)
                if ch1 is not None:
                    st.altair_chart(ch1, use_container_width=True)

                # Graphiques globaux
                ch2 = chart_avg_delay_by_route(summary_df)
                ch3 = chart_trip_status_counts(summary_df)
                if ch2 is not None or ch3 is not None:
                    gcol1, gcol2 = st.columns(2)
                    with gcol1:
                        if ch2 is not None:
                            st.altair_chart(ch2, use_container_width=True, theme=None)
                    with gcol2:
                        if ch3 is not None:
                            st.altair_chart(ch3, use_container_width=True, theme=None)

    # Debug / Tables brutes
    if show_raw_tables:
        st.divider()
        st.subheader("Debug: tables brutes")
        with st.expander("RT: Trips"):
            st.dataframe(rt_trips, use_container_width=True, height=250)
        with st.expander("RT: Stop Updates"):
            st.dataframe(rt_su, use_container_width=True, height=250)
        with st.expander("GTFS: trips (filtré)"):
            st.dataframe(trips_df, use_container_width=True, height=250)
        with st.expander("GTFS: stop_times (filtré)"):
            st.dataframe(stop_times_df.head(1000), use_container_width=True, height=250)
        with st.expander("GTFS: stops (filtré)"):
            st.dataframe(stops_df, use_container_width=True, height=250)
        with st.expander("GTFS: routes (filtré)"):
            st.dataframe(routes_df, use_container_width=True, height=250)
        with st.expander("GTFS: calendar / calendar_dates"):
            st.dataframe(cal_df, use_container_width=True, height=250)
            st.dataframe(cd_df, use_container_width=True, height=250)

    # Libération mémoire
    del gtfs_bytes, rt_bytes
    gc.collect()


if __name__ == "__main__":
    main()

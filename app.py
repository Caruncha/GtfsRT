# app.py
# ---------------------------------------------------------
# Streamlit - Analyse GTFS-RT TripUpdates vs GTFS statique
# Version simplifiée + visualisations demandées :
#   - Synthèse (CANCELED, ADDED, partiellement annulés, trip_id inconnus)
#   - Résumé par voyage
#   - Anomalies (table)
#   - GRAPHIQUES :
#       * Anomalies (sans doublons)
#       * Répartition des statuts des arrêts
#       * Répartition par minute (avance/retard) vs schedule (tous arrêts, pas = 1 min)
# Optimisé gros GTFS (filtrage par trips RT, lecture par chunks, dtypes compacts).
# ---------------------------------------------------------
# Dépendances :
#   pip install streamlit pandas altair gtfs-realtime-bindings protobuf
# ---------------------------------------------------------

from __future__ import annotations

import io
import math
import gc
import os
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# Chargement .env (silencieux si absent)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

try:
    import requests as _requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# Altair (charts)
try:
    import altair as alt
    _HAS_ALTAIR = True
    alt.data_transformers.disable_max_rows()
except Exception:
    _HAS_ALTAIR = False

# GTFS-RT protobuf (optionnel; fallback JSON si absent)
# On importe d'abord le fichier local patché (ajoute NEW=8, DELETED=7)
# puis on tombe sur google.transit si absent
try:
    import gtfs_realtime_pb2  # type: ignore  # local patched version (NEW=8)
    _HAS_GTFS_RT = True
except Exception:
    try:
        from google.transit import gtfs_realtime_pb2  # type: ignore
        _HAS_GTFS_RT = True
    except Exception:
        _HAS_GTFS_RT = False


# ---------------------------------------------
# Config Streamlit
# ---------------------------------------------
st.set_page_config(
    page_title="Analyse GTFS-RT",
    page_icon="🚌",
    layout="wide"
)

# ---------------------------------------------
# Constantes & utilitaires
# ---------------------------------------------
TRIP_SCHED_REL = {0: "SCHEDULED", 1: "ADDED", 2: "UNSCHEDULED", 3: "CANCELED", 7: "DELETED", 8: "NEW"}
STOP_SCHED_REL = {0: "SCHEDULED", 1: "SKIPPED", 2: "NO_DATA"}
VEHICLE_STATUS = {0: "INCOMING_AT", 1: "STOPPED_AT", 2: "IN_TRANSIT_TO"}
CONGESTION_LEVEL = {
    0: "UNKNOWN_CONGESTION", 1: "RUNNING_SMOOTHLY", 2: "STOP_AND_GO",
    3: "CONGESTION", 4: "SEVERE_CONGESTION",
}
OCCUPANCY_STATUS = {
    0: "EMPTY", 1: "MANY_SEATS_AVAILABLE", 2: "FEW_SEATS_AVAILABLE",
    3: "STANDING_ROOM_ONLY", 4: "CRUSHED_STANDING_ROOM_ONLY",
    5: "FULL", 6: "NOT_ACCEPTING_PASSENGERS", 7: "NO_DATA_AVAILABLE", 8: "NOT_BOARDABLE",
}

# Timezone locale pour convertir start_date + HH:MM:SS (Montréal)
TIMEZONE = "America/Toronto"  # couvre Montréal avec DST


def _load_api_config() -> dict:
    """Lit la configuration API depuis les variables d'environnement (chargées depuis .env)."""
    return {
        "tripupdates_url":    os.getenv("GTFSRT_TRIPUPDATES_URL", "").strip(),
        "vehiclepositions_url": os.getenv("GTFSRT_VEHICLEPOSITIONS_URL", "").strip(),
        "gtfs_zip_url":       os.getenv("GTFSRT_GTFS_ZIP_URL", "").strip(),
        "api_key":            os.getenv("GTFSRT_API_KEY", "").strip(),
        "api_key_header":     os.getenv("GTFSRT_API_KEY_HEADER", "apikey").strip(),
        "bearer_token":       os.getenv("GTFSRT_BEARER_TOKEN", "").strip(),
        "username":           os.getenv("GTFSRT_USERNAME", "").strip(),
        "password":           os.getenv("GTFSRT_PASSWORD", "").strip(),
        "timeout":            int(os.getenv("GTFSRT_TIMEOUT", "30")),
    }


def fetch_feed(url: str, cfg: dict) -> bytes:
    """
    Télécharge un flux depuis `url` en appliquant l'auth configurée.
    Lève une exception en cas d'erreur HTTP ou réseau.
    """
    if not _HAS_REQUESTS:
        raise RuntimeError("Le module `requests` n'est pas installé.")
    if not url:
        raise ValueError("URL non configurée.")

    headers: Dict[str, str] = {}
    auth = None

    if cfg.get("bearer_token"):
        headers["Authorization"] = f"Bearer {cfg['bearer_token']}"
    elif cfg.get("api_key"):
        headers[cfg["api_key_header"]] = cfg["api_key"]

    if cfg.get("username"):
        auth = (cfg["username"], cfg["password"])

    resp = _requests.get(url, headers=headers, auth=auth, timeout=cfg["timeout"])
    resp.raise_for_status()
    return resp.content


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def _hms_to_seconds(hms: str) -> Optional[int]:
    try:
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def _safe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _peek_csv_columns(zip_bytes: bytes, filename: str) -> Set[str]:
    """Lit uniquement l'en-tête d'un CSV dans un ZIP — très rapide, ne charge pas le contenu."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member = _get_zip_member_case_insensitive(zf, filename)
    if not member:
        return set()
    with zf.open(member, "r") as fb:
        first_line = io.TextIOWrapper(fb, encoding="utf-8", newline="").readline()
    return {c.strip().strip('"').strip() for c in first_line.split(",")}


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
    """Lecture optimisée d'un CSV dans un ZIP avec filtrage par chunk."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member = _get_zip_member_case_insensitive(zf, filename)
    if not member:
        return pd.DataFrame(columns=usecols)

    frames: List[pd.DataFrame] = []
    with zf.open(member, "r") as fb:
        text = io.TextIOWrapper(fb, encoding="utf-8", newline="")
        reader = pd.read_csv(text, usecols=usecols, dtype=dtypes, chunksize=chunksize)
        for chunk in reader:
            chunk = _normalize_cols(chunk)
            if filter_col and filter_values is not None and filter_col in chunk.columns:
                chunk = chunk[chunk[filter_col].isin(filter_values)]
            if keep_order_cols:
                for m in [c for c in keep_order_cols if c not in chunk.columns]:
                    chunk[m] = pd.NA
                chunk = chunk[keep_order_cols]
            if len(chunk):
                frames.append(chunk)

    return _safe_concat(frames)


# ---------------------------------------------
# Parsing GTFS-RT TripUpdates
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def parse_tripupdates_rt(rt_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Retourne :
      - rt_trips: [trip_id, route_id, start_date, start_time, rt_timestamp, trip_status, is_deleted]
      - rt_stop_updates: [trip_id, stop_id, stop_sequence, arrival_time, departure_time, arrival_delay, departure_delay, stop_status]
      - set des trip_id
    """
    rt_trips_records: List[Dict] = []
    rt_su_records: List[Dict] = []
    rt_trip_ids: Set[str] = set()

    # Protobuf
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

                trip_sr = TRIP_SCHED_REL.get(int(getattr(trip, "schedule_relationship", 0)),
                                             str(getattr(trip, "schedule_relationship", 0)))

                if trip_id:
                    rt_trip_ids.add(trip_id)

                rt_trips_records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "start_time": start_time,
                    "rt_timestamp": int(ts) if ts else pd.NA,
                    "trip_status": trip_sr,
                    "is_deleted": bool(is_deleted),
                })

                for stu in tu.stop_time_update:
                    arr = stu.arrival if stu.HasField("arrival") else None
                    dep = stu.departure if stu.HasField("departure") else None
                    stop_sr = STOP_SCHED_REL.get(int(getattr(stu, "schedule_relationship", 0)),
                                                 str(getattr(stu, "schedule_relationship", 0)))

                    rt_su_records.append({
                        "trip_id": trip_id,
                        "stop_id": (stu.stop_id or "").strip(),
                        "stop_sequence": getattr(stu, "stop_sequence", pd.NA),
                        "arrival_time": getattr(arr, "time", pd.NA) if arr else pd.NA,
                        "departure_time": getattr(dep, "time", pd.NA) if dep else pd.NA,
                        "arrival_delay": getattr(arr, "delay", pd.NA) if arr else pd.NA,
                        "departure_delay": getattr(dep, "delay", pd.NA) if dep else pd.NA,
                        "stop_status": stop_sr,
                    })
        except Exception:
            pass

    # JSON fallback
    if not rt_trips_records and not rt_su_records:
        import json
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

            sr = trip.get("schedule_relationship", 0)
            try:
                trip_sr = TRIP_SCHED_REL.get(int(sr), str(sr))
            except Exception:
                trip_sr = str(sr)

            if trip_id:
                rt_trip_ids.add(trip_id)

            rt_trips_records.append({
                "trip_id": trip_id,
                "route_id": route_id,
                "start_date": start_date,
                "start_time": start_time,
                "rt_timestamp": int(ts) if ts else pd.NA,
                "trip_status": trip_sr,
                "is_deleted": is_deleted,
            })

            for stu in tu.get("stop_time_update", []):
                arr = stu.get("arrival", {}) or {}
                dep = stu.get("departure", {}) or {}
                srs = stu.get("schedule_relationship", 0)
                try:
                    stop_sr = STOP_SCHED_REL.get(int(srs), str(srs))
                except Exception:
                    stop_sr = str(srs)

                rt_su_records.append({
                    "trip_id": trip_id,
                    "stop_id": (stu.get("stop_id") or "").strip(),
                    "stop_sequence": stu.get("stop_sequence"),
                    "arrival_time": arr.get("time"),
                    "departure_time": dep.get("time"),
                    "arrival_delay": arr.get("delay"),
                    "departure_delay": dep.get("delay"),
                    "stop_status": stop_sr,
                })

    rt_trips = pd.DataFrame(rt_trips_records)
    rt_su = pd.DataFrame(rt_su_records)

    # Types légers
    if not rt_trips.empty:
        for c in ["trip_id", "route_id", "start_date", "start_time", "trip_status"]:
            if c in rt_trips:
                rt_trips[c] = rt_trips[c].astype("string")
        if "rt_timestamp" in rt_trips:
            rt_trips["rt_timestamp"] = pd.to_numeric(rt_trips["rt_timestamp"], errors="coerce").astype("Int64")
        if "is_deleted" in rt_trips:
            rt_trips["is_deleted"] = rt_trips["is_deleted"].astype("boolean")

    if not rt_su.empty:
        for c in ["trip_id", "stop_id", "stop_status"]:
            if c in rt_su:
                rt_su[c] = rt_su[c].astype("string")
        for c in ["stop_sequence", "arrival_time", "departure_time", "arrival_delay", "departure_delay"]:
            if c in rt_su:
                rt_su[c] = pd.to_numeric(rt_su[c], errors="coerce").astype("Int64")

    return rt_trips, rt_su, set(rt_trips["trip_id"].dropna().astype(str).unique()) if not rt_trips.empty else set()


# ---------------------------------------------
# Parsing GTFS-RT VehiclePositions
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def parse_vehiclepositions_rt(rt_bytes: bytes) -> pd.DataFrame:
    """
    Retourne un DataFrame avec une ligne par véhicule :
      vehicle_id, vehicle_label, trip_id, route_id, start_date,
      latitude, longitude, bearing, speed,
      current_status, current_stop_sequence, stop_id,
      timestamp, congestion_level, occupancy_status, occupancy_percentage
    """
    records: List[Dict] = []

    # Protobuf
    if _HAS_GTFS_RT:
        try:
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(rt_bytes)
            for entity in feed.entity:
                if not entity.HasField("vehicle"):
                    continue
                vp = entity.vehicle

                # VehicleDescriptor
                vd = vp.vehicle if vp.HasField("vehicle") else None
                vehicle_id = (getattr(vd, "id", "") or "").strip() if vd else ""
                vehicle_label = (getattr(vd, "label", "") or "").strip() if vd else ""

                # TripDescriptor
                trip = vp.trip if vp.HasField("trip") else None
                trip_id = (getattr(trip, "trip_id", "") or "").strip() if trip else ""
                route_id = (getattr(trip, "route_id", "") or "").strip() if trip else ""
                start_date = (getattr(trip, "start_date", "") or "").strip() if trip else ""

                # Position
                pos = vp.position if vp.HasField("position") else None
                latitude = getattr(pos, "latitude", None) if pos else None
                longitude = getattr(pos, "longitude", None) if pos else None
                bearing = getattr(pos, "bearing", None) if pos else None
                speed = getattr(pos, "speed", None) if pos else None

                current_status_raw = int(getattr(vp, "current_status", 2))
                current_status = VEHICLE_STATUS.get(current_status_raw, str(current_status_raw))
                current_stop_sequence = getattr(vp, "current_stop_sequence", None) or None
                stop_id = (getattr(vp, "stop_id", "") or "").strip()
                ts = getattr(vp, "timestamp", 0)

                congestion_raw = int(getattr(vp, "congestion_level", 0))
                congestion = CONGESTION_LEVEL.get(congestion_raw, str(congestion_raw))

                occupancy_raw = getattr(vp, "occupancy_status", None)
                occupancy = OCCUPANCY_STATUS.get(int(occupancy_raw), str(occupancy_raw)) if occupancy_raw is not None else None
                occupancy_pct = getattr(vp, "occupancy_percentage", None) or None

                records.append({
                    "vehicle_id": vehicle_id or entity.id,
                    "vehicle_label": vehicle_label,
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "latitude": latitude,
                    "longitude": longitude,
                    "bearing": bearing,
                    "speed": speed,
                    "current_status": current_status,
                    "current_stop_sequence": current_stop_sequence,
                    "stop_id": stop_id,
                    "timestamp": int(ts) if ts else pd.NA,
                    "congestion_level": congestion,
                    "occupancy_status": occupancy,
                    "occupancy_percentage": occupancy_pct,
                })
        except Exception:
            pass

    # JSON fallback
    if not records:
        import json
        try:
            data = json.loads(rt_bytes.decode("utf-8"))
            for entity in data.get("entity", []):
                vp = entity.get("vehicle")
                if not vp:
                    continue
                vd = vp.get("vehicle", {}) or {}
                trip = vp.get("trip", {}) or {}
                pos = vp.get("position", {}) or {}

                cs_raw = vp.get("current_status", 2)
                try:
                    current_status = VEHICLE_STATUS.get(int(cs_raw), str(cs_raw))
                except Exception:
                    current_status = str(cs_raw)

                occ_raw = vp.get("occupancy_status")
                try:
                    occupancy = OCCUPANCY_STATUS.get(int(occ_raw), str(occ_raw)) if occ_raw is not None else None
                except Exception:
                    occupancy = str(occ_raw) if occ_raw is not None else None

                cong_raw = vp.get("congestion_level", 0)
                try:
                    congestion = CONGESTION_LEVEL.get(int(cong_raw), str(cong_raw))
                except Exception:
                    congestion = str(cong_raw)

                records.append({
                    "vehicle_id": (vd.get("id") or entity.get("id") or "").strip(),
                    "vehicle_label": (vd.get("label") or "").strip(),
                    "trip_id": (trip.get("trip_id") or "").strip(),
                    "route_id": (trip.get("route_id") or "").strip(),
                    "start_date": (trip.get("start_date") or "").strip(),
                    "latitude": pos.get("latitude"),
                    "longitude": pos.get("longitude"),
                    "bearing": pos.get("bearing"),
                    "speed": pos.get("speed"),
                    "current_status": current_status,
                    "current_stop_sequence": vp.get("current_stop_sequence"),
                    "stop_id": (vp.get("stop_id") or "").strip(),
                    "timestamp": vp.get("timestamp"),
                    "congestion_level": congestion,
                    "occupancy_status": occupancy,
                    "occupancy_percentage": vp.get("occupancy_percentage"),
                })
        except Exception:
            pass

    df = pd.DataFrame(records)
    if not df.empty:
        for c in ["vehicle_id", "vehicle_label", "trip_id", "route_id", "start_date",
                  "current_status", "stop_id", "congestion_level", "occupancy_status"]:
            if c in df.columns:
                df[c] = df[c].astype("string")
        for c in ["current_stop_sequence", "timestamp"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in ["latitude", "longitude", "bearing", "speed", "occupancy_percentage"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------
# Chargement GTFS (filtré)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_trips_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    required = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    optional = ["wheelchair_accessible"]
    all_cols = required + optional
    dtypes = {
        "route_id": "string", "service_id": "string", "trip_id": "string",
        "trip_headsign": "string", "direction_id": "Int64", "shape_id": "string",
        "wheelchair_accessible": "Int64",
    }
    if not keep_trip_ids:
        return pd.DataFrame(columns=all_cols)
    available = _peek_csv_columns(gtfs_zip_bytes, "trips.txt")
    usecols = required + [c for c in optional if c in available]
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "trips.txt", usecols, dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=200_000, keep_order_cols=all_cols
    )


@st.cache_data(show_spinner=False)
def load_stop_times_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    dtypes = {"trip_id": "string", "arrival_time": "string", "departure_time": "string",
              "stop_id": "string", "stop_sequence": "Int64"}
    if not keep_trip_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stop_times.txt", usecols, dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=400_000, keep_order_cols=usecols
    )


@st.cache_data(show_spinner=False)
def load_stops_filtered(gtfs_zip_bytes: bytes, keep_stop_ids: Set[str]) -> pd.DataFrame:
    required = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    optional = ["wheelchair_boarding"]
    all_cols = required + optional
    dtypes = {
        "stop_id": "string", "stop_name": "string",
        "stop_lat": "float64", "stop_lon": "float64",
        "wheelchair_boarding": "Int64",
    }
    if not keep_stop_ids:
        return pd.DataFrame(columns=all_cols)
    available = _peek_csv_columns(gtfs_zip_bytes, "stops.txt")
    usecols = required + [c for c in optional if c in available]
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stops.txt", usecols, dtypes,
        filter_col="stop_id", filter_values=keep_stop_ids,
        chunksize=200_000, keep_order_cols=all_cols
    )


@st.cache_data(show_spinner=False)
def load_routes_filtered(gtfs_zip_bytes: bytes, keep_route_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "route_short_name", "route_long_name"]
    dtypes = {"route_id": "string", "route_short_name": "string", "route_long_name": "string"}
    if not keep_route_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "routes.txt", usecols, dtypes,
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
def fill_start_time_from_schedule(rt_trips: pd.DataFrame, stop_times_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit start_time dans rt_trips quand absent du flux RT.
    Stratégie : première arrival_time (ou departure_time) du trip dans stop_times,
    ordonné par stop_sequence — identique au fallback de gtfsrt_tripupdates_report.py.
    """
    if rt_trips.empty or stop_times_df.empty or "start_time" not in rt_trips.columns:
        return rt_trips

    mask_empty = rt_trips["start_time"].isna() | (rt_trips["start_time"].astype(str) == "")
    if not mask_empty.any():
        return rt_trips

    st = stop_times_df.copy()
    if "stop_sequence" in st.columns:
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        st = st.sort_values(["trip_id", "stop_sequence"])

    first_per_trip = st.groupby("trip_id", as_index=False).first()
    first_per_trip["fallback"] = first_per_trip.get("arrival_time", pd.NA)
    if "departure_time" in first_per_trip.columns:
        na_mask = first_per_trip["fallback"].isna() | (first_per_trip["fallback"].astype(str) == "")
        first_per_trip.loc[na_mask, "fallback"] = first_per_trip.loc[na_mask, "departure_time"]

    fallback_map: Dict[str, str] = dict(zip(
        first_per_trip["trip_id"].astype(str),
        first_per_trip["fallback"].astype(str),
    ))

    df = rt_trips.copy()
    df.loc[mask_empty, "start_time"] = df.loc[mask_empty, "trip_id"].astype(str).map(fallback_map)
    df["start_time"] = df["start_time"].fillna("").astype("string")
    return df


def compute_schedule_vs_rt(
    rt_stop_updates: pd.DataFrame,
    stop_times_filtered: pd.DataFrame,
    stops_df: pd.DataFrame
) -> pd.DataFrame:
    """Jointure RT vs schedule par arrêt (trip_id + stop_id, fallback stop_sequence)."""
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

    merged = pd.merge(left, sched, how="left", on=join_cols, suffixes=("_rt", "_sched"), copy=False)

    if not stops_df.empty and "stop_id" in merged.columns:
        merged = pd.merge(merged, stops_df[["stop_id", "stop_name"]], how="left", on="stop_id", copy=False)

    for c in ["arrival_time_sec", "departure_time_sec"]:
        if c in merged.columns:
            merged[c] = merged[c].astype("Int64")

    # Délai "best" natif RT (utile pour anomalies / debug)
    if "arrival_delay" in merged.columns and "departure_delay" in merged.columns:
        merged["delay_best"] = merged[["arrival_delay", "departure_delay"]].bfill(axis=1).iloc[:, 0]
        merged["delay_best"] = pd.to_numeric(merged["delay_best"], errors="coerce").astype("Int64")

    return merged


def attach_schedule_based_delay(
    sched_vs_rt: pd.DataFrame, rt_trips: pd.DataFrame, timezone: str = TIMEZONE
) -> pd.DataFrame:
    """
    Calcule 'delay_from_sched' par arrêt en comparant:
      RT (arrival/departure epoch)  VS  start_date (local tz) + HH:MM:SS planifié (stop_times)
    -> Résultat en secondes (Int64). Gère les colonnes suffixées (_rt).
    """
    if sched_vs_rt.empty:
        return sched_vs_rt

    df = sched_vs_rt.copy()

    # 1) start_date par trip
    if not rt_trips.empty and "trip_id" in df.columns and "trip_id" in rt_trips.columns:
        start_dates = rt_trips[["trip_id", "start_date"]].drop_duplicates()
        df = pd.merge(df, start_dates, on="trip_id", how="left")
    else:
        df["start_date"] = pd.NA

    # 2) secondes planifiées (arrival_time_sec prioritaire)
    sched_cols = [c for c in ["arrival_time_sec", "departure_time_sec"] if c in df.columns]
    if sched_cols:
        df["sched_sec"] = pd.to_numeric(df[sched_cols].bfill(axis=1).iloc[:, 0], errors="coerce")
    else:
        df["sched_sec"] = pd.NA

    # 3) datetime planifié local -> UTC
    start_local = pd.to_datetime(df["start_date"], format="%Y%m%d", errors="coerce")
    try:
        start_local = start_local.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        start_local = start_local.dt.tz_localize(timezone, errors="coerce")

    sched_dt_local = start_local + pd.to_timedelta(df["sched_sec"], unit="s")
    try:
        sched_dt_utc = sched_dt_local.dt.tz_convert("UTC")
    except Exception:
        sched_dt_utc = sched_dt_local

    # 4) datetime RT (epoch -> UTC) (arrival puis departure ; suffixes _rt pris en charge)
    rt_time_candidates = [c for c in ["arrival_time_rt", "arrival_time", "departure_time_rt", "departure_time"] if c in df.columns]
    if rt_time_candidates:
        rt_epoch = pd.to_numeric(df[rt_time_candidates].bfill(axis=1).iloc[:, 0], errors="coerce")
        df["rt_dt"] = pd.to_datetime(rt_epoch, unit="s", utc=True, errors="coerce")
    else:
        df["rt_dt"] = pd.NaT

    # 5) retard vs schedule
    delta = (df["rt_dt"] - sched_dt_utc).dt.total_seconds()
    df["delay_from_sched"] = pd.to_numeric(delta, errors="coerce").round().astype("Int64")

    return df


def summarize_trips(
    rt_trips: pd.DataFrame,
    rt_stop_updates: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> pd.DataFrame:
    """Résumé par trip: statut, nb updates, nb SKIPPED / NO_DATA, stats de délai (RT natif)."""
    if rt_trips.empty:
        return pd.DataFrame()

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
        out = pd.merge(out, trips_df[["trip_id", "route_id", "trip_headsign", "direction_id"]],
                       on="trip_id", how="left", suffixes=("", "_sched"))
    if not routes_df.empty and "route_id" in out.columns:
        out = pd.merge(out, routes_df, on="route_id", how="left")

    for c in ["avg_delay_sec", "max_delay_sec", "min_delay_sec"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    prefer = [
        "trip_id","route_id","route_short_name","route_long_name","trip_headsign","direction_id",
        "trip_status","is_deleted","updates_count","skipped_stops","nodata_stops",
        "avg_delay_sec","max_delay_sec","min_delay_sec","start_date","start_time","rt_timestamp"
    ]
    cols = [c for c in prefer if c in out.columns] + [c for c in out.columns if c not in prefer]
    return out[cols]


# ---------------------------------------------
# Anomalies
# ---------------------------------------------
def detect_anomalies(
    rt_stop_updates: pd.DataFrame,
    stops_df: pd.DataFrame,
    stop_times_df: pd.DataFrame,
    sched_vs_rt: pd.DataFrame,
    extreme_delay_threshold: int = 7200  # 2h
) -> pd.DataFrame:
    """
    Renvoie un DF : [anomaly_type, trip_id, stop_id, stop_sequence, details]
    Règles : stop_id inconnu, séquence impossible, arrivée<départ (RT),
             SCHEDULED sans correspondance schedule, stop_sequence manquant (SCHEDULED),
             retards extrêmes (|delay_best| > seuil).
    """
    anomalies: List[Dict] = []

    known_stops: Set[str] = set()
    if stops_df is not None and not stops_df.empty and "stop_id" in stops_df:
        known_stops = set(stops_df["stop_id"].dropna().astype(str).unique())

    su = rt_stop_updates.copy()

    # stop_id inconnu
    if "stop_id" in su.columns and known_stops:
        mask_unknown = ~su["stop_id"].isin(known_stops)
        for _, r in su[mask_unknown].iterrows():
            anomalies.append({
                "anomaly_type": "stop_id_inconnu",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": "stop_id absent de stops.txt"
            })

    # séquence impossible
    if "stop_sequence" in su.columns:
        su_seq = su.dropna(subset=["trip_id", "stop_sequence"]).copy()
        su_seq["stop_sequence"] = pd.to_numeric(su_seq["stop_sequence"], errors="coerce")
        su_seq = su_seq.dropna(subset=["stop_sequence"])
        # doublons
        su_seq_sorted = su_seq.sort_values(["trip_id", "stop_sequence"])
        dup_mask = su_seq_sorted.duplicated(subset=["trip_id", "stop_sequence"], keep=False)
        for _, r in su_seq_sorted[dup_mask].iterrows():
            anomalies.append({
                "anomaly_type": "sequence_impossible",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": "stop_sequence dupliqué pour ce trip"
            })
        # non-croissant (ordre original RT)
        su_order = su.dropna(subset=["trip_id", "stop_sequence"]).copy()
        su_order["stop_sequence"] = pd.to_numeric(su_order["stop_sequence"], errors="coerce")
        su_order["prev_seq"] = su_order.groupby("trip_id")["stop_sequence"].shift(1)
        bad_seq = su_order[(su_order["prev_seq"].notna()) & (su_order["stop_sequence"] <= su_order["prev_seq"])]
        for _, r in bad_seq.iterrows():
            anomalies.append({
                "anomaly_type": "sequence_impossible",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": f"stop_sequence non croissante (prev={int(r['prev_seq'])}, curr={int(r['stop_sequence'])})"
            })

    # arrivée < départ (RT)
    if "arrival_time" in su.columns and "departure_time" in su.columns:
        su_time = su.dropna(subset=["arrival_time", "departure_time"]).copy()
        su_time = su_time[(su_time["arrival_time"].astype("Int64").notna()) &
                          (su_time["departure_time"].astype("Int64").notna())]
        bad_time = su_time[su_time["arrival_time"] < su_time["departure_time"]]
        for _, r in bad_time.iterrows():
            anomalies.append({
                "anomaly_type": "arrivee_inferieure_depart_RT",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": f"arrival_time({r.get('arrival_time')}) < departure_time({r.get('departure_time')})"
            })

    # SCHEDULED sans correspondance schedule
    if sched_vs_rt is not None and not sched_vs_rt.empty:
        sched_missing = sched_vs_rt[
            (sched_vs_rt.get("stop_status") == "SCHEDULED") &
            (sched_vs_rt.get("arrival_time_sec").isna()) &
            (sched_vs_rt.get("departure_time_sec").isna())
        ]
        for _, r in sched_missing.iterrows():
            anomalies.append({
                "anomaly_type": "cle_schedule_introuvable",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": "Aucune ligne correspondante dans stop_times pour (trip_id, stop_id/seq)"
            })

    # stop_sequence manquant (SCHEDULED)
    if "stop_status" in su.columns:
        missing_seq = su[(su["stop_status"] == "SCHEDULED") & (su.get("stop_sequence").isna())]
        for _, r in missing_seq.iterrows():
            anomalies.append({
                "anomaly_type": "stop_sequence_manquant",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": "stop_sequence manquant pour un arrêt SCHEDULED"
            })

    # retards extrêmes (basés sur delay_best RT natif)
    if sched_vs_rt is not None and not sched_vs_rt.empty and "delay_best" in sched_vs_rt.columns:
        extreme = sched_vs_rt[sched_vs_rt["delay_best"].abs() > extreme_delay_threshold]
        for _, r in extreme.iterrows():
            anomalies.append({
                "anomaly_type": "retard_extreme",
                "trip_id": r.get("trip_id"),
                "stop_id": r.get("stop_id"),
                "stop_sequence": r.get("stop_sequence"),
                "details": f"delay_best={r.get('delay_best')}s > {extreme_delay_threshold}s"
            })

    if not anomalies:
        return pd.DataFrame(columns=["anomaly_type","trip_id","stop_id","stop_sequence","details"])

    df = pd.DataFrame(anomalies)
    for c in ["trip_id","stop_id","anomaly_type","details"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    if "stop_sequence" in df.columns:
        df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------
# Charts (Altair)
# ---------------------------------------------
def chart_anomalies_dedup(anomalies_df: pd.DataFrame):
    """Horizontal bar chart du nombre d'anomalies par type, après dé-duplication."""
    if not _HAS_ALTAIR or anomalies_df.empty:
        return None
    dedup = anomalies_df.drop_duplicates(subset=["anomaly_type","trip_id","stop_id","stop_sequence"])
    agg = dedup.groupby("anomaly_type", as_index=False).size().rename(columns={"size":"count"})
    return alt.Chart(agg).mark_bar().encode(
        y=alt.Y("anomaly_type:N", title=None, sort="-x"),
        x=alt.X("count:Q", title="Nombre (sans doublons)"),
        color=alt.Color("anomaly_type:N", legend=None),
        tooltip=[alt.Tooltip("anomaly_type:N", title="Type"), alt.Tooltip("count:Q", title="Nb")],
    ).properties(title="Anomalies (dé-dupliquées)").interactive()


def chart_stop_status_distribution(rt_su: pd.DataFrame):
    """Horizontal bar chart de la répartition des statuts d'arrêts."""
    if not _HAS_ALTAIR or rt_su.empty:
        return None
    df = rt_su.copy()
    df["stop_status"] = df.get("stop_status").fillna("UNKNOWN").astype(str)
    agg = df.groupby("stop_status", as_index=False).size().rename(columns={"size":"count"})
    return alt.Chart(agg).mark_bar().encode(
        y=alt.Y("stop_status:N", title=None, sort="-x"),
        x=alt.X("count:Q", title="Nombre d'arrêts"),
        color=alt.Color("stop_status:N", legend=None),
        tooltip=[alt.Tooltip("stop_status:N", title="Statut"), alt.Tooltip("count:Q", title="Nb")],
    ).properties(title="Répartition des statuts des arrêts").interactive()
    
def chart_delay_distribution_per_minute(sched_vs_rt_with_delay: pd.DataFrame):
    """
    Histogramme (pas 1 minute) de la répartition avance/retard vs schedule
    sur l'ensemble des arrêts, borné à [-30, +30] minutes.
    - Les données en dehors de cette plage sont EXCLUES.
    - On affiche toutes les minutes de -30 à +30, même sans occurrence.
    - Les bacs à 0 sont fortement dé‑accentués (couleur/opacity).
    """
    if not _HAS_ALTAIR or sched_vs_rt_with_delay.empty:
        return None

    df = sched_vs_rt_with_delay.copy()
    if "delay_from_sched" not in df.columns:
        return None

    # 1) Filtrer STRICTEMENT à [-30, +30] minutes (en secondes pour éviter l'effet d'arrondi)
    df = df[pd.notna(df["delay_from_sched"])]
    if df.empty:
        return None

    min_sec, max_sec = -30 * 60, 30 * 60
    df = df[(df["delay_from_sched"] >= min_sec) & (df["delay_from_sched"] <= max_sec)]
    if df.empty:
        return None

    # 2) Discrétisation au pas de 1 minute (arrondi à la minute la plus proche)
    df["delay_min"] = (df["delay_from_sched"].astype("float") / 60.0).round().astype("Int64")

    # 3) Agrégation + complétion de TOUTES les minutes [-30..30]
    agg = df.groupby("delay_min", as_index=False).size().rename(columns={"size": "count"})
    full_range = pd.DataFrame({"delay_min": list(range(-30, 31))})
    full = full_range.merge(agg, on="delay_min", how="left")
    full["count"] = full["count"].fillna(0).astype("Int64")
    full["is_zero"] = (full["count"] == 0)

    # 4) Graphique : barres avec dé‑accentuation pour les zéros + règle verticale à 0
    bars = alt.Chart(full).mark_bar().encode(
        x=alt.X(
            "delay_min:Q",
            title="Délai vs schedule (minutes) — avance < 0 | retard > 0",
            scale=alt.Scale(domain=[-30, 30], nice=False, zero=False),
        ),
        y=alt.Y("count:Q", title="Nombre d'arrêts"),
        # Dé‑accentuation visuelle des zéros
        color=alt.condition("datum.is_zero", alt.value("#d3d3d3"), alt.value("#1f77b4")),
        opacity=alt.condition("datum.is_zero", alt.value(0.25), alt.value(0.9)),
        tooltip=[
            alt.Tooltip("delay_min:Q", title="Minute"),
            alt.Tooltip("count:Q", title="Nb d'arrêts")
        ]
    ).properties(
        title="Répartition avance/retard par minute (tous arrêts, borné à [-30, +30])"
    )

    zero_rule = alt.Chart(pd.DataFrame({"delay_min": [0]})).mark_rule(
        color="black", strokeDash=[4, 4]
    ).encode(
        x=alt.X("delay_min:Q")
    )

    return (bars + zero_rule).interactive()


WHEELCHAIR_LABELS = {0: "Inconnu", 1: "Accessible ♿", 2: "Non accessible"}
WHEELCHAIR_BOARDING_LABELS = {0: "Inconnu / hérité", 1: "Accessible ♿", 2: "Non accessible"}
BIKES_LABELS = {0: "Inconnu", 1: "Vélos autorisés 🚲", 2: "Vélos interdits"}


def compute_accessibility(
    trips_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    rt_trips: pd.DataFrame,
    routes_df: pd.DataFrame = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calcule les stats d'accessibilité avec granularité par route/direction.

    Retourne un dict avec :
      - 'trips'                        : détail par voyage (avec route, direction, labels lisibles)
      - 'wheelchair_accessible_summary': comptages globaux par statut
      - 'wheelchair_boarding_summary'  : comptages arrêts par statut
      - 'wc_by_route'                  : accessibilité fauteuil roulant par ligne + direction
    """
    out: Dict[str, pd.DataFrame] = {}
    if routes_df is None:
        routes_df = pd.DataFrame()

    # --- Voyages ---
    if not trips_df.empty and not rt_trips.empty:
        trip_cols = ["trip_id"] + [c for c in ["route_id", "direction_id",
                                                 "wheelchair_accessible"]
                                    if c in trips_df.columns]
        merged = pd.merge(
            rt_trips[["trip_id", "trip_status"]].drop_duplicates("trip_id"),
            trips_df[trip_cols],
            on="trip_id", how="left",
        )
        # Joindre le nom de ligne
        if not routes_df.empty and "route_id" in merged.columns:
            r_cols = ["route_id"] + [c for c in ["route_short_name", "route_long_name"]
                                     if c in routes_df.columns]
            merged = pd.merge(merged, routes_df[r_cols], on="route_id", how="left")

        # Labels lisibles
        if "wheelchair_accessible" in merged.columns:
            merged["wheelchair_label"] = (merged["wheelchair_accessible"].fillna(0)
                                          .astype(int).map(WHEELCHAIR_LABELS).fillna("Inconnu"))
        if "bikes_allowed" in merged.columns:
            merged["bikes_label"] = (merged["bikes_allowed"].fillna(0)
                                     .astype(int).map(BIKES_LABELS).fillna("Inconnu"))

        out["trips"] = merged

        # Résumés globaux
        for col, label_col, labels in [
            ("wheelchair_accessible", "wheelchair_label", WHEELCHAIR_LABELS),
            ("bikes_allowed", "bikes_label", BIKES_LABELS),
        ]:
            if label_col in merged.columns:
                s = merged[label_col].value_counts().reset_index()
                s.columns = ["statut", "count"]
                out[f"{col}_summary"] = s

        # Granularité par route + direction
        route_label = next((c for c in ["route_short_name", "route_id"] if c in merged.columns), None)
        if route_label:
            for col, label_col in [("wheelchair_accessible", "wheelchair_label"),
                                    ("bikes_allowed", "bikes_label")]:
                if label_col not in merged.columns:
                    continue
                group_cols = [route_label]
                if "direction_id" in merged.columns:
                    group_cols.append("direction_id")
                by_route = (merged.groupby(group_cols + [label_col], as_index=False)
                            .size().rename(columns={"size": "count"}))
                by_route = by_route.rename(columns={label_col: "statut"})
                by_route["route_dir"] = by_route[route_label].astype(str) + (
                    (" — dir " + by_route["direction_id"].astype(str))
                    if "direction_id" in by_route.columns else ""
                )
                out[f"{col.replace('_accessible','').replace('_allowed','')}_by_route"] = by_route
    else:
        out["trips"] = pd.DataFrame()

    # --- Arrêts ---
    if not stops_df.empty and "wheelchair_boarding" in stops_df.columns:
        s = (stops_df["wheelchair_boarding"].fillna(0).astype(int)
             .map(WHEELCHAIR_BOARDING_LABELS).fillna("Inconnu"))
        out["stops"] = stops_df
        bd = s.value_counts().reset_index()
        bd.columns = ["statut", "count"]
        out["wheelchair_boarding_summary"] = bd
    else:
        out["stops"] = stops_df

    return out


def chart_accessibility_by_route(by_route_df: pd.DataFrame, title: str):
    """Horizontal stacked bar : accessibilité par ligne/direction."""
    if not _HAS_ALTAIR or by_route_df.empty or "route_dir" not in by_route_df.columns:
        return None
    n_routes = by_route_df["route_dir"].nunique()
    height = max(200, n_routes * 28)
    return (
        alt.Chart(by_route_df).mark_bar().encode(
            y=alt.Y("route_dir:N", title=None, sort="-x"),
            x=alt.X("count:Q", title="Nombre de voyages", stack="normalize",
                    axis=alt.Axis(format="%")),
            color=alt.Color("statut:N",
                            scale=alt.Scale(domain=list(ACC_COLOR_MAP.keys()),
                                            range=list(ACC_COLOR_MAP.values())),
                            title="Statut"),
            tooltip=[
                alt.Tooltip("route_dir:N", title="Ligne / Direction"),
                alt.Tooltip("statut:N", title="Statut"),
                alt.Tooltip("count:Q", title="Nb voyages"),
            ],
        ).properties(title=title, height=height).interactive()
    )


def compute_concordance(
    rt_trips: pd.DataFrame,
    vp_df: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Compare TripUpdates et VehiclePositions pour détecter les écarts :
      - Voyages présents dans un seul flux
      - Divergences route_id / start_date sur les trip_id communs
      - Véhicules VP sans trip_id
    """
    out: Dict[str, object] = {}

    tu_ids: Set[str] = set(rt_trips["trip_id"].dropna().astype(str).unique()) - {""}
    vp_ids: Set[str] = set(vp_df["trip_id"].dropna().astype(str).unique()) - {""}

    only_tu = tu_ids - vp_ids
    only_vp = vp_ids - tu_ids
    common  = tu_ids & vp_ids

    out["tu_total"]    = len(tu_ids)
    out["vp_total"]    = len(vp_ids)
    out["only_tu"]     = len(only_tu)
    out["only_vp"]     = len(only_vp)
    out["common"]      = len(common)

    # Véhicules sans trip_id
    vp_no_trip = vp_df[vp_df["trip_id"].fillna("") == ""]
    out["vp_no_trip_count"] = len(vp_no_trip)
    out["vp_no_trip_df"]    = vp_no_trip

    # Détails des trips uniquement dans TU
    tu_only_df = rt_trips[rt_trips["trip_id"].isin(only_tu)].copy()
    if not routes_df.empty and "route_id" in tu_only_df.columns:
        tu_only_df = pd.merge(tu_only_df, routes_df, on="route_id", how="left")
    out["only_tu_df"] = tu_only_df

    # Détails des trips uniquement dans VP
    vp_only_df = vp_df[vp_df["trip_id"].isin(only_vp)].copy()
    if not routes_df.empty and "route_id" in vp_only_df.columns:
        vp_only_df = pd.merge(vp_only_df, routes_df, on="route_id", how="left")
    out["only_vp_df"] = vp_only_df

    # Divergences sur les trips communs
    if common:
        tu_sub = (rt_trips[rt_trips["trip_id"].isin(common)]
                  [["trip_id", "route_id", "start_date", "trip_status"]]
                  .drop_duplicates("trip_id"))
        vp_sub = (vp_df[vp_df["trip_id"].isin(common)]
                  [["trip_id", "route_id", "start_date"]]
                  .drop_duplicates("trip_id"))
        div = pd.merge(tu_sub, vp_sub, on="trip_id", suffixes=("_tu", "_vp"))

        route_div = div[
            div["route_id_tu"].fillna("") != div["route_id_vp"].fillna("")
        ].copy()
        date_div = div[
            (div.get("start_date_tu", pd.Series(dtype=str)).fillna("") != "") &
            (div.get("start_date_vp", pd.Series(dtype=str)).fillna("") != "") &
            (div["start_date_tu"].fillna("") != div["start_date_vp"].fillna(""))
        ].copy() if "start_date_tu" in div.columns else pd.DataFrame()

        if not routes_df.empty:
            if not route_div.empty and "route_id_tu" in route_div.columns:
                route_div = pd.merge(route_div,
                                     routes_df.rename(columns={"route_id": "route_id_tu",
                                                                "route_short_name": "route_short_name_tu"}),
                                     on="route_id_tu", how="left")

        out["route_divergence_df"] = route_div
        out["date_divergence_df"]  = date_div
        out["route_div_count"]     = len(route_div)
        out["date_div_count"]      = len(date_div)

        # Divergences par ligne (pour le chart)
        if not route_div.empty:
            rname = "route_short_name_tu" if "route_short_name_tu" in route_div.columns else "route_id_tu"
            rd_agg = route_div.groupby(rname, as_index=False).size().rename(columns={"size": "count"})
            rd_agg.columns = ["route", "count"]
            out["route_div_by_route"] = rd_agg
        else:
            out["route_div_by_route"] = pd.DataFrame()
    else:
        out["route_divergence_df"] = pd.DataFrame()
        out["date_divergence_df"]  = pd.DataFrame()
        out["route_div_count"]     = 0
        out["date_div_count"]      = 0
        out["route_div_by_route"]  = pd.DataFrame()

    # Résumé par ligne : coverage TU vs VP
    if not trips_df.empty and "route_id" in trips_df.columns:
        tu_by_route = (rt_trips[rt_trips["trip_id"].isin(tu_ids)]
                       .merge(trips_df[["trip_id","route_id"]], on="trip_id", how="left",
                              suffixes=("","_sched"))
                       .groupby("route_id", as_index=False).size()
                       .rename(columns={"size": "tu_count"}))
        vp_by_route = (vp_df[vp_df["trip_id"].isin(vp_ids)]
                       .merge(trips_df[["trip_id","route_id"]], on="trip_id", how="left",
                              suffixes=("","_sched"))
                       .groupby("route_id", as_index=False).size()
                       .rename(columns={"size": "vp_count"}))
        coverage = pd.merge(tu_by_route, vp_by_route, on="route_id", how="outer").fillna(0)
        coverage["tu_count"] = coverage["tu_count"].astype(int)
        coverage["vp_count"] = coverage["vp_count"].astype(int)
        if not routes_df.empty:
            coverage = pd.merge(coverage, routes_df, on="route_id", how="left")
        out["coverage_by_route"] = coverage
    else:
        out["coverage_by_route"] = pd.DataFrame()

    return out


def chart_concordance_coverage(coverage_df: pd.DataFrame):
    """Horizontal grouped bar : nb trips TU vs VP par ligne."""
    if not _HAS_ALTAIR or coverage_df.empty:
        return None
    label_col = next((c for c in ["route_short_name", "route_id"] if c in coverage_df.columns), None)
    if not label_col:
        return None
    melted = coverage_df[[label_col, "tu_count", "vp_count"]].melt(
        id_vars=label_col, var_name="flux", value_name="count"
    )
    melted["flux"] = melted["flux"].map({"tu_count": "TripUpdates", "vp_count": "VehiclePositions"})
    n = melted[label_col].nunique()
    return (
        alt.Chart(melted).mark_bar().encode(
            y=alt.Y(f"{label_col}:N", title=None, sort="-x"),
            x=alt.X("count:Q", title="Nombre de voyages"),
            color=alt.Color("flux:N", title="Flux",
                            scale=alt.Scale(domain=["TripUpdates","VehiclePositions"],
                                            range=["#1f77b4","#ff7f0e"])),
            yOffset="flux:N",
            tooltip=[alt.Tooltip(f"{label_col}:N", title="Ligne"),
                     alt.Tooltip("flux:N", title="Flux"),
                     alt.Tooltip("count:Q", title="Nb voyages")],
        ).properties(title="Couverture par ligne (TU vs VP)", height=max(200, n * 45)).interactive()
    )


def chart_concordance_divergences(div_df: pd.DataFrame, title: str):
    """Horizontal bar : divergences par ligne."""
    if not _HAS_ALTAIR or div_df.empty or "count" not in div_df.columns:
        return None
    return (
        alt.Chart(div_df).mark_bar(color="#e74c3c").encode(
            y=alt.Y("route:N", title=None, sort="-x"),
            x=alt.X("count:Q", title="Nb divergences"),
            tooltip=[alt.Tooltip("route:N", title="Ligne"),
                     alt.Tooltip("count:Q", title="Nb")],
        ).properties(title=title, height=max(150, len(div_df) * 28)).interactive()
    )


ACC_COLOR_MAP = {
    "Accessible ♿": "#2ecc71",
    "Non accessible": "#e74c3c",
    "Inconnu": "#95a5a6",
    "Inconnu / hérité": "#95a5a6",
    "Vélos autorisés 🚲": "#3498db",
    "Vélos interdits": "#e67e22",
}


def _accessibility_bar(df: pd.DataFrame, title: str):
    """Horizontal bar chart générique statut → count pour l'accessibilité."""
    if not _HAS_ALTAIR or df.empty:
        return None
    return (
        alt.Chart(df.copy()).mark_bar().encode(
            y=alt.Y("statut:N", title=None, sort="-x"),
            x=alt.X("count:Q", title="Nombre"),
            color=alt.Color("statut:N",
                            scale=alt.Scale(domain=list(ACC_COLOR_MAP.keys()),
                                            range=list(ACC_COLOR_MAP.values())),
                            legend=None),
            tooltip=[alt.Tooltip("statut:N", title="Statut"),
                     alt.Tooltip("count:Q", title="Nb")],
        ).properties(title=title).interactive()
    )


def chart_vehicle_status_distribution(vp_df: pd.DataFrame):
    """Horizontal bar chart : répartition INCOMING_AT / STOPPED_AT / IN_TRANSIT_TO."""
    if not _HAS_ALTAIR or vp_df.empty or "current_status" not in vp_df.columns:
        return None
    agg = vp_df.groupby("current_status", as_index=False).size().rename(columns={"size": "count"})
    return alt.Chart(agg).mark_bar().encode(
        y=alt.Y("current_status:N", title=None, sort="-x"),
        x=alt.X("count:Q", title="Nombre de véhicules"),
        color=alt.Color("current_status:N", legend=None),
        tooltip=[alt.Tooltip("current_status:N", title="Statut"), alt.Tooltip("count:Q", title="Nb")],
    ).properties(title="Répartition des statuts de position").interactive()


def chart_occupancy_distribution(vp_df: pd.DataFrame):
    """Horizontal bar chart : répartition des taux d'occupation."""
    if not _HAS_ALTAIR or vp_df.empty or "occupancy_status" not in vp_df.columns:
        return None
    sub = vp_df.dropna(subset=["occupancy_status"])
    if sub.empty:
        return None
    agg = sub.groupby("occupancy_status", as_index=False).size().rename(columns={"size": "count"})
    ORDER = ["EMPTY", "MANY_SEATS_AVAILABLE", "FEW_SEATS_AVAILABLE", "STANDING_ROOM_ONLY",
             "CRUSHED_STANDING_ROOM_ONLY", "FULL", "NOT_ACCEPTING_PASSENGERS",
             "NO_DATA_AVAILABLE", "NOT_BOARDABLE"]
    agg["_order"] = agg["occupancy_status"].apply(lambda v: ORDER.index(v) if v in ORDER else 99)
    agg = agg.sort_values("_order").drop(columns="_order")
    return alt.Chart(agg).mark_bar().encode(
        y=alt.Y("occupancy_status:N", title=None, sort=None),
        x=alt.X("count:Q", title="Nombre de véhicules"),
        color=alt.Color("occupancy_status:N", legend=None),
        tooltip=[alt.Tooltip("occupancy_status:N", title="Occupation"), alt.Tooltip("count:Q", title="Nb")],
    ).properties(title="Répartition de l'occupation des véhicules").interactive()


def render_vehicles_map(vp_df: pd.DataFrame) -> None:
    """
    Carte pydeck avec fond OpenStreetMap (CARTO Light) des positions de véhicules.
    Couleur par statut courant. Tooltip avec détails du véhicule.
    Utilise st.pydeck_chart() directement (pas de valeur de retour).
    """
    try:
        import pydeck as pdk
    except ImportError:
        st.warning("pydeck non installé — `pip install pydeck`")
        return

    sub = vp_df.dropna(subset=["latitude", "longitude"]).copy()
    if sub.empty:
        st.info("Aucune position GPS disponible.")
        return

    sub["lat"] = sub["latitude"].astype(float)
    sub["lon"] = sub["longitude"].astype(float)

    # Couleur RGB par statut
    STATUS_COLOR = {
        "INCOMING_AT":  [255, 165,   0],   # orange
        "STOPPED_AT":   [ 46, 204, 113],   # vert
        "IN_TRANSIT_TO":[52, 152, 219],    # bleu
    }
    default_color = [150, 150, 150]
    sub["_color"] = sub["current_status"].map(
        lambda s: STATUS_COLOR.get(str(s), default_color)
    )

    # Centre automatique sur les données
    center_lat = float(sub["lat"].median())
    center_lon = float(sub["lon"].median())

    tooltip_fields = {c: c for c in ["vehicle_id", "vehicle_label", "route_id", "trip_id",
                                      "current_status", "occupancy_status", "speed", "bearing"]
                      if c in sub.columns}
    tooltip_html = "<br>".join(f"<b>{k}:</b> {{{v}}}" for k, v in tooltip_fields.items())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sub,
        get_position=["lon", "lat"],
        get_fill_color="_color",
        get_radius=80,
        radius_min_pixels=5,
        radius_max_pixels=18,
        pickable=True,
        opacity=0.85,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={"html": tooltip_html, "style": {"fontSize": "12px"}},
    )

    st.pydeck_chart(deck, use_container_width=True)


# ---------------------------------------------
# UI Streamlit (simplifiée + graphiques)
# ---------------------------------------------
def main():
    st.title("🚌 Analyse GTFS-RT vs GTFS statique")
    st.caption("TripUpdates & VehiclePositions — synthèse, anomalies, graphiques. Optimisé pour gros GTFS.")

    cfg = _load_api_config()

    # Init session_state pour les flux fetchés
    for key in ("fetched_gtfs_bytes", "fetched_rt_bytes", "fetched_vp_bytes",
                "fetched_rt_label", "fetched_vp_label", "fetched_gtfs_label"):
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.header("Fichiers d'entrée")

        # --- GTFS statique ---
        gtfs_zip = st.file_uploader("GTFS statique (.zip)", type=["zip"])
        if cfg["gtfs_zip_url"]:
            if st.button("Récupérer le GTFS statique", use_container_width=True):
                with st.spinner("Téléchargement GTFS statique…"):
                    try:
                        st.session_state.fetched_gtfs_bytes = fetch_feed(cfg["gtfs_zip_url"], cfg)
                        st.session_state.fetched_gtfs_label = f"GTFS statique — {len(st.session_state.fetched_gtfs_bytes) // 1024} Ko"
                        st.success("GTFS statique récupéré ✅")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            if st.session_state.fetched_gtfs_bytes:
                st.caption(f"✅ {st.session_state.fetched_gtfs_label}")

        st.divider()

        # --- TripUpdates ---
        rt_file = st.file_uploader("TripUpdates (.pb/.bin/.json)", type=["pb", "bin", "json"])
        if cfg["tripupdates_url"]:
            if st.button("Récupérer les TripUpdates", use_container_width=True):
                with st.spinner("Téléchargement TripUpdates…"):
                    try:
                        st.session_state.fetched_rt_bytes = fetch_feed(cfg["tripupdates_url"], cfg)
                        st.session_state.fetched_rt_label = f"TripUpdates — {len(st.session_state.fetched_rt_bytes) // 1024} Ko"
                        st.success("TripUpdates récupérés ✅")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            if st.session_state.fetched_rt_bytes:
                st.caption(f"✅ {st.session_state.fetched_rt_label}")

        st.divider()

        # --- VehiclePositions ---
        vp_file = st.file_uploader("VehiclePositions (.pb/.bin/.json)", type=["pb", "bin", "json"])
        if cfg["vehiclepositions_url"]:
            if st.button("Récupérer les VehiclePositions", use_container_width=True):
                with st.spinner("Téléchargement VehiclePositions…"):
                    try:
                        st.session_state.fetched_vp_bytes = fetch_feed(cfg["vehiclepositions_url"], cfg)
                        st.session_state.fetched_vp_label = f"VehiclePositions — {len(st.session_state.fetched_vp_bytes) // 1024} Ko"
                        st.success("VehiclePositions récupérées ✅")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            if st.session_state.fetched_vp_bytes:
                st.caption(f"✅ {st.session_state.fetched_vp_label}")

        st.divider()
        st.subheader("Options")
        focus_only_rt_trips = st.checkbox("Se focaliser sur les voyages du TripUpdate (recommandé)", value=True)
        limit_trips = st.number_input("Limiter à N trips (optionnel)", min_value=0, value=0, step=100)
        show_raw_tables = st.checkbox("Afficher tables brutes (debug)", value=False)

        st.divider()
        run_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if not run_btn:
        st.info("💡 Charge les fichiers (ou récupère-les via l'API) puis clique sur **Lancer l'analyse**.")
        return

    # Résolution des bytes : fichier uploadé > donnée fetchée en session
    gtfs_bytes = (
        (gtfs_zip.getvalue() if hasattr(gtfs_zip, "getvalue") else gtfs_zip.read())
        if gtfs_zip else st.session_state.fetched_gtfs_bytes
    )
    rt_bytes = (
        (rt_file.getvalue() if hasattr(rt_file, "getvalue") else rt_file.read())
        if rt_file else st.session_state.fetched_rt_bytes
    )
    vp_bytes = (
        (vp_file.getvalue() if hasattr(vp_file, "getvalue") else vp_file.read())
        if vp_file else st.session_state.fetched_vp_bytes
    )

    if not rt_bytes and not vp_bytes:
        st.error("Aucun flux GTFS-RT disponible. Charge un fichier ou configure l'API dans le .env.")
        return

    # Tabs
    tabs = []
    if rt_bytes:
        tabs.append("TripUpdates")
    if vp_bytes:
        tabs.append("VehiclePositions")
    if rt_bytes and vp_bytes:
        tabs.append("Concordance TU ↔ VP")
    if not tabs:
        st.error("Aucun fichier GTFS-RT chargé.")
        return

    tab_objects = st.tabs(tabs)

    # ------------------------------------------------------------------
    # Onglet TripUpdates
    # ------------------------------------------------------------------
    if rt_bytes:
        tab_tu = tab_objects[tabs.index("TripUpdates")]
        with tab_tu:
            with st.status("📥 Lecture du TripUpdate...", expanded=False) as status:
                rt_trips, rt_su, rt_trip_ids = parse_tripupdates_rt(rt_bytes)
                status.update(label=f"TripUpdate: {len(rt_trips)} trips, {len(rt_su)} stop updates.", state="complete")

            if gtfs_bytes:
                # Filtrage ciblé
                if focus_only_rt_trips:
                    keep_trip_ids = {tid for tid in rt_trip_ids if isinstance(tid, str) and tid}
                    if limit_trips and limit_trips > 0 and keep_trip_ids:
                        keep_trip_ids = set(list(keep_trip_ids)[: int(limit_trips)])
                else:
                    st.warning("⚠️ Filtrage par trips RT désactivé — utilisation mémoire potentiellement élevée.")
                    keep_trip_ids = set()

                progress = st.progress(0, text="Chargement trips...")
                trips_df = load_trips_filtered(gtfs_bytes, keep_trip_ids)
                progress.progress(25, text="Chargement stop_times...")
                stop_times_df = load_stop_times_filtered(gtfs_bytes, keep_trip_ids)

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
                rt_trips = fill_start_time_from_schedule(rt_trips, stop_times_df)
                sched_vs_rt = compute_schedule_vs_rt(rt_su, stop_times_df, stops_df)
                sched_vs_rt = attach_schedule_based_delay(sched_vs_rt, rt_trips, timezone=TIMEZONE)
                progress.progress(100, text="Synthèse...")
                summary_df = summarize_trips(rt_trips, rt_su, trips_df, routes_df)
            else:
                trips_df = pd.DataFrame()
                stop_times_df = pd.DataFrame()
                stops_df = pd.DataFrame()
                routes_df = pd.DataFrame()
                sched_vs_rt = pd.DataFrame()
                summary_df = summarize_trips(rt_trips, rt_su, pd.DataFrame(), pd.DataFrame())

            st.success("Analyse TripUpdates terminée ✅")

            # Synthèse
            canceled_total = int((rt_trips["trip_status"] == "CANCELED").sum()) if "trip_status" in rt_trips.columns else 0
            added_total = int((rt_trips["trip_status"] == "ADDED").sum()) if "trip_status" in rt_trips.columns else 0
            new_total = int((rt_trips["trip_status"] == "NEW").sum()) if "trip_status" in rt_trips.columns else 0

            partially_canceled = 0
            if not rt_su.empty and "stop_status" in rt_su.columns:
                trips_with_skipped = set(rt_su.loc[rt_su["stop_status"] == "SKIPPED", "trip_id"].dropna().astype(str))
                if "trip_status" in rt_trips.columns:
                    not_canceled_trips = set(rt_trips.loc[rt_trips["trip_status"] != "CANCELED", "trip_id"].dropna().astype(str))
                    partially_canceled = len(trips_with_skipped & not_canceled_trips)
                else:
                    partially_canceled = len(trips_with_skipped)

            if not trips_df.empty and "trip_id" in trips_df.columns:
                known_trip_ids = set(trips_df["trip_id"].dropna().astype(str))
                unknown_trip_ids = sorted(tid for tid in rt_trip_ids if tid not in known_trip_ids)
            else:
                unknown_trip_ids = []

            skipped_total = int((rt_su["stop_status"] == "SKIPPED").sum()) if (not rt_su.empty and "stop_status" in rt_su.columns) else 0
            trips_count = rt_trips["trip_id"].nunique() if not rt_trips.empty else 0

            k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
            with k1: st.metric("CANCELED", f"{canceled_total:,}")
            with k2: st.metric("ADDED", f"{added_total:,}")
            with k3: st.metric("NEW", f"{new_total:,}")
            with k4: st.metric("Partiellement annulés", f"{partially_canceled:,}")
            with k5: st.metric("Trip IDs inconnus", f"{len(unknown_trip_ids):,}")
            with k6: st.metric("Trips RT", f"{trips_count:,}")
            with k7: st.metric("Arrêts SKIPPED", f"{skipped_total:,}")

            if unknown_trip_ids:
                with st.expander("Voir la liste des trip_id inconnus"):
                    st.write(unknown_trip_ids)

            st.subheader("Résumé par voyage")
            if summary_df.empty:
                st.info("Aucun résumé disponible.")
            else:
                order_cols = [c for c in [
                    "trip_id","route_id","route_short_name","route_long_name","trip_headsign","direction_id",
                    "trip_status","is_deleted","updates_count","skipped_stops","nodata_stops",
                    "avg_delay_sec","max_delay_sec","min_delay_sec","start_date","start_time","rt_timestamp"
                ] if c in summary_df.columns]
                st.dataframe(
                    summary_df[order_cols].sort_values(["trip_status","route_id","trip_id"], na_position="last"),
                    use_container_width=True, height=360
                )

            st.subheader("Anomalies détectées")
            anomalies_df = detect_anomalies(rt_su, stops_df, stop_times_df, sched_vs_rt)
            if anomalies_df.empty:
                st.success("Aucune anomalie détectée selon les règles en vigueur.")
            else:
                st.dataframe(anomalies_df, use_container_width=True, height=300)

            st.subheader("Visualisations")
            if not _HAS_ALTAIR:
                st.warning("Altair n'est pas installé — `pip install altair` pour activer les graphiques.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    ch1 = chart_anomalies_dedup(anomalies_df)
                    if ch1 is not None:
                        st.altair_chart(ch1, use_container_width=True)
                    else:
                        st.info("Aucune anomalie (ou aucune après dé-duplication).")
                with c2:
                    ch2 = chart_stop_status_distribution(rt_su)
                    if ch2 is not None:
                        st.altair_chart(ch2, use_container_width=True)
                    else:
                        st.info("Aucun statut d'arrêt exploitable.")

                ch3 = chart_delay_distribution_per_minute(sched_vs_rt)
                if ch3 is not None:
                    st.altair_chart(ch3, use_container_width=True)
                else:
                    st.info("Impossible de calculer la répartition par minute (données manquantes ou GTFS statique non fourni).")

            # -------------------------
            # Accessibilité
            # -------------------------
            if gtfs_bytes and (not trips_df.empty or not stops_df.empty):
                st.subheader("Accessibilité")
                acc = compute_accessibility(trips_df, stops_df, rt_trips, routes_df)

                has_wc   = "wheelchair_accessible_summary" in acc and not acc["wheelchair_accessible_summary"].empty
                has_bike = "bikes_allowed_summary" in acc and not acc["bikes_allowed_summary"].empty
                has_stop = "wheelchair_boarding_summary" in acc and not acc["wheelchair_boarding_summary"].empty

                if not has_wc and not has_bike and not has_stop:
                    st.info("Données d'accessibilité absentes du GTFS statique (champs optionnels).")
                else:
                    # Métriques synthétiques
                    if has_wc:
                        wc_df = acc["wheelchair_accessible_summary"]
                        total_trips_acc = int(wc_df.loc[wc_df["statut"] == "Accessible ♿", "count"].sum())
                        total_trips_noacc = int(wc_df.loc[wc_df["statut"] == "Non accessible", "count"].sum())
                        total_trips_unk = int(wc_df.loc[wc_df["statut"] == "Inconnu", "count"].sum())
                        ma1, ma2, ma3 = st.columns(3)
                        with ma1: st.metric("Voyages accessibles ♿", f"{total_trips_acc:,}")
                        with ma2: st.metric("Voyages non accessibles", f"{total_trips_noacc:,}")
                        with ma3: st.metric("Voyages sans info", f"{total_trips_unk:,}")

                    # Charts
                    chart_cols = [c for c in [
                        ("wheelchair_accessible_summary", "Accessibilité voyages (fauteuil roulant)") if has_wc else None,
                        ("bikes_allowed_summary", "Vélos à bord") if has_bike else None,
                        ("wheelchair_boarding_summary", "Accessibilité arrêts (fauteuil roulant)") if has_stop else None,
                    ] if c is not None]

                    cols_ui = st.columns(len(chart_cols)) if chart_cols else []
                    for (key, title), col_ui in zip(chart_cols, cols_ui):
                        with col_ui:
                            ch = _accessibility_bar(acc[key], title)
                            if ch:
                                st.altair_chart(ch, use_container_width=True)

                    # Granularité par ligne
                    ch_wc_route = chart_accessibility_by_route(
                        acc.get("wheelchair_by_route", pd.DataFrame()),
                        "Accessibilité ♿ par ligne / direction"
                    )
                    if ch_wc_route:
                        st.altair_chart(ch_wc_route, use_container_width=True)

                    ch_bike_route = chart_accessibility_by_route(
                        acc.get("bikes_by_route", pd.DataFrame()),
                        "Vélos à bord par ligne / direction"
                    )
                    if ch_bike_route:
                        st.altair_chart(ch_bike_route, use_container_width=True)

                    # Détail tabulaire par voyage
                    if not acc["trips"].empty:
                        display_cols = [c for c in [
                            "trip_id", "route_id", "route_short_name", "direction_id",
                            "trip_status", "wheelchair_label", "bikes_label",
                        ] if c in acc["trips"].columns]
                        with st.expander("Détail accessibilité par voyage"):
                            st.dataframe(acc["trips"][display_cols], use_container_width=True, height=300)

            if show_raw_tables:
                st.divider()
                st.subheader("Debug: tables brutes")
                with st.expander("RT: Trips"):
                    st.dataframe(rt_trips, use_container_width=True, height=220)
                with st.expander("RT: Stop Updates"):
                    st.dataframe(rt_su, use_container_width=True, height=220)
                with st.expander("GTFS: stop_times (filtré)"):
                    st.dataframe(stop_times_df.head(1000), use_container_width=True, height=220)
                with st.expander("Join: sched_vs_rt (+ delay_from_sched)"):
                    st.dataframe(sched_vs_rt.head(1000), use_container_width=True, height=220)

    # ------------------------------------------------------------------
    # Onglet VehiclePositions
    # ------------------------------------------------------------------
    if vp_bytes:
        tab_vp = tab_objects[tabs.index("VehiclePositions")]
        with tab_vp:
            with st.status("📥 Lecture des VehiclePositions...", expanded=False) as status:
                vp_df = parse_vehiclepositions_rt(vp_bytes)
                status.update(label=f"VehiclePositions: {len(vp_df)} véhicules.", state="complete")

            if vp_df.empty:
                st.warning("Aucun véhicule trouvé dans le fichier VehiclePositions.")
            else:
                st.success(f"Analyse VehiclePositions terminée ✅ — {len(vp_df):,} véhicules")

                # Métriques
                total_veh = len(vp_df)
                has_pos = int(vp_df[["latitude", "longitude"]].notna().all(axis=1).sum()) if "latitude" in vp_df.columns else 0
                has_occ = int(vp_df["occupancy_status"].notna().sum()) if "occupancy_status" in vp_df.columns else 0
                unique_routes = vp_df["route_id"].nunique() if "route_id" in vp_df.columns else 0

                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("Véhicules", f"{total_veh:,}")
                with m2: st.metric("Avec position GPS", f"{has_pos:,}")
                with m3: st.metric("Avec donnée occupation", f"{has_occ:,}")
                with m4: st.metric("Lignes distinctes", f"{unique_routes:,}")

                # Carte
                st.subheader("Carte des véhicules")
                render_vehicles_map(vp_df)

                # Graphiques
                st.subheader("Visualisations")
                if not _HAS_ALTAIR:
                    st.warning("Altair n'est pas installé.")
                else:
                    cv1, cv2 = st.columns(2)
                    with cv1:
                        ch_status = chart_vehicle_status_distribution(vp_df)
                        if ch_status is not None:
                            st.altair_chart(ch_status, use_container_width=True)
                        else:
                            st.info("Statuts non disponibles.")
                    with cv2:
                        ch_occ = chart_occupancy_distribution(vp_df)
                        if ch_occ is not None:
                            st.altair_chart(ch_occ, use_container_width=True)
                        else:
                            st.info("Données d'occupation non disponibles.")

                # Table
                st.subheader("Détail par véhicule")
                col_order = [c for c in [
                    "vehicle_id", "vehicle_label", "route_id", "trip_id",
                    "current_status", "stop_id", "current_stop_sequence",
                    "latitude", "longitude", "bearing", "speed",
                    "occupancy_status", "occupancy_percentage",
                    "congestion_level", "timestamp", "start_date",
                ] if c in vp_df.columns]
                st.dataframe(vp_df[col_order], use_container_width=True, height=400)

                # -------------------------
                # Accessibilité VP
                # -------------------------
                if gtfs_bytes:
                    # Charger trips pour les véhicules du flux VP
                    vp_trip_ids: Set[str] = set(
                        vp_df["trip_id"].dropna().astype(str).unique()
                    ) - {""}
                    if vp_trip_ids:
                        with st.spinner("Chargement accessibilité…"):
                            vp_trips_df = load_trips_filtered(gtfs_bytes, vp_trip_ids)
                            vp_stops_df = pd.DataFrame()  # stops VP non chargés (optionnel)

                        vp_routes_df = load_routes_filtered(gtfs_bytes, set(vp_trips_df["route_id"].dropna().astype(str).unique()) - {""}) if not vp_trips_df.empty else pd.DataFrame()
                        acc_vp = compute_accessibility(vp_trips_df, vp_stops_df, vp_df.rename(columns={"current_status": "trip_status"}), vp_routes_df)

                        has_wc_vp   = "wheelchair_accessible_summary" in acc_vp and not acc_vp["wheelchair_accessible_summary"].empty
                        has_bike_vp = "bikes_allowed_summary" in acc_vp and not acc_vp["bikes_allowed_summary"].empty

                        if has_wc_vp or has_bike_vp:
                            st.subheader("Accessibilité des véhicules")
                            if has_wc_vp:
                                wc_vp = acc_vp["wheelchair_accessible_summary"]
                                mv1, mv2, mv3 = st.columns(3)
                                with mv1: st.metric("Véhicules sur voyage accessible ♿", f"{int(wc_vp.loc[wc_vp['statut']=='Accessible ♿','count'].sum()):,}")
                                with mv2: st.metric("Véhicules non accessibles", f"{int(wc_vp.loc[wc_vp['statut']=='Non accessible','count'].sum()):,}")
                                with mv3: st.metric("Véhicules sans info", f"{int(wc_vp.loc[wc_vp['statut']=='Inconnu','count'].sum()):,}")

                            chart_cols_vp = [c for c in [
                                ("wheelchair_accessible_summary", "Accessibilité (fauteuil roulant)") if has_wc_vp else None,
                                ("bikes_allowed_summary", "Vélos à bord") if has_bike_vp else None,
                            ] if c is not None]
                            cols_vp = st.columns(len(chart_cols_vp))
                            for (key, title), col_ui in zip(chart_cols_vp, cols_vp):
                                with col_ui:
                                    ch = _accessibility_bar(acc_vp[key], title)
                                    if ch:
                                        st.altair_chart(ch, use_container_width=True)
                        else:
                            st.info("Données d'accessibilité absentes du GTFS statique pour ces véhicules.")

                if show_raw_tables:
                    st.divider()
                    with st.expander("Debug: DataFrame brut VehiclePositions"):
                        st.dataframe(vp_df, use_container_width=True, height=300)

    # ------------------------------------------------------------------
    # Onglet Concordance TU ↔ VP
    # ------------------------------------------------------------------
    if rt_bytes and vp_bytes:
        tab_conc = tab_objects[tabs.index("Concordance TU ↔ VP")]
        with tab_conc:
            # On a besoin des deux DataFrames — ils sont déjà calculés dans leurs onglets
            # respectifs, mais on doit s'assurer qu'ils existent dans ce scope.
            # On les re-parse (cache Streamlit évite le double calcul).
            with st.status("Calcul de la concordance…", expanded=False) as s_conc:
                _rt_trips_c, _rt_su_c, _rt_ids_c = parse_tripupdates_rt(rt_bytes)
                _vp_df_c = parse_vehiclepositions_rt(vp_bytes)

                _trips_df_c  = pd.DataFrame()
                _routes_df_c = pd.DataFrame()
                if gtfs_bytes:
                    _keep = ({tid for tid in _rt_ids_c if tid} |
                             set(_vp_df_c["trip_id"].dropna().astype(str).unique()) - {""})
                    _trips_df_c  = load_trips_filtered(gtfs_bytes, _keep)
                    _rids = set(_trips_df_c["route_id"].dropna().astype(str).unique()) - {""}
                    _routes_df_c = load_routes_filtered(gtfs_bytes, _rids)

                conc = compute_concordance(_rt_trips_c, _vp_df_c, _trips_df_c, _routes_df_c)
                s_conc.update(label="Concordance calculée.", state="complete")

            # Métriques
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Voyages TripUpdates", f"{conc['tu_total']:,}")
            with c2: st.metric("Voyages VehiclePositions", f"{conc['vp_total']:,}")
            with c3: st.metric("Trip IDs communs", f"{conc['common']:,}")
            with c4: st.metric("Uniquement dans TU", f"{conc['only_tu']:,}", delta=None)
            with c5: st.metric("Uniquement dans VP", f"{conc['only_vp']:,}", delta=None)

            c6, c7 = st.columns(2)
            with c6: st.metric("Divergences route_id", f"{conc.get('route_div_count', 0):,}")
            with c7: st.metric("Véhicules VP sans trip_id", f"{conc.get('vp_no_trip_count', 0):,}")

            st.divider()

            # Couverture par ligne
            ch_cov = chart_concordance_coverage(conc.get("coverage_by_route", pd.DataFrame()))
            if ch_cov:
                st.subheader("Couverture par ligne")
                st.altair_chart(ch_cov, use_container_width=True)

            # Divergences route_id par ligne
            ch_div = chart_concordance_divergences(
                conc.get("route_div_by_route", pd.DataFrame()),
                "Divergences route_id par ligne (trips communs)"
            )
            if ch_div:
                st.subheader("Divergences route_id")
                st.altair_chart(ch_div, use_container_width=True)

            st.divider()

            # Tables détaillées
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Uniquement dans TripUpdates ({conc['only_tu']:,})")
                df_only_tu = conc.get("only_tu_df", pd.DataFrame())
                if df_only_tu.empty:
                    st.info("Aucun.")
                else:
                    show_cols = [c for c in ["trip_id","route_id","route_short_name",
                                              "trip_status","start_date"] if c in df_only_tu.columns]
                    st.dataframe(df_only_tu[show_cols], use_container_width=True, height=300)

            with col_b:
                st.subheader(f"Uniquement dans VehiclePositions ({conc['only_vp']:,})")
                df_only_vp = conc.get("only_vp_df", pd.DataFrame())
                if df_only_vp.empty:
                    st.info("Aucun.")
                else:
                    show_cols = [c for c in ["trip_id","vehicle_id","route_id","route_short_name",
                                              "current_status","start_date"] if c in df_only_vp.columns]
                    st.dataframe(df_only_vp[show_cols], use_container_width=True, height=300)

            route_div_df = conc.get("route_divergence_df", pd.DataFrame())
            if not route_div_df.empty:
                with st.expander(f"Détail divergences route_id ({len(route_div_df):,} trips)"):
                    st.dataframe(route_div_df, use_container_width=True, height=300)

            date_div_df = conc.get("date_divergence_df", pd.DataFrame())
            if not date_div_df.empty:
                with st.expander(f"Détail divergences start_date ({len(date_div_df):,} trips)"):
                    st.dataframe(date_div_df, use_container_width=True, height=300)

            vp_no_trip_df = conc.get("vp_no_trip_df", pd.DataFrame())
            if not vp_no_trip_df.empty:
                with st.expander(f"Véhicules VP sans trip_id ({len(vp_no_trip_df):,})"):
                    show_cols = [c for c in ["vehicle_id","vehicle_label","route_id",
                                              "current_status","latitude","longitude"]
                                 if c in vp_no_trip_df.columns]
                    st.dataframe(vp_no_trip_df[show_cols], use_container_width=True, height=250)

    # Mémoire
    del gtfs_bytes, rt_bytes, vp_bytes
    gc.collect()


if __name__ == "__main__":
    main()






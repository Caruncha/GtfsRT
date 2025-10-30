# app.py
# ---------------------------------------------------------
# Streamlit - Analyse GTFS-RT TripUpdates vs GTFS statique
# Version simplifiée + visualisations demandées :
#   - Synthèse (voyages annulés & partiellement annulés)
#   - Résumé par voyage
#   - Anomalies (table) + GRAPHIQUES :
#       * Anomalies (sans doublons)
#       * Répartition des statuts d'arrêts
#       * Retards moyens (par route si dispo, sinon global)
# Retiré précédemment : téléchargements, détails par arrêt, autres graphiques.
# Optimisé gros GTFS (filtrage par trips RT, lecture par chunks, dtypes compacts).
# ---------------------------------------------------------
# Dépendances :
#   pip install streamlit pandas altair gtfs-realtime-bindings protobuf
# ---------------------------------------------------------

from __future__ import annotations

import io
import gc
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# Altair (charts)
try:
    import altair as alt
    _HAS_ALTAIR = True
    alt.data_transformers.disable_max_rows()
except Exception:
    _HAS_ALTAIR = False

# GTFS-RT protobuf (optionnel; fallback JSON si absent)
try:
    from google.transit import gtfs_realtime_pb2  # type: ignore
    _HAS_GTFS_RT = True
except Exception:
    _HAS_GTFS_RT = False


# ---------------------------------------------
# Config Streamlit
# ---------------------------------------------
st.set_page_config(
    page_title="Analyse GTFS-RT TripUpdates",
    page_icon="🚌",
    layout="wide"
)

# ---------------------------------------------
# Constantes & utilitaires
# ---------------------------------------------
TRIP_SCHED_REL = {0: "SCHEDULED", 1: "ADDED", 2: "UNSCHEDULED", 3: "CANCELED"}
STOP_SCHED_REL = {0: "SCHEDULED", 1: "SKIPPED", 2: "NO_DATA"}


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
# Chargement GTFS (filtré)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_trips_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    dtypes = {"route_id": "string", "service_id": "string", "trip_id": "string",
              "trip_headsign": "string", "direction_id": "Int64", "shape_id": "string"}
    if not keep_trip_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "trips.txt", usecols, dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=200_000, keep_order_cols=usecols
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
    usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    dtypes = {"stop_id": "string", "stop_name": "string", "stop_lat": "float64", "stop_lon": "float64"}
    if not keep_stop_ids:
        return pd.DataFrame(columns=usecols)
    return _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stops.txt", usecols, dtypes,
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
    """Résumé par trip: statut, nb updates, nb SKIPPED / NO_DATA, stats de délai."""
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
        # non-croissant (dans l'ordre original)
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

    # retards extrêmes
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
def _route_label(df: pd.DataFrame) -> pd.Series:
    if "route_short_name" in df.columns:
        lbl = df["route_short_name"].fillna(df.get("route_long_name"))
    else:
        lbl = df.get("route_long_name", pd.Series([""]*len(df)))
    return lbl.fillna(df.get("route_id")).astype(str)


def chart_anomalies_dedup(anomalies_df: pd.DataFrame):
    """Bar chart du nombre d'anomalies par type, après dé-duplication logique."""
    if not _HAS_ALTAIR or anomalies_df.empty:
        return None
    # Dé-dup: on considère (anomaly_type, trip_id, stop_id, stop_sequence) comme clé unique
    dedup = anomalies_df.drop_duplicates(subset=["anomaly_type","trip_id","stop_id","stop_sequence"])
    agg = dedup.groupby("anomaly_type", as_index=False).size().rename(columns={"size":"count"})
    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X("anomaly_type:N", title="Type d'anomalie"),
        y=alt.Y("count:Q", title="Nombre (sans doublons)"),
        color=alt.Color("anomaly_type:N", legend=None),
        tooltip=[alt.Tooltip("anomaly_type:N", title="Type"), alt.Tooltip("count:Q", title="Nb")]
    ).properties(title="Anomalies (dé-dupliquées)")
    return chart.interactive()


def chart_stop_status_distribution(rt_su: pd.DataFrame):
    """Bar chart de la répartition des statuts d'arrêts (SCHEDULED, SKIPPED, NO_DATA, ...)."""
    if not _HAS_ALTAIR or rt_su.empty:
        return None
    df = rt_su.copy()
    df["stop_status"] = df.get("stop_status").fillna("UNKNOWN").astype(str)
    agg = df.groupby("stop_status", as_index=False).size().rename(columns={"size":"count"})
    chart = alt.Chart(agg).mark_bar().encode(
        x=alt.X("stop_status:N", title="Statut d'arrêt"),
        y=alt.Y("count:Q", title="Nombre d'arrêts"),
        color=alt.Color("stop_status:N", legend=None),
        tooltip=[alt.Tooltip("stop_status:N", title="Statut"), alt.Tooltip("count:Q", title="Nb")]
    ).properties(title="Répartition des statuts des arrêts")
    return chart.interactive()


def chart_mean_delays(rt_su: pd.DataFrame, trips_df: pd.DataFrame, routes_df: pd.DataFrame):
    """Bar chart des retards moyens : par route si dispo, sinon global."""
    if not _HAS_ALTAIR or rt_su.empty:
        return None
    df = rt_su.copy()
    # delay_best
    df["delay_best"] = df[["arrival_delay","departure_delay"]].bfill(axis=1).iloc[:,0]
    df = df[pd.notna(df["delay_best"])]
    if df.empty:
        return None

    # Si route connue : moyenne par route_label
    has_routes = (not trips_df.empty) and ("route_id" in trips_df.columns)
    if has_routes:
        df = df.merge(trips_df[["trip_id","route_id"]], on="trip_id", how="left")
        if not routes_df.empty and "route_id" in routes_df.columns:
            df = df.merge(routes_df, on="route_id", how="left")
        df["route_label"] = _route_label(df)
        df = df[pd.notna(df["route_label"])]
        if df.empty:
            has_routes = False

    if has_routes:
        agg = df.groupby("route_label", as_index=False)["delay_best"].mean()
        agg = agg.sort_values("delay_best", ascending=False)
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("route_label:N", title="Route", sort="-y"),
            y=alt.Y("delay_best:Q", title="Retard moyen (s)"),
            color=alt.Color("delay_best:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=[alt.Tooltip("route_label:N", title="Route"),
                     alt.Tooltip("delay_best:Q", title="Retard moyen (s)", format=".1f")]
        ).properties(title="Retards moyens par route")
        return chart.interactive()
    else:
        # Global unique barre
        agg = pd.DataFrame({
            "label": ["Global"],
            "delay_best": [float(df["delay_best"].mean())]
        })
        chart = alt.Chart(agg).mark_bar().encode(
            x=alt.X("label:N", title=""),
            y=alt.Y("delay_best:Q", title="Retard moyen (s)"),
            color=alt.Color("delay_best:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=[alt.Tooltip("delay_best:Q", title="Retard moyen (s)", format=".1f")]
        ).properties(title="Retard moyen (global)")
        return chart.interactive()


# ---------------------------------------------
# UI Streamlit (simplifiée + graphiques)
# ---------------------------------------------
def main():
    st.title("🚌 Analyse GTFS-RT TripUpdates vs GTFS (simplifiée + graphiques)")
    st.caption("Synthèse, résumé, anomalies & 3 graphiques (anomalies, statuts d'arrêts, retards moyens). Optimisée pour gros GTFS.")

    with st.sidebar:
        st.header("Fichiers d'entrée")
        gtfs_zip = st.file_uploader("GTFS statique (.zip)", type=["zip"])
        rt_file = st.file_uploader("TripUpdates GTFS-RT (.pb/.bin ou .json)", type=["pb", "bin", "json"])

        st.divider()
        st.subheader("Options")
        focus_only_rt_trips = st.checkbox("Se focaliser sur les voyages du TripUpdate (recommandé)", value=True)
        limit_trips = st.number_input("Limiter à N trips (optionnel)", min_value=0, value=0, step=100)
        show_raw_tables = st.checkbox("Afficher tables brutes (debug)", value=False)

        st.divider()
        run_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if not run_btn:
        st.info("💡 Charge les fichiers puis clique sur **Lancer l'analyse**.")
        return
    if not gtfs_zip or not rt_file:
        st.error("Merci de fournir le **GTFS statique (.zip)** et le **TripUpdate (.pb/.bin/.json)**.")
        return
    # Lecture bytes
    gtfs_bytes = gtfs_zip.getvalue() if hasattr(gtfs_zip, "getvalue") else gtfs_zip.read()
    rt_bytes = rt_file.getvalue() if hasattr(rt_file, "getvalue") else rt_file.read()

    # Parse RT
    with st.status("📥 Lecture du TripUpdate...", expanded=False) as status:
        rt_trips, rt_su, rt_trip_ids = parse_tripupdates_rt(rt_bytes)
        status.update(label=f"TripUpdate: {len(rt_trips)} trips, {len(rt_su)} stop updates.", state="complete")

    # Filtrage ciblé
    if focus_only_rt_trips:
        keep_trip_ids = {tid for tid in rt_trip_ids if isinstance(tid, str) and tid}
        if limit_trips and limit_trips > 0 and keep_trip_ids:
            keep_trip_ids = set(list(keep_trip_ids)[: int(limit_trips)])
    else:
        st.warning("⚠️ Filtrage par trips RT désactivé — utilisation mémoire potentiellement élevée.")
        keep_trip_ids = set()

    # Chargement GTFS (filtré)
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
    sched_vs_rt = compute_schedule_vs_rt(rt_su, stop_times_df, stops_df)

    progress.progress(100, text="Synthèse...")
    summary_df = summarize_trips(rt_trips, rt_su, trips_df, routes_df)
    st.success("Analyse terminée ✅")

    # -----------------------------
    # Synthèse (annulations)
    # -----------------------------
    canceled_total = int((rt_trips["trip_status"] == "CANCELED").sum()) if "trip_status" in rt_trips.columns else 0
    partially_canceled = 0
    if not rt_su.empty and "stop_status" in rt_su.columns:
        trips_with_skipped = set(rt_su.loc[rt_su["stop_status"] == "SKIPPED", "trip_id"].dropna().astype(str))
        if "trip_status" in rt_trips.columns:
            not_canceled_trips = set(rt_trips.loc[rt_trips["trip_status"] != "CANCELED", "trip_id"].dropna().astype(str))
            partially_canceled = len(trips_with_skipped & not_canceled_trips)
        else:
            partially_canceled = len(trips_with_skipped)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Voyages totalement annulés (CANCELED)", f"{canceled_total:,}")
    with k2:
        st.metric("Voyages partiellement annulés (≥1 SKIPPED)", f"{partially_canceled:,}")
    with k3:
        st.metric("Trips RT", f"{rt_trips['trip_id'].nunique() if not rt_trips.empty else 0:,}")
    with k4:
        skipped = int((rt_su["stop_status"] == "SKIPPED").sum()) if (not rt_su.empty and "stop_status" in rt_su.columns) else 0
        st.metric("Arrêts SKIPPED", f"{skipped:,}")

    # -----------------------------
    # Résumé voyages (table)
    # -----------------------------
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

    # -----------------------------
    # Anomalies (table)
    # -----------------------------
    st.subheader("Anomalies détectées")
    anomalies_df = detect_anomalies(rt_su, stops_df, stop_times_df, sched_vs_rt)
    if anomalies_df.empty:
        st.success("Aucune anomalie détectée selon les règles en vigueur.")
    else:
        st.dataframe(anomalies_df, use_container_width=True, height=300)

    # -----------------------------
    # Visualisations demandées
    # -----------------------------
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

        ch3 = chart_mean_delays(rt_su, trips_df, routes_df)
        if ch3 is not None:
            st.altair_chart(ch3, use_container_width=True)
        else:
            st.info("Impossible de calculer un retard moyen (données de delay manquantes).")

    # Debug
    if show_raw_tables:
        st.divider()
        st.subheader("Debug: tables brutes")
        with st.expander("RT: Trips"):
            st.dataframe(rt_trips, use_container_width=True, height=220)
        with st.expander("RT: Stop Updates"):
            st.dataframe(rt_su, use_container_width=True, height=220)
        with st.expander("GTFS: trips (filtré)"):
            st.dataframe(trips_df, use_container_width=True, height=220)
        with st.expander("GTFS: stop_times (filtré)"):
            st.dataframe(stop_times_df.head(1000), use_container_width=True, height=220)
        with st.expander("GTFS: stops (filtré)"):
            st.dataframe(stops_df, use_container_width=True, height=220)
        with st.expander("GTFS: routes (filtré)"):
            st.dataframe(routes_df, use_container_width=True, height=220)

    # Mémoire
    del gtfs_bytes, rt_bytes
    gc.collect()


if __name__ == "__main__":
    main()

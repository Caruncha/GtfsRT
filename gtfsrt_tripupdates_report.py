#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyseur GTFS-rt TripUpdates — Rapport complet (CLI)

Fonctions :
- Charge un fichier TripUpdates (Protocol Buffer).
- (Optionnel) Charge un GTFS statique pour valider la cohérence et comparer au planifié.
- Produit un rapport : volumes, annulations, qualité des timestamps, incohérences.
- Exporte : summary.md, summary.json, summary.html, trips.csv, stop_updates.csv, anomalies.csv, (optionnel) schedule_compare.csv

Installation : pip install pandas gtfs-realtime-bindings
"""

import argparse
import os
import sys
import json
import zipfile
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# Import GTFS-rt
try:
    from google.transit import gtfs_realtime_pb2 as gtfs_rt
except Exception:
    print("💥 Erreur : impossible d'importer gtfs-realtime-bindings.\n"
          "Installe : pip install gtfs-realtime-bindings")
    raise

# --------------------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def unix_to_iso(ts: Optional[int]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None

def plausible_unix_seconds(x: Optional[int]) -> bool:
    """Vérifie que ts est entre 2000-01-01 et 2100-01-01 (~946684800 à 4102444800)."""
    if x is None:
        return False
    try:
        xi = int(x)
    except Exception:
        return False
    return 946_684_800 <= xi <= 4_102_444_800

def detect_and_fix_ms(epoch_values: List[int]) -> bool:
    """
    Détecte si une proportion significative d'horodatages semble être en millisecondes.
    Heuristique: >10% des valeurs entre 1e12 et 1e13 → on considère du ms.
    """
    if not epoch_values:
        return False
    large = [v for v in epoch_values if isinstance(v, (int, float)) and 1e12 < v < 1e13]
    ratio = (len(large) / len(epoch_values)) if epoch_values else 0
    return ratio > 0.10

def time_to_seconds_24hplus(hms: str) -> Optional[int]:
    """Convertit HH:MM:SS (peut dépasser 24h) en secondes depuis minuit."""
    if pd.isna(hms):
        return None
    try:
        parts = str(hms).split(":")
        if len(parts) != 3:
            return None
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return None

def trip_key(trip, entity_id: str) -> Tuple[str, str, str]:
    """
    Clé de voyage pour distinguer les réalisations d'un même trip_id selon date/heure de départ.
    """
    tid = getattr(trip, "trip_id", "") or ""
    sd  = getattr(trip, "start_date", "") or ""
    st  = getattr(trip, "start_time", "") or ""
    if not (tid or sd or st):
        return (f"entity:{entity_id}", "", "")
    return (tid, sd, st)

# --------------------------------------------------------------------------------------
# Chargement GTFS statique (optionnel)
# --------------------------------------------------------------------------------------

def load_static_gtfs(path: Optional[str]) -> Dict[str, pd.DataFrame]:
    """
    Charge un GTFS statique depuis un zip ou un dossier.
    Retourne un dict {stops, trips, stop_times, routes, agency}.
    """
    if not path:
        return {
            "stops": pd.DataFrame(),
            "trips": pd.DataFrame(),
            "stop_times": pd.DataFrame(),
            "routes": pd.DataFrame(),
            "agency": pd.DataFrame(),
        }

    def read_csv_from_zip(zf: zipfile.ZipFile, name: str, dtype=None) -> pd.DataFrame:
        try:
            with zf.open(name) as f:
                return pd.read_csv(f, dtype=dtype)
        except KeyError:
            return pd.DataFrame()

    def read_csv_from_dir(dirp: str, name: str, dtype=None) -> pd.DataFrame:
        fp = os.path.join(dirp, name)
        if os.path.exists(fp):
            return pd.read_csv(fp, dtype=dtype)
        return pd.DataFrame()

    if os.path.isdir(path):
        stops      = read_csv_from_dir(path, "stops.txt",  dtype={"stop_id": str, "stop_name": str})
        trips      = read_csv_from_dir(path, "trips.txt",  dtype={"trip_id": str, "route_id": str})
        stop_times = read_csv_from_dir(path, "stop_times.txt",
                                       dtype={"trip_id": str, "stop_id": str, "stop_sequence": int,
                                              "arrival_time": str, "departure_time": str})
        routes     = read_csv_from_dir(path, "routes.txt", dtype={"route_id": str, "agency_id": str})
        agency     = read_csv_from_dir(path, "agency.txt", dtype={"agency_id": str, "agency_timezone": str})
    else:
        with zipfile.ZipFile(path, "r") as zf:
            stops      = read_csv_from_zip(zf, "stops.txt",  dtype={"stop_id": str, "stop_name": str})
            trips      = read_csv_from_zip(zf, "trips.txt",  dtype={"trip_id": str, "route_id": str})
            stop_times = read_csv_from_zip(zf, "stop_times.txt",
                                           dtype={"trip_id": str, "stop_id": str, "stop_sequence": int,
                                                  "arrival_time": str, "departure_time": str})
            routes     = read_csv_from_zip(zf, "routes.txt", dtype={"route_id": str, "agency_id": str})
            agency     = read_csv_from_zip(zf, "agency.txt", dtype={"agency_id": str, "agency_timezone": str})

    # Normalisations
    for df, cols in [(stops, ["stop_id", "stop_name"]), (trips, ["trip_id", "route_id"]),
                     (routes, ["route_id", "agency_id"]), (agency, ["agency_id", "agency_timezone"])]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str)

    if not stop_times.empty:
        stop_times["trip_id"] = stop_times["trip_id"].astype(str)
        stop_times["stop_id"] = stop_times["stop_id"].astype(str)
        stop_times["arr_sec"] = stop_times["arrival_time"].apply(time_to_seconds_24hplus)
        stop_times["dep_sec"] = stop_times["departure_time"].apply(time_to_seconds_24hplus)

    return {"stops": stops, "trips": trips, "stop_times": stop_times, "routes": routes, "agency": agency}

# --------------------------------------------------------------------------------------
# Comparaison planifié vs RT
# --------------------------------------------------------------------------------------

def _default_agency_tz(static_gtfs: Dict[str, pd.DataFrame]) -> str:
    ag = static_gtfs.get("agency", pd.DataFrame())
    if not ag.empty and "agency_timezone" in ag.columns and pd.notna(ag.iloc[0]["agency_timezone"]):
        return str(ag.iloc[0]["agency_timezone"])
    return "UTC"

def _trip_timezone_map(static_gtfs: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """trip_id -> agency_timezone via routes.agency_id ; repli = tz par défaut."""
    trips = static_gtfs.get("trips", pd.DataFrame())
    routes = static_gtfs.get("routes", pd.DataFrame())
    agency = static_gtfs.get("agency", pd.DataFrame())
    if trips.empty:
        return {}
    tz_default = _default_agency_tz(static_gtfs)
    if not routes.empty and not agency.empty and "agency_id" in routes.columns and "agency_id" in agency.columns:
        routes_tz = routes.merge(agency[["agency_id", "agency_timezone"]], on="agency_id", how="left")
        trip_tz = trips.merge(routes_tz[["route_id", "agency_timezone"]], on="route_id", how="left")
        trip_tz["agency_timezone"] = trip_tz["agency_timezone"].fillna(tz_default)
        return dict(zip(trip_tz["trip_id"], trip_tz["agency_timezone"].astype(str)))
    else:
        return {tid: tz_default for tid in trips["trip_id"].astype(str).tolist()}

def _service_midnight_epoch_utc(start_date: str, tz_str: str) -> Optional[int]:
    """start_date: YYYYMMDD → epoch UTC de minuit local de ce jour de service."""
    try:
        year = int(start_date[0:4]); month = int(start_date[4:6]); day = int(start_date[6:8])
        dt_local = datetime(year, month, day, 0, 0, 0, tzinfo=ZoneInfo(tz_str))
        return int(dt_local.timestamp())
    except Exception:
        return None

def compute_schedule_deltas(stu_df: pd.DataFrame, static_gtfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """
    Calcule les deltas (prédiction - planifié) en secondes pour arrival/departure.
    Retourne (df_compare, stats_dict).
    """
    stop_times = static_gtfs.get("stop_times", pd.DataFrame())
    if stu_df.empty or stop_times.empty:
        return pd.DataFrame(), {}

    # Nécessite trip_id ET start_date
    base = stu_df[(stu_df["trip_id"].notna()) & (stu_df["trip_id"] != "") &
                  (stu_df["start_date"].notna()) & (stu_df["start_date"] != "")]
    if base.empty:
        return pd.DataFrame(), {}

    # Jointure par stop_sequence (prioritaire)
    st_cols_seq = ["trip_id", "stop_sequence", "stop_id", "arr_sec", "dep_sec"]
    st_seq = stop_times[st_cols_seq].dropna(subset=["stop_sequence"]).copy()
    df_seq = base.dropna(subset=["stop_sequence"]).merge(
        st_seq, on=["trip_id", "stop_sequence"], how="left", suffixes=("", "_sched")
    )

    # Repli par stop_id (si pas de stop_sequence)
    st_cols_id = ["trip_id", "stop_id", "stop_sequence", "arr_sec", "dep_sec"]
    st_id = stop_times[st_cols_id].copy()
    df_no_seq = base[base["stop_sequence"].isna()].merge(
        st_id, on=["trip_id", "stop_id"], how="left", suffixes=("", "_sched")
    )

    comp = pd.concat([df_seq, df_no_seq], ignore_index=True, sort=False)
    if comp.empty:
        return pd.DataFrame(), {}

    # Mapping fuseau
    trip_tz_map = _trip_timezone_map(static_gtfs)
    tz_default = _default_agency_tz(static_gtfs)
    comp["tz_used"] = comp["trip_id"].map(trip_tz_map).fillna(tz_default)

    # Minuit UTC du jour de service
    comp["service_midnight_epoch_utc"] = comp.apply(
        lambda r: _service_midnight_epoch_utc(str(r["start_date"]), str(r["tz_used"])), axis=1
    )

    # Époques planifiées (UTC)
    comp["arr_sched_epoch"] = np.where(
        comp["arr_sec"].notna() & comp["service_midnight_epoch_utc"].notna(),
        comp["service_midnight_epoch_utc"].astype("float") + comp["arr_sec"].astype("float"),
        np.nan
    )
    comp["dep_sched_epoch"] = np.where(
        comp["dep_sec"].notna() & comp["service_midnight_epoch_utc"].notna(),
        comp["service_midnight_epoch_utc"].astype("float") + comp["dep_sec"].astype("float"),
        np.nan
    )

    # Deltas
    comp["arr_delta_sec"] = comp.apply(
        lambda r: (r["arrival_time"] - r["arr_sched_epoch"]) if pd.notna(r.get("arrival_time")) and pd.notna(r.get("arr_sched_epoch")) else np.nan,
        axis=1
    )
    comp["dep_delta_sec"] = comp.apply(
        lambda r: (r["departure_time"] - r["dep_sched_epoch"]) if pd.notna(r.get("departure_time")) and pd.notna(r.get("dep_sched_epoch")) else np.nan,
        axis=1
    )

    # Stats utiles
    def _stats(series: pd.Series) -> Dict:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {"count": 0}
        abs_s = s.abs()
        return {
            "count": int(s.size),
            "mean_signed_sec": float(s.mean()),
            "median_signed_sec": float(s.median()),
            "p90_abs_sec": float(abs_s.quantile(0.90)),
            "p95_abs_sec": float(abs_s.quantile(0.95)),
            "within_1_min_pct": float((abs_s <= 60).mean() * 100.0),
            "within_5_min_pct": float((abs_s <= 300).mean() * 100.0),
        }

    stats = {
        "rows_compared": int(len(comp)),
        "arrival": _stats(comp["arr_delta_sec"]),
        "departure": _stats(comp["dep_delta_sec"]),
    }

    cols = [
        "trip_key", "trip_id", "start_date", "start_time", "route_id",
        "stop_id", "stop_sequence",
        "arrival_time", "departure_time",
        "arr_sched_epoch", "dep_sched_epoch",
        "arr_delta_sec", "dep_delta_sec",
        "tz_used"
    ]
    comp = comp[[c for c in cols if c in comp.columns]].copy()

    return comp, stats

# --------------------------------------------------------------------------------------
# Analyse GTFS-realtime TripUpdates
# --------------------------------------------------------------------------------------

def analyze_tripupdates(pb_path: str, static_gtfs: Dict[str, pd.DataFrame]):
    # Lecture du feed
    with open(pb_path, "rb") as f:
        data = f.read()
    feed = gtfs_rt.FeedMessage()
    feed.ParseFromString(data)

    header_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None
    feed_ts_iso = unix_to_iso(header_ts)

    trips_rows = []
    stu_rows = []
    anomalies = []

    # Collecte pour heuristique ms vs s
    time_samples = []
    entity_count = 0

    # Prépare sets rapides
    known_stop_ids = set(static_gtfs["stops"]["stop_id"]) if not static_gtfs["stops"].empty else set()
    known_trip_ids = set(static_gtfs["trips"]["trip_id"]) if not static_gtfs["trips"].empty else set()

    # Itération
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        entity_count += 1
        tu = ent.trip_update
        td = tu.trip

        tkey = trip_key(td, ent.id)

        t_sched_rel = td.schedule_relationship if td.HasField("schedule_relationship") else 0
        t_route_id  = td.route_id if td.HasField("route_id") else ""
        t_trip_id   = td.trip_id if td.HasField("trip_id") else ""
        t_start_date = td.start_date if td.HasField("start_date") else ""
        t_start_time = td.start_time if td.HasField("start_time") else ""

        last_time_for_monotonic = None
        last_seq = -1
        canceled_stops = 0
        nodata_stops = 0
        empty_time_fields = 0
        seq_duplicates = 0
        time_not_monotonic = 0
        arrival_gt_departure = 0

        for stu in tu.stop_time_update:
            stu_sr = stu.schedule_relationship if stu.HasField("schedule_relationship") else 0
            if stu_sr == gtfs_rt.TripUpdate.StopTimeUpdate.SKIPPED:
                canceled_stops += 1
            if stu_sr == gtfs_rt.TripUpdate.StopTimeUpdate.NO_DATA:
                nodata_stops += 1

            stop_id = stu.stop_id if stu.HasField("stop_id") else ""
            stop_seq = stu.stop_sequence if stu.HasField("stop_sequence") else None

            arr_time = stu.arrival.time if (stu.HasField("arrival") and stu.arrival.HasField("time")) else None
            dep_time = stu.departure.time if (stu.HasField("departure") and stu.departure.HasField("time")) else None
            arr_delay = stu.arrival.delay if (stu.HasField("arrival") and stu.arrival.HasField("delay")) else None
            dep_delay = stu.departure.delay if (stu.HasField("departure") and stu.departure.HasField("delay")) else None

            if arr_time is None and dep_time is None and arr_delay is None and dep_delay is None:
                empty_time_fields += 1

            if arr_time is not None:
                time_samples.append(arr_time)
            if dep_time is not None:
                time_samples.append(dep_time)

            if arr_time is not None and dep_time is not None and arr_time > dep_time:
                arrival_gt_departure += 1
                anomalies.append({
                    "type": "arrival_after_departure",
                    "severity": "warning",
                    "trip_key": "|".join(tkey),
                    "stop_id": stop_id,
                    "stop_sequence": stop_seq,
                    "detail": f"arrival_time({arr_time}) > departure_time({dep_time})"
                })

            if stop_seq is not None:
                if stop_seq <= last_seq:
                    seq_duplicates += 1
                    anomalies.append({
                        "type": "stop_sequence_not_increasing",
                        "severity": "warning",
                        "trip_key": "|".join(tkey),
                        "stop_id": stop_id,
                        "stop_sequence": stop_seq,
                        "detail": f"stop_sequence {stop_seq} ≤ précédent {last_seq}"
                    })
                last_seq = stop_seq

            t_for_order = dep_time if dep_time is not None else arr_time
            if t_for_order is not None:
                if last_time_for_monotonic is not None and t_for_order < last_time_for_monotonic:
                    time_not_monotonic += 1
                    anomalies.append({
                        "type": "times_not_monotonic",
                        "severity": "warning",
                        "trip_key": "|".join(tkey),
                        "stop_id": stop_id,
                        "stop_sequence": stop_seq,
                        "detail": f"time {t_for_order} < précédent {last_time_for_monotonic}"
                    })
                last_time_for_monotonic = t_for_order

            if known_stop_ids and stop_id and (stop_id not in known_stop_ids):
                anomalies.append({
                    "type": "unknown_stop_id",
                    "severity": "error",
                    "trip_key": "|".join(tkey),
                    "stop_id": stop_id,
                    "stop_sequence": stop_seq,
                    "detail": "stop_id non présent dans stops.txt"
                })

            stu_rows.append({
                "trip_key": "|".join(tkey),
                "trip_id": t_trip_id,
                "start_date": t_start_date,
                "start_time": t_start_time,
                "route_id": t_route_id,
                "stop_id": stop_id,
                "stop_sequence": stop_seq,
                "stu_schedule_relationship": stu_sr,
                "arrival_time": arr_time,
                "departure_time": dep_time,
                "arrival_delay_sec": arr_delay,
                "departure_delay_sec": dep_delay
            })

        if known_trip_ids and t_trip_id and (t_trip_id not in known_trip_ids):
            anomalies.append({
                "type": "unknown_trip_id",
                "severity": "error",
                "trip_key": "|".join(tkey),
                "stop_id": None,
                "stop_sequence": None,
                "detail": "trip_id non présent dans trips.txt"
            })

        trips_rows.append({
            "trip_key": "|".join(tkey),
            "trip_id": t_trip_id,
            "start_date": t_start_date,
            "start_time": t_start_time,
            "route_id": t_route_id,
            "trip_schedule_relationship": t_sched_rel,
            "canceled_stops": canceled_stops,
            "nodata_stops": nodata_stops,
            "empty_time_fields": empty_time_fields,
            "seq_not_increasing": seq_duplicates,
            "time_not_monotonic": time_not_monotonic,
            "arrival_gt_departure": arrival_gt_departure
        })

    trips_df = pd.DataFrame(trips_rows)
    stu_df = pd.DataFrame(stu_rows)

    # Détection ms → s
    ms_detected = detect_and_fix_ms(time_samples)
    corrected = False
    if ms_detected and not stu_df.empty:
        for col in ["arrival_time", "departure_time"]:
            if col in stu_df.columns:
                stu_df[col] = stu_df[col].apply(lambda v: int(round(v/1000)) if pd.notna(v) else v)
        corrected = True

    # Comparaison au planifié (si GTFS statique dispo)
    schedule_compare_df, schedule_stats = compute_schedule_deltas(stu_df, static_gtfs) if not static_gtfs["stop_times"].empty else (pd.DataFrame(), {})

    # Métadonnées
    meta = {
        "feed_timestamp": int(header_ts) if header_ts is not None else None,
        "feed_timestamp_iso": feed_ts_iso,
        "entities_with_trip_update": entity_count,
        "ms_to_s_corrected": bool(corrected)
    }

    # Agrégations & indicateurs
    summary = {}
    if not trips_df.empty:
        total_trips = trips_df["trip_key"].nunique()
        canceled_trips = (trips_df["trip_schedule_relationship"] == gtfs_rt.TripDescriptor.CANCELED).sum()
        added_trips = (trips_df["trip_schedule_relationship"] == gtfs_rt.TripDescriptor.ADDED).sum()
        unscheduled_trips = (trips_df["trip_schedule_relationship"] == gtfs_rt.TripDescriptor.UNSCHEDULED).sum()

        if not stu_df.empty:
            sk = stu_df[stu_df["stu_schedule_relationship"] == gtfs_rt.TripUpdate.StopTimeUpdate.SKIPPED]
            trips_with_skipped = set(sk["trip_key"]) if not sk.empty else set()
            fully_canceled = set(trips_df.loc[
                trips_df["trip_schedule_relationship"] == gtfs_rt.TripDescriptor.CANCELED, "trip_key"
            ])
            partial_canceled_trips = len(trips_with_skipped - fully_canceled)
            canceled_stops_total = int(len(sk))
            nodata_stops_total = int((stu_df["stu_schedule_relationship"] == gtfs_rt.TripUpdate.StopTimeUpdate.NO_DATA).sum())
        else:
            partial_canceled_trips = 0
            canceled_stops_total = 0
            nodata_stops_total = 0

        summary.update({
            "total_trips": int(total_trips),
            "canceled_trips": int(canceled_trips),
            "added_trips": int(added_trips),
            "unscheduled_trips": int(unscheduled_trips),
            "partial_canceled_trips": int(partial_canceled_trips),
            "canceled_stops_total": int(canceled_stops_total),
            "nodata_stops_total": int(nodata_stops_total),
        })
    else:
        summary.update({
            "total_trips": 0,
            "canceled_trips": 0,
            "added_trips": 0,
            "unscheduled_trips": 0,
            "partial_canceled_trips": 0,
            "canceled_stops_total": 0,
            "nodata_stops_total": 0,
        })

    # Qualité des timestamps
    ts_quality = {}
    if not stu_df.empty:
        both_times = stu_df["arrival_time"].notna() & stu_df["departure_time"].notna()
        any_time = stu_df["arrival_time"].notna() | stu_df["departure_time"].notna()
        neither_time = ~any_time

        ts_quality["stus_with_both_times_pct"] = float(100 * both_times.mean())
        ts_quality["stus_with_any_time_pct"] = float(100 * any_time.mean())
        ts_quality["stus_with_no_times_pct"] = float(100 * neither_time.mean())

        plausible_arr = stu_df["arrival_time"].dropna().apply(plausible_unix_seconds)
        plausible_dep = stu_df["departure_time"].dropna().apply(plausible_unix_seconds)
        total_times = plausible_arr.size + plausible_dep.size
        plausible_count = plausible_arr.sum() + plausible_dep.sum()
        ts_quality["plausible_unix_times_pct"] = float(100 * plausible_count / total_times) if total_times else None

        ts_quality["arrival_after_departure_violations"] = int(trips_df["arrival_gt_departure"].sum()) if "arrival_gt_departure" in trips_df.columns else None
        ts_quality["non_monotonic_times_trips"] = int(trips_df["time_not_monotonic"].sum()) if "time_not_monotonic" in trips_df.columns else None
    else:
        ts_quality = {
            "stus_with_both_times_pct": None,
            "stus_with_any_time_pct": None,
            "stus_with_no_times_pct": None,
            "plausible_unix_times_pct": None,
            "arrival_after_departure_violations": None,
            "non_monotonic_times_trips": None,
        }

    return {
        "meta": meta,
        "summary": summary,
        "ts_quality": ts_quality,
        "trips_df": trips_df,
        "stu_df": stu_df,
        "anomalies": pd.DataFrame(anomalies),
        "schedule_compare_df": schedule_compare_df,
        "schedule_stats": schedule_stats,
        "static_meta": {
            "default_timezone": _default_agency_tz(static_gtfs) if static_gtfs else "UTC"
        }
    }

# --------------------------------------------------------------------------------------
# Rendu du rapport (fichiers)
# --------------------------------------------------------------------------------------

def _build_summary_html(summary_payload: Dict) -> str:
    """Construit un HTML autonome simple (sans graphiques) pour le résumé."""
    meta = summary_payload.get("meta", {})
    summary = summary_payload.get("summary", {})
    tsq = summary_payload.get("timestamp_quality", {})
    schedule = summary_payload.get("schedule_compare", {})

    def li_items(d: Dict) -> str:
        return "".join([f"<li><b>{k}</b> : {v}</li>" for k, v in d.items()])

    html = f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8"/>
<title>Rapport TripUpdates</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1, h2 {{ color: #222; }}
.card {{ border:1px solid #e5e5e5; border-radius:8px; padding:16px; margin-bottom:16px; }}
small {{ color:#666; }}
</style></head>
<body>
<h1>Rapport GTFS-rt : TripUpdates</h1>
<div class="card">
  <h2>Source</h2>
  <ul>
    <li><b>Feed timestamp</b> : {meta.get('feed_timestamp')} <small>({meta.get('feed_timestamp_iso')})</small></li>
    <li><b>Correction ms→s</b> : {meta.get('ms_to_s_corrected')}</li>
    <li><b>Entités TripUpdate</b> : {meta.get('entities_with_trip_update')}</li>
  </ul>
</div>
<div class="card">
  <h2>Synthèse</h2>
  <ul>{li_items(summary)}</ul>
</div>
<div class="card">
  <h2>Qualité des timestamps</h2>
  <ul>{li_items(tsq)}</ul>
</div>
<div class="card">
  <h2>Écart vs horaire planifié</h2>
  <pre style="white-space:pre-wrap;">{json.dumps(schedule, ensure_ascii=False, indent=2)}</pre>
</div>
<footer><small>Généré par gtfsrt_tripupdates_report.py</small></footer>
</body></html>"""
    return html

def write_reports(analysis: Dict, out_dir: str, pb_path: str, gtfs_path: Optional[str]):
    ensure_dir(out_dir)

    trips_df = analysis["trips_df"].copy()
    stu_df = analysis["stu_df"].copy()
    anomalies_df = analysis["anomalies"].copy()
    sched_df = analysis.get("schedule_compare_df", pd.DataFrame())

    # Sauvegardes CSV
    trips_csv = os.path.join(out_dir, "trips.csv")
    stus_csv = os.path.join(out_dir, "stop_updates.csv")
    anomalies_csv = os.path.join(out_dir, "anomalies.csv")
    trips_df.to_csv(trips_csv, index=False)
    stu_df.to_csv(stus_csv, index=False)
    anomalies_df.to_csv(anomalies_csv, index=False)

    # Export comparaison planifié vs prédiction (si dispo)
    sched_cmp_csv = None
    if not sched_df.empty:
        sched_cmp_csv = os.path.join(out_dir, "schedule_compare.csv")
        sched_df.to_csv(sched_cmp_csv, index=False)

    # JSON résumé
    summary_payload = {
        "source": {
            "tripupdates_file": os.path.abspath(pb_path),
            "gtfs_static": os.path.abspath(gtfs_path) if gtfs_path else None
        },
        "meta": analysis["meta"],
        "summary": analysis["summary"],
        "timestamp_quality": analysis["ts_quality"],
        "schedule_compare": analysis.get("schedule_stats", {}),
        "counts": {
            "trip_rows": int(len(trips_df)),
            "stop_time_updates": int(len(stu_df)),
            "anomalies": int(len(anomalies_df)),
            "schedule_compare_rows": int(len(sched_df)) if not sched_df.empty else 0
        }
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # Markdown lisible
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Rapport TripUpdates\n\n")
        f.write(f"- Fichier TripUpdates : `{os.path.abspath(pb_path)}`\n")
        if gtfs_path:
            f.write(f"- GTFS statique : `{os.path.abspath(gtfs_path)}`\n")
        f.write(f"- Feed timestamp : `{analysis['meta'].get('feed_timestamp')}` "
                f"({analysis['meta'].get('feed_timestamp_iso')})\n")
        if analysis["meta"].get("ms_to_s_corrected"):
            f.write(f"- Correction appliquée : temps en **ms → s**\n")
        f.write("\n## Synthèse\n")
        for k, v in analysis["summary"].items():
            f.write(f"- **{k}** : {v}\n")
        f.write("\n## Qualité des timestamps\n")
        for k, v in analysis["ts_quality"].items():
            f.write(f"- **{k}** : {v}\n")
        f.write("\n## Écart vs horaire planifié (sec)\n")
        sc = analysis.get("schedule_stats", {})
        if sc:
            f.write(f"- Lignes comparées : {sc.get('rows_compared', 0)}\n")
            arr = sc.get("arrival", {})
            dep = sc.get("departure", {})
            f.write("\n**Arrival**\n")
            for k in ["count", "mean_signed_sec", "median_signed_sec", "p90_abs_sec", "p95_abs_sec",
                      "within_1_min_pct", "within_5_min_pct"]:
                f.write(f"- {k} : {arr.get(k)}\n")
            f.write("\n**Departure**\n")
            for k in ["count", "mean_signed_sec", "median_signed_sec", "p90_abs_sec", "p95_abs_sec",
                      "within_1_min_pct", "within_5_min_pct"]:
                f.write(f"- {k} : {dep.get(k)}\n")
        else:
            f.write("- Aucune comparaison possible (GTFS statique manquant ou données insuffisantes)\n")
        f.write("\n## Fichiers générés\n")
        f.write(f"- `trips.csv` : par voyage (annulations partielles, incohérences locales)\n")
        f.write(f"- `stop_updates.csv` : chaque STU normalisé\n")
        f.write(f"- `anomalies.csv` : incohérences détectées (type, sévérité, détails)\n")
        if sched_cmp_csv:
            f.write(f"- `schedule_compare.csv` : écart prédiction vs planifié par STU\n")

    # HTML simple
    html_path = os.path.join(out_dir, "summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_build_summary_html(summary_payload))

    return {
        "trips_csv": trips_csv,
        "stus_csv": stus_csv,
        "anomalies_csv": anomalies_csv,
        "schedule_compare_csv": sched_cmp_csv,
        "summary_json": os.path.join(out_dir, "summary.json"),
        "summary_md": md_path,
        "summary_html": html_path
    }

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Génère un rapport complet à partir d'un fichier GTFS-rt TripUpdates (.pb)."
    )
    parser.add_argument("--tripupdates", required=True, help="Fichier TripUpdates .pb (Protocol Buffer)")
    parser.add_argument("--gtfs", required=False, help="GTFS statique (zip ou dossier) pour validations et comparaison")
    parser.add_argument("--out", required=True, help="Dossier de sortie du rapport")
    args = parser.parse_args()

    if not os.path.exists(args.tripupdates):
        print(f"💥 Introuvable : {args.tripupdates}")
        sys.exit(1)
    if args.gtfs and not os.path.exists(args.gtfs):
        print(f"💥 GTFS statique introuvable : {args.gtfs}")
        sys.exit(1)

    static_gtfs = load_static_gtfs(args.gtfs)
    analysis = analyze_tripupdates(args.tripupdates, static_gtfs)
    outputs = write_reports(analysis, args.out, args.tripupdates, args.gtfs)

    print("✅ Rapport généré :")
    for k, v in outputs.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()

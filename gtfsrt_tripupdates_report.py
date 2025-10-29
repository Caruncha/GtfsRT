#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimisations de performance / mémoire :
- Ajout d'un **pré-parsing** du TripUpdates pour extraire `trip_id`/`stop_id`.
- Lecture **par morceaux (chunks)** des fichiers GTFS dans le zip.
- Filtrage **sur les voyages/arrêts présents dans le TripUpdates** (fast mode).
- Tables minimales (usecols + dtypes string), conversions numériques ciblées.
"""
import argparse
import io
import os
import sys
import json
import zipfile
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List, Union, Iterable, Set

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

try:
    from google.transit import gtfs_realtime_pb2 as gtfs_rt
except Exception:
    print("💥 Erreur : impossible d'importer gtfs-realtime-bindings.\nInstalle : pip install gtfs-realtime-bindings")
    raise

# ---------------------------------
# Utilitaires généraux
# ---------------------------------
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
    if x is None:
        return False
    try:
        xi = int(x)
    except Exception:
        return False
    return 946_684_800 <= xi <= 4_102_444_800  # 2000-2100

def detect_ms(epoch_values: List[int]) -> bool:
    if not epoch_values:
        return False
    large = [v for v in epoch_values if isinstance(v, (int, float)) and 1e12 < v < 1e13]
    return (len(large) / len(epoch_values)) > 0.10

def time_to_seconds_24hplus(hms: str) -> Optional[int]:
    if pd.isna(hms):
        return None
    try:
        h, m, s = [int(x) for x in str(hms).split(":")]
        return h * 3600 + m * 60 + s
    except Exception:
        return None

# ---------------------------------
# Pré-parsing TripUpdates → ids
# ---------------------------------
def extract_ids_from_tripupdates(pb_path: str) -> Tuple[Set[str], Set[str]]:
    trip_ids: Set[str] = set()
    stop_ids: Set[str] = set()
    feed = gtfs_rt.FeedMessage()
    with open(pb_path, 'rb') as f:
        feed.ParseFromString(f.read())
    for ent in feed.entity:
        if not ent.HasField('trip_update'):
            continue
        tu = ent.trip_update
        if tu.trip and tu.trip.trip_id:
            trip_ids.add(str(tu.trip.trip_id))
        for stu in tu.stop_time_update:
            if stu.stop_id:
                stop_ids.add(str(stu.stop_id))
    return trip_ids, stop_ids

# ---------------------------------
# Lecteurs GTFS par chunks (zip ou dossier)
# ---------------------------------
def _iter_csv_from_zip(zip_path: str, member: str, usecols=None, dtype=None, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            return
        with zf.open(member) as fh:
            for chunk in pd.read_csv(
                fh,
                usecols=usecols,
                dtype=dtype,
                chunksize=chunksize,
                low_memory=True,
                encoding='utf-8',
                on_bad_lines='skip'
            ):
                yield chunk

def _iter_csv_from_file(path: str, usecols=None, dtype=None, chunksize: int = 200_000) -> Iterable[pd.DataFrame]:
    if not os.path.exists(path):
        return
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=True,
        encoding='utf-8',
        on_bad_lines='skip'
    ):
        yield chunk

# ---------------------------------
# Chargement GTFS (filtré)
# ---------------------------------
def load_static_gtfs(path: Optional[str], trip_ids: Optional[Set[str]] = None, stop_ids: Optional[Set[str]] = None, fast: bool = True, chunksize: int = 200_000) -> Dict[str, pd.DataFrame]:
    """
    Charge un GTFS statique (zip ou dossier). Si `fast` et des ids sont fournis,
    ne charge que les lignes nécessaires:
      - trips.txt filtré sur trip_ids
      - stop_times.txt filtré sur les trip_ids
      - stops.txt filtré sur (stop_ids du RT) ∪ (stop_ids des stop_times filtrés)
      - routes.txt filtré sur les route_id présents dans trips filtrés
      - agency.txt (entière, petite)
    """
    if not path:
        return {"stops": pd.DataFrame(), "trips": pd.DataFrame(), "stop_times": pd.DataFrame(), "routes": pd.DataFrame(), "agency": pd.DataFrame()}

    def iter_csv(name: str, usecols=None, dtype=None):
        if os.path.isdir(path):
            return _iter_csv_from_file(os.path.join(path, name), usecols=usecols, dtype=dtype, chunksize=chunksize)
        else:
            return _iter_csv_from_zip(path, name, usecols=usecols, dtype=dtype, chunksize=chunksize)

    # Agency (petit): on tente lecture directe complète
    agency = pd.DataFrame()
    try:
        if os.path.isdir(path):
            agency_path = os.path.join(path, 'agency.txt')
            if os.path.exists(agency_path):
                agency = pd.read_csv(agency_path, dtype={'agency_id': 'string', 'agency_timezone': 'string'})
        else:
            with zipfile.ZipFile(path) as zf:
                if 'agency.txt' in zf.namelist():
                    with zf.open('agency.txt') as fh:
                        agency = pd.read_csv(fh, dtype={'agency_id': 'string', 'agency_timezone': 'string'})
    except Exception:
        agency = pd.DataFrame()

    if not fast or not trip_ids:
        # Mode non filtré (ou pas d'ids): on lit tout (toujours par chunks mais concat complet)
        trips = pd.concat(list(iter_csv('trips.txt', usecols=['trip_id','route_id'], dtype='string') or []), ignore_index=True) if any(True for _ in iter_csv('trips.txt', usecols=['trip_id','route_id'], dtype='string')) else pd.DataFrame(columns=['trip_id','route_id'])
        stop_times = pd.concat(list(iter_csv('stop_times.txt', usecols=['trip_id','stop_id','stop_sequence','arrival_time','departure_time'], dtype='string') or []), ignore_index=True) if any(True for _ in iter_csv('stop_times.txt', usecols=['trip_id','stop_id','stop_sequence','arrival_time','departure_time'], dtype='string')) else pd.DataFrame(columns=['trip_id','stop_id','stop_sequence','arrival_time','departure_time'])
        stops = pd.concat(list(iter_csv('stops.txt', usecols=['stop_id','stop_name','stop_lat','stop_lon'], dtype='string') or []), ignore_index=True) if any(True for _ in iter_csv('stops.txt', usecols=['stop_id','stop_name','stop_lat','stop_lon'], dtype='string')) else pd.DataFrame(columns=['stop_id','stop_name','stop_lat','stop_lon'])
        routes = pd.concat(list(iter_csv('routes.txt', usecols=['route_id','agency_id','route_short_name','route_long_name','route_color'], dtype='string') or []), ignore_index=True) if any(True for _ in iter_csv('routes.txt', usecols=['route_id','agency_id','route_short_name','route_long_name','route_color'], dtype='string')) else pd.DataFrame(columns=['route_id','agency_id','route_short_name','route_long_name','route_color'])
    else:
        # Filtré
        trip_ids = set(str(x) for x in trip_ids)
        # trips.txt
        trips_parts = []
        for df in iter_csv('trips.txt', usecols=['trip_id','route_id'], dtype='string') or []:
            df = df[df['trip_id'].isin(trip_ids)]
            if not df.empty:
                trips_parts.append(df)
        trips = pd.concat(trips_parts, ignore_index=True) if trips_parts else pd.DataFrame(columns=['trip_id','route_id'])
        # stop_times.txt
        st_parts = []
        if not trips.empty:
            keep_trip_ids = set(trips['trip_id'].dropna().unique())
            for df in iter_csv('stop_times.txt', usecols=['trip_id','stop_id','stop_sequence','arrival_time','departure_time'], dtype='string') or []:
                df = df[df['trip_id'].isin(keep_trip_ids)]
                if not df.empty:
                    st_parts.append(df)
        stop_times = pd.concat(st_parts, ignore_index=True) if st_parts else pd.DataFrame(columns=['trip_id','stop_id','stop_sequence','arrival_time','departure_time'])
        # stops.txt
        static_stop_ids = set(stop_times['stop_id'].dropna().unique())
        all_stop_ids = set(stop_ids or []) | static_stop_ids
        stops_parts = []
        if all_stop_ids:
            for df in iter_csv('stops.txt', usecols=['stop_id','stop_name','stop_lat','stop_lon'], dtype='string') or []:
                df = df[df['stop_id'].isin(all_stop_ids)]
                if not df.empty:
                    stops_parts.append(df)
        stops = pd.concat(stops_parts, ignore_index=True) if stops_parts else pd.DataFrame(columns=['stop_id','stop_name','stop_lat','stop_lon'])
        # routes.txt
        route_ids = set(trips['route_id'].dropna().unique()) if not trips.empty else set()
        routes_parts = []
        if route_ids:
            for df in iter_csv('routes.txt', usecols=['route_id','agency_id','route_short_name','route_long_name','route_color'], dtype='string') or []:
                df = df[df['route_id'].isin(route_ids)]
                if not df.empty:
                    routes_parts.append(df)
        routes = pd.concat(routes_parts, ignore_index=True) if routes_parts else pd.DataFrame(columns=['route_id','agency_id','route_short_name','route_long_name','route_color'])

    # Normalisations
    for df, cols in [(trips, ['trip_id','route_id']), (stops, ['stop_id','stop_name']), (routes, ['route_id','agency_id'])]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype('string')
    if not agency.empty and 'agency_timezone' in agency.columns:
        agency['agency_timezone'] = agency['agency_timezone'].astype('string')

    # Precompute sec columns for schedule compare
    if not stop_times.empty:
        stop_times['arr_sec'] = stop_times['arrival_time'].apply(time_to_seconds_24hplus)
        stop_times['dep_sec'] = stop_times['departure_time'].apply(time_to_seconds_24hplus)
        # safe numeric
        stop_times['stop_sequence'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce').astype('Int64')

    return {"stops": stops, "trips": trips, "stop_times": stop_times, "routes": routes, "agency": agency}

# ---------------------------------
# Analyse TripUpdates (reprend la logique existante)
# ---------------------------------
def trip_key(trip, entity_id: str) -> Tuple[str, str, str]:
    tid = getattr(trip, "trip_id", "") or ""
    sd = getattr(trip, "start_date", "") or ""
    st = getattr(trip, "start_time", "") or ""
    if not (tid or sd or st):
        return (f"entity:{entity_id}", "", "")
    return (tid, sd, st)

def analyze_tripupdates(pb_path: str, static_gtfs: Dict[str, pd.DataFrame]):
    with open(pb_path, 'rb') as f:
        data = f.read()
    feed = gtfs_rt.FeedMessage()
    feed.ParseFromString(data)

    header_ts = feed.header.timestamp if feed.header.HasField('timestamp') else None
    trips_rows: List[Dict] = []
    stu_rows: List[Dict] = []
    anomalies: List[Dict] = []

    time_samples: List[int] = []
    entity_count = 0

    known_stop_ids = set(static_gtfs.get("stops", pd.DataFrame()).get("stop_id", pd.Series([], dtype='string')))
    known_trip_ids = set(static_gtfs.get("trips", pd.DataFrame()).get("trip_id", pd.Series([], dtype='string')))

    for ent in feed.entity:
        if not ent.HasField('trip_update'):
            continue
        entity_count += 1
        tu = ent.trip_update
        td = tu.trip
        tkey = trip_key(td, ent.id)
        t_sched_rel = td.schedule_relationship if td.HasField('schedule_relationship') else 0
        t_route_id = td.route_id if td.HasField('route_id') else ""
        t_trip_id = td.trip_id if td.HasField('trip_id') else ""
        t_start_date = td.start_date if td.HasField('start_date') else ""
        t_start_time = td.start_time if td.HasField('start_time') else ""

        last_time_for_monotonic = None
        last_seq = -1
        canceled_stops = 0
        nodata_stops = 0
        empty_time_fields = 0
        seq_duplicates = 0
        time_not_monotonic = 0
        arrival_gt_departure = 0

        for stu in tu.stop_time_update:
            stu_sr = stu.schedule_relationship if stu.HasField('schedule_relationship') else 0
            if stu_sr == gtfs_rt.TripUpdate.StopTimeUpdate.SKIPPED:
                canceled_stops += 1
            if stu_sr == gtfs_rt.TripUpdate.StopTimeUpdate.NO_DATA:
                nodata_stops += 1

            stop_id = stu.stop_id if stu.HasField('stop_id') else ""
            stop_seq = stu.stop_sequence if stu.HasField('stop_sequence') else None
            arr_time = stu.arrival.time if (stu.HasField('arrival') and stu.arrival.HasField('time')) else None
            dep_time = stu.departure.time if (stu.HasField('departure') and stu.departure.HasField('time')) else None
            arr_delay = stu.arrival.delay if (stu.HasField('arrival') and stu.arrival.HasField('delay')) else None
            dep_delay = stu.departure.delay if (stu.HasField('departure') and stu.departure.HasField('delay')) else None

            if arr_time is None and dep_time is None and arr_delay is None and dep_delay is None:
                empty_time_fields += 1

            if arr_time is not None:
                time_samples.append(arr_time)
            if dep_time is not None:
                time_samples.append(dep_time)
            if (arr_time is not None) and (dep_time is not None) and (arr_time > dep_time):
                arrival_gt_departure += 1
                anomalies.append({
                    'type': 'arrival_after_departure', 'severity': 'warning',
                    'trip_key': "\n".join(tkey), 'stop_id': stop_id, 'stop_sequence': stop_seq,
                    'detail': f'arrival_time({arr_time}) > departure_time({dep_time})'
                })

            if stop_seq is not None:
                if stop_seq <= last_seq:
                    seq_duplicates += 1
                    anomalies.append({
                        'type': 'stop_sequence_not_increasing', 'severity': 'warning',
                        'trip_key': "\n".join(tkey), 'stop_id': stop_id, 'stop_sequence': stop_seq,
                        'detail': f'stop_sequence {stop_seq} ≤ prev {last_seq}'
                    })
                last_seq = stop_seq

            t_for_order = dep_time if dep_time is not None else arr_time
            if t_for_order is not None:
                if last_time_for_monotonic is not None and t_for_order < last_time_for_monotonic:
                    time_not_monotonic += 1
                    anomalies.append({
                        'type': 'times_not_monotonic', 'severity': 'warning',
                        'trip_key': "\n".join(tkey), 'stop_id': stop_id, 'stop_sequence': stop_seq,
                        'detail': f'time {t_for_order} < prev {last_time_for_monotonic}'
                    })
                last_time_for_monotonic = t_for_order

            if known_stop_ids and stop_id and (stop_id not in known_stop_ids):
                anomalies.append({'type': 'unknown_stop_id', 'severity': 'error', 'trip_key': "\n".join(tkey), 'stop_id': stop_id, 'stop_sequence': stop_seq, 'detail': 'stop_id absent de stops.txt'})

            stu_rows.append({'trip_key': "\n".join(tkey), 'trip_id': t_trip_id, 'start_date': t_start_date, 'start_time': t_start_time, 'route_id': t_route_id, 'stop_id': stop_id, 'stop_sequence': stop_seq, 'stu_schedule_relationship': stu_sr, 'arrival_time': arr_time, 'departure_time': dep_time, 'arrival_delay_sec': arr_delay, 'departure_delay_sec': dep_delay})

        if known_trip_ids and t_trip_id and (t_trip_id not in known_trip_ids):
            anomalies.append({'type': 'unknown_trip_id', 'severity': 'error', 'trip_key': "\n".join(tkey), 'stop_id': None, 'stop_sequence': None, 'detail': 'trip_id absent de trips.txt'})

        trips_rows.append({'trip_key': "\n".join(tkey), 'trip_id': t_trip_id, 'start_date': t_start_date, 'start_time': t_start_time, 'route_id': t_route_id, 'trip_schedule_relationship': t_sched_rel, 'canceled_stops': canceled_stops, 'nodata_stops': nodata_stops, 'empty_time_fields': empty_time_fields, 'seq_not_increasing': seq_duplicates, 'time_not_monotonic': time_not_monotonic, 'arrival_gt_departure': arrival_gt_departure})

    trips_df = pd.DataFrame(trips_rows)
    stu_df = pd.DataFrame(stu_rows)

    ms_detected = detect_ms(time_samples)
    corrected = False
    if ms_detected and not stu_df.empty:
        for col in ['arrival_time','departure_time']:
            if col in stu_df.columns:
                stu_df[col] = stu_df[col].apply(lambda v: int(round(v/1000)) if pd.notna(v) else v)
        corrected = True

    # Schedule compare
    schedule_compare_df, schedule_stats = compute_schedule_deltas(stu_df, static_gtfs) if not static_gtfs.get('stop_times', pd.DataFrame()).empty else (pd.DataFrame(), {})

    # Meta & summary
    meta = {'feed_timestamp': int(header_ts) if header_ts is not None else None, 'feed_timestamp_iso': unix_to_iso(header_ts), 'entities_with_trip_update': entity_count, 'ms_to_s_corrected': bool(corrected)}

    if not trips_df.empty:
        total_trips = trips_df['trip_key'].nunique()
        canceled_trips = (trips_df['trip_schedule_relationship'] == gtfs_rt.TripDescriptor.CANCELED).sum()
        added_trips = (trips_df['trip_schedule_relationship'] == gtfs_rt.TripDescriptor.ADDED).sum()
        unscheduled_trips = (trips_df['trip_schedule_relationship'] == gtfs_rt.TripDescriptor.UNSCHEDULED).sum()
        if not stu_df.empty:
            sk = stu_df[stu_df['stu_schedule_relationship'] == gtfs_rt.TripUpdate.StopTimeUpdate.SKIPPED]
            trips_with_skipped = set(sk['trip_key']) if not sk.empty else set()
            fully_canceled = set(trips_df.loc[trips_df['trip_schedule_relationship'] == gtfs_rt.TripDescriptor.CANCELED, 'trip_key'])
            partial_canceled_trips = len(trips_with_skipped - fully_canceled)
            canceled_stops_total = int(len(sk))
            nodata_stops_total = int((stu_df['stu_schedule_relationship'] == gtfs_rt.TripUpdate.StopTimeUpdate.NO_DATA).sum())
        else:
            partial_canceled_trips = 0
            canceled_stops_total = 0
            nodata_stops_total = 0
        summary = {
            'total_trips': int(total_trips),
            'canceled_trips': int(canceled_trips),
            'added_trips': int(added_trips),
            'unscheduled_trips': int(unscheduled_trips),
            'partial_canceled_trips': int(partial_canceled_trips),
            'canceled_stops_total': int(canceled_stops_total),
            'nodata_stops_total': int(nodata_stops_total),
        }
    else:
        summary = {k: 0 for k in ['total_trips','canceled_trips','added_trips','unscheduled_trips','partial_canceled_trips','canceled_stops_total','nodata_stops_total']}

    if not stu_df.empty:
        both_times = stu_df['arrival_time'].notna() & stu_df['departure_time'].notna()
        any_time = stu_df['arrival_time'].notna() | stu_df['departure_time'].notna()
        neither_time = ~any_time
        plausible_arr = stu_df['arrival_time'].dropna().apply(plausible_unix_seconds)
        plausible_dep = stu_df['departure_time'].dropna().apply(plausible_unix_seconds)
        total_times = plausible_arr.size + plausible_dep.size
        plausible_count = plausible_arr.sum() + plausible_dep.sum()
        ts_quality = {
            'stus_with_both_times_pct': float(100 * both_times.mean()),
            'stus_with_any_time_pct': float(100 * any_time.mean()),
            'stus_with_no_times_pct': float(100 * neither_time.mean()),
            'plausible_unix_times_pct': float(100 * plausible_count / total_times) if total_times else None,
            'arrival_after_departure_violations': int(trips_df['arrival_gt_departure'].sum()) if 'arrival_gt_departure' in trips_df.columns else None,
            'non_monotonic_times_trips': int(trips_df['time_not_monotonic'].sum()) if 'time_not_monotonic' in trips_df.columns else None,
        }
    else:
        ts_quality = {k: None for k in ['stus_with_both_times_pct','stus_with_any_time_pct','stus_with_no_times_pct','plausible_unix_times_pct','arrival_after_departure_violations','non_monotonic_times_trips']}

    return {
        'meta': meta,
        'summary': summary,
        'ts_quality': ts_quality,
        'trips_df': trips_df,
        'stu_df': stu_df,
        'anomalies': pd.DataFrame(anomalies),
        'schedule_compare_df': schedule_compare_df,
        'schedule_stats': schedule_stats,
        'static_meta': {
            'default_timezone': _default_agency_tz({'agency': static_gtfs.get('agency', pd.DataFrame())}) if static_gtfs else 'UTC'
        }
    }

# ---------------------------------
# Schedule compare (inchangée sauf dépendances)
# ---------------------------------
def _default_agency_tz(static_gtfs: Dict[str, pd.DataFrame]) -> str:
    ag = static_gtfs.get('agency', pd.DataFrame())
    if not ag.empty and 'agency_timezone' in ag.columns and pd.notna(ag.iloc[0]['agency_timezone']):
        return str(ag.iloc[0]['agency_timezone'])
    return 'UTC'

def _trip_timezone_map(static_gtfs: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    trips = static_gtfs.get('trips', pd.DataFrame())
    routes = static_gtfs.get('routes', pd.DataFrame())
    agency = static_gtfs.get('agency', pd.DataFrame())
    if trips.empty:
        return {}
    tz_default = _default_agency_tz(static_gtfs)
    if not routes.empty and not agency.empty and 'agency_id' in routes.columns and 'agency_id' in agency.columns:
        routes_tz = routes.merge(agency[['agency_id','agency_timezone']], on='agency_id', how='left')
        trip_tz = trips.merge(routes_tz[['route_id','agency_timezone']], on='route_id', how='left')
        trip_tz['agency_timezone'] = trip_tz['agency_timezone'].fillna(tz_default)
        return dict(zip(trip_tz['trip_id'], trip_tz['agency_timezone'].astype('string')))
    else:
        return {tid: tz_default for tid in trips['trip_id'].astype('string').tolist()}

def _service_midnight_epoch_utc(start_date: str, tz_str: str) -> Optional[int]:
    try:
        year = int(start_date[0:4]); month = int(start_date[4:6]); day = int(start_date[6:8])
        dt_local = datetime(year, month, day, 0, 0, 0, tzinfo=ZoneInfo(tz_str))
        return int(dt_local.timestamp())
    except Exception:
        return None

def compute_schedule_deltas(stu_df: pd.DataFrame, static_gtfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    stop_times = static_gtfs.get('stop_times', pd.DataFrame())
    if stu_df.empty or stop_times.empty:
        return pd.DataFrame(), {}
    base = stu_df[(stu_df['trip_id'].notna()) & (stu_df['trip_id'] != '') & (stu_df['start_date'].notna()) & (stu_df['start_date'] != '')]
    if base.empty:
        return pd.DataFrame(), {}
    st_cols_seq = ['trip_id','stop_sequence','stop_id','arr_sec','dep_sec']
    st_seq = stop_times[st_cols_seq].dropna(subset=['stop_sequence']).copy()
    df_seq = base.dropna(subset=['stop_sequence']).merge(st_seq, on=['trip_id','stop_sequence'], how='left', suffixes=("","_sched"))
    st_cols_id = ['trip_id','stop_id','stop_sequence','arr_sec','dep_sec']
    st_id = stop_times[st_cols_id].copy()
    df_no_seq = base[base['stop_sequence'].isna()].merge(st_id, on=['trip_id','stop_id'], how='left', suffixes=("","_sched"))
    comp = pd.concat([df_seq, df_no_seq], ignore_index=True, sort=False)
    if comp.empty:
        return pd.DataFrame(), {}
    trip_tz_map = _trip_timezone_map(static_gtfs)
    tz_default = _default_agency_tz(static_gtfs)
    comp['tz_used'] = comp['trip_id'].map(trip_tz_map).fillna(tz_default)
    comp['service_midnight_epoch_utc'] = comp.apply(lambda r: _service_midnight_epoch_utc(str(r['start_date']), str(r['tz_used'])), axis=1)
    comp['arr_sched_epoch'] = np.where(comp['arr_sec'].notna() & comp['service_midnight_epoch_utc'].notna(), comp['service_midnight_epoch_utc'].astype('float') + comp['arr_sec'].astype('float'), np.nan)
    comp['dep_sched_epoch'] = np.where(comp['dep_sec'].notna() & comp['service_midnight_epoch_utc'].notna(), comp['service_midnight_epoch_utc'].astype('float') + comp['dep_sec'].astype('float'), np.nan)
    comp['arrival_time_clean'] = comp['arrival_time'].where(comp['arrival_time'].apply(lambda v: isinstance(v,(int,float)) and plausible_unix_seconds(v)), np.nan)
    comp['departure_time_clean'] = comp['departure_time'].where(comp['departure_time'].apply(lambda v: isinstance(v,(int,float)) and plausible_unix_seconds(v)), np.nan)
    comp['arr_delta_sec'] = comp.apply(lambda r: (r['arrival_time_clean'] - r['arr_sched_epoch']) if pd.notna(r.get('arrival_time_clean')) and pd.notna(r.get('arr_sched_epoch')) else np.nan, axis=1)
    comp['dep_delta_sec'] = comp.apply(lambda r: (r['departure_time_clean'] - r['dep_sched_epoch']) if pd.notna(r.get('departure_time_clean')) and pd.notna(r.get('dep_sched_epoch')) else np.nan, axis=1)

    def _stats(series: pd.Series) -> Dict:
        s = pd.to_numeric(series, errors='coerce').dropna()
        if s.empty:
            return {'count': 0}
        abs_s = s.abs()
        return {'count': int(s.size), 'mean_signed_sec': float(s.mean()), 'median_signed_sec': float(s.median()), 'p90_abs_sec': float(abs_s.quantile(0.90)), 'p95_abs_sec': float(abs_s.quantile(0.95)), 'within_1_min_pct': float((abs_s <= 60).mean()*100.0), 'within_5_min_pct': float((abs_s <= 300).mean()*100.0)}

    stats = {'rows_compared': int(len(comp)), 'arrival': _stats(comp['arr_delta_sec']), 'departure': _stats(comp['dep_delta_sec'])}
    cols = ['trip_key','trip_id','start_date','start_time','route_id','stop_id','stop_sequence','arrival_time','departure_time','arr_sched_epoch','dep_sched_epoch','arr_delta_sec','dep_delta_sec','tz_used']
    comp = comp[[c for c in cols if c in comp.columns]].copy()
    return comp, stats

# ---------------------------------
# Rendu fichiers (identique)
# ---------------------------------
def write_reports(analysis: Dict, out_dir: str, pb_path: str, gtfs_path: Optional[str], validation: Optional[Dict]=None):
    ensure_dir(out_dir)
    trips_df = analysis['trips_df'].copy()
    stu_df = analysis['stu_df'].copy()
    anomalies_df = analysis['anomalies'].copy()
    sched_df = analysis.get('schedule_compare_df', pd.DataFrame())

    trips_csv = os.path.join(out_dir, 'trips.csv'); trips_df.to_csv(trips_csv, index=False)
    stus_csv = os.path.join(out_dir, 'stop_updates.csv'); stu_df.to_csv(stus_csv, index=False)
    anomalies_csv = os.path.join(out_dir, 'anomalies.csv'); anomalies_df.to_csv(anomalies_csv, index=False)
    sched_cmp_csv = None
    if not sched_df.empty:
        sched_cmp_csv = os.path.join(out_dir, 'schedule_compare.csv'); sched_df.to_csv(sched_cmp_csv, index=False)

    summary_payload = {
        'source': {'tripupdates_file': os.path.abspath(pb_path), 'gtfs_static': os.path.abspath(gtfs_path) if gtfs_path else None},
        'meta': analysis['meta'], 'summary': analysis['summary'], 'timestamp_quality': analysis['ts_quality'], 'schedule_compare': analysis.get('schedule_stats', {}), 'counts': {'trip_rows': int(len(trips_df)), 'stop_time_updates': int(len(stu_df)), 'anomalies': int(len(anomalies_df)), 'schedule_compare_rows': int(len(sched_df)) if not sched_df.empty else 0}
    }
    if validation:
        summary_payload['cancellations'] = {'summary': validation.get('cancellations_summary', {}), 'window': validation.get('window', {}), 'files': {'results_csv': validation.get('cancellations_validation_csv')}}
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(out_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Rapport TripUpdates\n\n")
        f.write(f"- Fichier TripUpdates : `{os.path.abspath(pb_path)}`\n")
        if gtfs_path:
            f.write(f"- GTFS statique : `{os.path.abspath(gtfs_path)}`\n")
        f.write(f"- Feed timestamp : `{analysis['meta'].get('feed_timestamp')}` ({analysis['meta'].get('feed_timestamp_iso')})\n")
        if analysis['meta'].get('ms_to_s_corrected'):
            f.write(f"- Correction appliquée : temps en **ms → s**\n")
        f.write("\n## Synthèse\n")
        for k, v in analysis['summary'].items():
            f.write(f"- **{k}** : {v}\n")
        f.write("\n## Qualité des timestamps\n")
        for k, v in analysis['ts_quality'].items():
            f.write(f"- **{k}** : {v}\n")
        f.write("\n## Écart vs horaire planifié (sec)\n")
        sc = analysis.get('schedule_stats', {})
        if sc:
            f.write(f"- Lignes comparées : {sc.get('rows_compared', 0)}\n")
            arr = sc.get('arrival', {}); dep = sc.get('departure', {})
            f.write("\n**Arrival**\n")
            for k in ["count","mean_signed_sec","median_signed_sec","p90_abs_sec","p95_abs_sec","within_1_min_pct","within_5_min_pct"]:
                f.write(f"- {k} : {arr.get(k)}\n")
            f.write("\n**Departure**\n")
            for k in ["count","mean_signed_sec","median_signed_sec","p90_abs_sec","p95_abs_sec","within_1_min_pct","within_5_min_pct"]:
                f.write(f"- {k} : {dep.get(k)}\n")
        else:
            f.write("- Aucune comparaison possible (GTFS statique manquant ou données insuffisantes)\n")
        f.write("\n## Fichiers générés\n")
        f.write("- `trips.csv`\n- `stop_updates.csv`\n- `anomalies.csv`\n")
        if sched_cmp_csv:
            f.write("- `schedule_compare.csv`\n")

    return {'trips_csv': trips_csv, 'stus_csv': stus_csv, 'anomalies_csv': anomalies_csv, 'schedule_compare_csv': sched_cmp_csv, 'summary_json': os.path.join(out_dir, 'summary.json'), 'summary_md': md_path}

# ---------------------------------
# CLI (inchangé)
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génère un rapport à partir d'un GTFS-rt TripUpdates (.pb).")
    parser.add_argument('--tripupdates', required=True, help='Fichier TripUpdates (.pb)')
    parser.add_argument('--gtfs', required=False, help='GTFS statique (.zip ou dossier)')
    parser.add_argument('--out', required=True, help='Dossier de sortie')
    parser.add_argument('--fast', action='store_true', help='Charger uniquement les trips du TripUpdates (recommandé)')
    parser.add_argument('--chunksize', type=int, default=200_000, help='Taille de chunk pour lecture GTFS (défaut: 200k)')
    args = parser.parse_args()

    if not os.path.exists(args.tripupdates):
        print(f"💥 Introuvable : {args.tripupdates}", file=sys.stderr); sys.exit(1)
    if args.gtfs and not os.path.exists(args.gtfs):
        print(f"💥 GTFS statique introuvable : {args.gtfs}", file=sys.stderr); sys.exit(1)

    # Pré-parsing pour filtrage
    trip_ids, stop_ids = extract_ids_from_tripupdates(args.tripupdates)
    static_gtfs = load_static_gtfs(args.gtfs, trip_ids=trip_ids if args.fast else None, stop_ids=stop_ids if args.fast else None, fast=args.fast, chunksize=args.chunksize) if args.gtfs else {"stops": pd.DataFrame(), "trips": pd.DataFrame(), "stop_times": pd.DataFrame(), "routes": pd.DataFrame(), "agency": pd.DataFrame()}

    analysis = analyze_tripupdates(args.tripupdates, static_gtfs)

    ensure_dir(args.out)
    outputs = write_reports(analysis, args.out, args.tripupdates, args.gtfs)
    print("✅ Rapport généré:")
    for k,v in outputs.items():
        print(f" - {k}: {v}")

if __name__ == '__main__':
    main()

# app.py
# ---------------------------------------------
# Streamlit - Analyse des GTFS-RT TripUpdates vs GTFS statique (optimisé gros GTFS)
# Focalisation sur les voyages présents dans le TripUpdate pour réduire l'empreinte mémoire.
# ---------------------------------------------
# Dépendances recommandées:
#   pip install streamlit pandas gtfs-realtime-bindings protobuf
# ---------------------------------------------

from __future__ import annotations

import io
import gc
import os
import sys
import zipfile
import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

# GTFS-RT protobuf (optionnel selon ton RT)
# Fallback JSON si le fichier RT est en JSON.
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
# Utilitaires généraux
# ---------------------------------------------

def _bytes_hash(b: bytes) -> str:
    """Hash stable pour cache et clés."""
    return hashlib.sha256(b).hexdigest()


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Trim des colonnes string et normalisation whitespace."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def _hms_to_seconds(hms: str) -> Optional[int]:
    """Convertit 'HH:MM:SS' (HH peut dépasser 24) en secondes."""
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
    """Retourne le nom exact d'un membre (insensible à la casse) ou None."""
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
            usecols=[c for c in usecols if c in pd.read_csv(io.StringIO(""), nrows=0).columns] if False else usecols,  # noop but kept for readability
            dtype=dtypes,
            chunksize=chunksize
        )
        for chunk in reader:
            # Normalisation légère
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
      - rt_trips: une ligne par trip (trip_id, route_id, start_date, start_time, timestamp)
      - rt_stop_updates: une ligne par arrêt (trip_id, stop_id, stop_sequence, arrival_time, departure_time, arrival_delay, departure_delay)
      - rt_trip_ids: set de trip_id
    """
    # Tentative Protobuf d'abord
    rt_trips_records: List[Dict] = []
    rt_su_records: List[Dict] = []
    rt_trip_ids: Set[str] = set()

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

                if trip_id:
                    rt_trip_ids.add(trip_id)

                rt_trips_records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "start_time": start_time,
                    "rt_timestamp": int(ts) if ts else pd.NA
                })

                for stu in tu.stop_time_update:
                    su = {
                        "trip_id": trip_id,
                        "stop_id": (stu.stop_id or "").strip(),
                        "stop_sequence": getattr(stu, "stop_sequence", pd.NA),
                        "arrival_time": getattr(stu.arrival, "time", pd.NA) if stu.HasField("arrival") else pd.NA,
                        "departure_time": getattr(stu.departure, "time", pd.NA) if stu.HasField("departure") else pd.NA,
                        "arrival_delay": getattr(stu.arrival, "delay", pd.NA) if stu.HasField("arrival") else pd.NA,
                        "departure_delay": getattr(stu.departure, "delay", pd.NA) if stu.HasField("departure") else pd.NA,
                        "schedule_relationship": str(stu.schedule_relationship) if hasattr(stu, "schedule_relationship") else pd.NA,
                    }
                    rt_su_records.append(su)
        except Exception:
            # On va essayer JSON ci-dessous
            pass

    if not rt_trips_records and not rt_su_records:
        # Fallback JSON (ex: dict {'entity':[{'trip_update': ...}]})
        import json
        try:
            data = json.loads(rt_bytes.decode("utf-8"))
            for entity in data.get("entity", []):
                tu = entity.get("trip_update")
                if not tu:
                    continue
                trip = tu.get("trip", {})
                trip_id = (trip.get("trip_id") or "").strip()
                route_id = (trip.get("route_id") or "").strip()
                start_date = (trip.get("start_date") or "").strip()
                start_time = (trip.get("start_time") or "").strip()
                ts = tu.get("timestamp")

                if trip_id:
                    rt_trip_ids.add(trip_id)

                rt_trips_records.append({
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "start_date": start_date,
                    "start_time": start_time,
                    "rt_timestamp": int(ts) if ts else pd.NA
                })

                for stu in tu.get("stop_time_update", []):
                    arr = stu.get("arrival", {})
                    dep = stu.get("departure", {})
                    su = {
                        "trip_id": trip_id,
                        "stop_id": (stu.get("stop_id") or "").strip(),
                        "stop_sequence": stu.get("stop_sequence"),
                        "arrival_time": arr.get("time"),
                        "departure_time": dep.get("time"),
                        "arrival_delay": arr.get("delay"),
                        "departure_delay": dep.get("delay"),
                        "schedule_relationship": str(stu.get("schedule_relationship")) if "schedule_relationship" in stu else pd.NA,
                    }
                    rt_su_records.append(su)
        except Exception as e:
            raise ValueError("Impossible de parser le TripUpdate (ni protobuf ni JSON).") from e

    rt_trips = pd.DataFrame(rt_trips_records)
    rt_su = pd.DataFrame(rt_su_records)

    # Types légers
    if not rt_trips.empty:
        rt_trips["trip_id"] = rt_trips["trip_id"].astype("string")
        if "route_id" in rt_trips:
            rt_trips["route_id"] = rt_trips["route_id"].astype("string")
        for c in ["start_date", "start_time"]:
            if c in rt_trips:
                rt_trips[c] = rt_trips[c].astype("string")
        if "rt_timestamp" in rt_trips:
            rt_trips["rt_timestamp"] = pd.to_numeric(rt_trips["rt_timestamp"], errors="coerce").astype("Int64")

    if not rt_su.empty:
        for c in ["trip_id", "stop_id", "schedule_relationship"]:
            if c in rt_su:
                rt_su[c] = rt_su[c].astype("string")
        for c in ["stop_sequence", "arrival_time", "departure_time", "arrival_delay", "departure_delay"]:
            if c in rt_su:
                rt_su[c] = pd.to_numeric(rt_su[c], errors="coerce").astype("Int64")

    return rt_trips, rt_su, rt_trip_ids


# ---------------------------------------------
# Chargement GTFS statique filtré par trips RT
# ---------------------------------------------

@st.cache_data(show_spinner=False)
def load_trips_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    dtypes = {
        "route_id": "string",
        "service_id": "string",
        "trip_id": "string",
        "trip_headsign": "string",
        "direction_id": "Int64",
        "shape_id": "string",
    }
    if not keep_trip_ids:
        # Si pas de trips dans RT, charger minimalement (evite gros DF)
        return pd.DataFrame(columns=usecols)
    df = _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "trips.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=200_000, keep_order_cols=usecols
    )
    return df


@st.cache_data(show_spinner=False)
def load_stop_times_filtered(gtfs_zip_bytes: bytes, keep_trip_ids: Set[str]) -> pd.DataFrame:
    usecols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    dtypes = {
        "trip_id": "string",
        "arrival_time": "string",
        "departure_time": "string",
        "stop_id": "string",
        "stop_sequence": "Int64",
    }
    if not keep_trip_ids:
        return pd.DataFrame(columns=usecols)
    df = _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stop_times.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="trip_id", filter_values=keep_trip_ids,
        chunksize=400_000,  # souvent très gros -> chunk plus large mais filtré
        keep_order_cols=usecols
    )
    return df


@st.cache_data(show_spinner=False)
def load_stops_filtered(gtfs_zip_bytes: bytes, keep_stop_ids: Set[str]) -> pd.DataFrame:
    usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    dtypes = {
        "stop_id": "string",
        "stop_name": "string",
        "stop_lat": "float64",
        "stop_lon": "float64",
    }
    if not keep_stop_ids:
        return pd.DataFrame(columns=usecols)
    df = _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "stops.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="stop_id", filter_values=keep_stop_ids,
        chunksize=200_000, keep_order_cols=usecols
    )
    return df


@st.cache_data(show_spinner=False)
def load_routes_filtered(gtfs_zip_bytes: bytes, keep_route_ids: Set[str]) -> pd.DataFrame:
    usecols = ["route_id", "route_short_name", "route_long_name"]
    dtypes = {
        "route_id": "string",
        "route_short_name": "string",
        "route_long_name": "string",
    }
    if not keep_route_ids:
        return pd.DataFrame(columns=usecols)
    df = _read_csv_from_zip_filtered(
        gtfs_zip_bytes, "routes.txt",
        usecols=usecols, dtypes=dtypes,
        filter_col="route_id", filter_values=keep_route_ids,
        chunksize=200_000, keep_order_cols=usecols
    )
    return df


@st.cache_data(show_spinner=False)
def load_calendar(gtfs_zip_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal_cols = ["service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"]
    cal_dtypes = {c:"Int64" for c in cal_cols if c not in ["service_id","start_date","end_date"]}
    cal_dtypes.update({"service_id":"string","start_date":"string","end_date":"string"})

    cd_cols = ["service_id","date","exception_type"]
    cd_dtypes = {"service_id":"string","date":"string","exception_type":"Int64"}

    zf = zipfile.ZipFile(io.BytesIO(gtfs_zip_bytes))

    # calendar.txt
    cal_df = pd.DataFrame(columns=cal_cols)
    member = _get_zip_member_case_insensitive(zf, "calendar.txt")
    if member:
        with zf.open(member, "r") as fb:
            text = io.TextIOWrapper(fb, encoding="utf-8", newline="")
            cal_df = pd.read_csv(text, usecols=[c for c in cal_cols if c], dtype=cal_dtypes)
            cal_df = _normalize_cols(cal_df)

    # calendar_dates.txt
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
    Jointure RT vs Schedule au niveau stop; calcul des heures planifiées en secondes.
    * Utilise trip_id + stop_id (et fallback sur stop_sequence si présent).
    * N'essaie PAS de recalculer heures absolues (timezone / service day) -> on utilise les delays RT si fournis.
    """
    if rt_stop_updates.empty or stop_times_filtered.empty:
        return pd.DataFrame()

    # Prépare schedule: secondes depuis minuit
    sched = stop_times_filtered.copy()
    for col in ["arrival_time", "departure_time"]:
        if col in sched.columns:
            sched[col + "_sec"] = sched[col].map(_hms_to_seconds).astype("Int64")

    # Jointure principale par (trip_id, stop_id)
    # NB: dans certains feeds, stop_id peut manquer; fallback: (trip_id, stop_sequence)
    left = rt_stop_updates.copy()

    join_cols = []
    if "stop_id" in left.columns and "stop_id" in sched.columns:
        join_cols = ["trip_id", "stop_id"]
    elif "stop_sequence" in left.columns and "stop_sequence" in sched.columns:
        join_cols = ["trip_id", "stop_sequence"]
    else:
        # Pas de clé fiable -> on tente uniquement par trip_id (risqué)
        join_cols = ["trip_id"]

    merged = pd.merge(
        left,
        sched,
        how="left",
        on=join_cols,
        suffixes=("_rt", "_sched"),
        copy=False
    )

    # Ajoute libellé d'arrêt si disponible
    if not stops_df.empty and "stop_id" in merged.columns:
        merged = pd.merge(
            merged,
            stops_df[["stop_id", "stop_name"]],
            how="left",
            on="stop_id",
            copy=False
        )

    # Types légers
    for c in ["arrival_time_sec", "departure_time_sec"]:
        if c in merged.columns:
            merged[c] = merged[c].astype("Int64")

    return merged


def summarize_rt(
    rt_trips: pd.DataFrame,
    rt_stop_updates: pd.DataFrame,
    trips_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Résumé par trip: nb d'updates, retard moyen (si disponible), avec infos route.
    """
    if rt_trips.empty:
        return pd.DataFrame()

    su = rt_stop_updates.copy()
    agg = []
    if not su.empty:
        # Moyenne des delays si présents
        su["delay_any"] = su[["arrival_delay", "departure_delay"]].mean(axis=1, skipna=True)
        agg = su.groupby("trip_id", as_index=False).agg(
            updates_count=("stop_id", "count"),
            avg_delay_sec=("delay_any", "mean")
        )
    else:
        agg = pd.DataFrame({"trip_id": rt_trips["trip_id"].unique(), "updates_count": 0, "avg_delay_sec": pd.NA})

    out = pd.merge(rt_trips, agg, on="trip_id", how="left")
    if not trips_df.empty:
        out = pd.merge(out, trips_df[["trip_id", "route_id", "trip_headsign", "direction_id"]], on="trip_id", how="left", suffixes=("", "_sched"))

    if not routes_df.empty and "route_id" in out.columns:
        out = pd.merge(out, routes_df, on="route_id", how="left")

    # Types
    if "avg_delay_sec" in out.columns:
        out["avg_delay_sec"] = pd.to_numeric(out["avg_delay_sec"], errors="coerce").round(1).astype("Float64")

    return out


# ---------------------------------------------
# UI Streamlit
# ---------------------------------------------

def main():
    st.title("🚌 Analyse GTFS-RT TripUpdates vs GTFS (optimisée gros fichiers)")
    st.caption("Chargement ciblé des trips présents dans le flux RT pour minimiser l'empreinte mémoire. Conçu pour des GTFS statiques volumineux (~70 Mo et +).")

    with st.sidebar:
        st.header("Fichiers d'entrée")
        gtfs_zip = st.file_uploader("GTFS statique (.zip)", type=["zip"])
        rt_file = st.file_uploader("TripUpdates GTFS-RT (.pb/.bin ou .json)", type=["pb", "bin", "json"])

        st.divider()
        st.subheader("Options")
        focus_only_rt_trips = st.checkbox("Se focaliser sur les voyages du TripUpdate (recommandé)", value=True)
        limit_trips = st.number_input("Limiter à N trips (optionnel, pour tester/accélérer)", min_value=0, value=0, step=100, help="0 = pas de limite")
        show_raw_tables = st.checkbox("Afficher les tables brutes (debug)", value=False)

        st.divider()
        run_btn = st.button("Lancer l'analyse", type="primary", use_container_width=True)

    if not run_btn:
        st.info("💡 Charge les fichiers dans la barre latérale puis clique sur **Lancer l'analyse**.")
        return

    # Contrôles
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

    # Parse TripUpdates
    with st.status("📥 Lecture du TripUpdate en cours...", expanded=False) as status:
        try:
            rt_trips, rt_su, rt_trip_ids = parse_tripupdates_rt(rt_bytes)
        except Exception as e:
            st.exception(e)
            status.update(label="Erreur lors du parsing du TripUpdate.", state="error")
            return
        status.update(label=f"TripUpdate chargé: {len(rt_trips)} trips, {len(rt_su)} mises à jour d'arrêt.", state="complete")

    if focus_only_rt_trips:
        keep_trip_ids = {tid for tid in rt_trip_ids if isinstance(tid, str) and len(tid) > 0}
    else:
        keep_trip_ids = set()  # signifie: pas de filtrage (mais déconseillé pour gros jeux)
        st.warning("⚠️ Tu as désactivé le filtrage par trips RT — ça peut utiliser beaucoup de mémoire.")

    if limit_trips and limit_trips > 0 and keep_trip_ids:
        # Limiter l'analyse à N trips pour accélérer tests
        keep_trip_ids = set(list(keep_trip_ids)[: int(limit_trips)])

    # Chargement ciblé des tables GTFS
    progress = st.progress(0, text="Chargement des trips...")
    trips_df = load_trips_filtered(gtfs_bytes, keep_trip_ids)
    progress.progress(20, text="Chargement des stop_times (filtré)...")
    stop_times_df = load_stop_times_filtered(gtfs_bytes, keep_trip_ids)

    # Déduire stops et routes à charger
    progress.progress(40, text="Préparation des listes stops/routes...")
    keep_stop_ids: Set[str] = set()
    keep_route_ids: Set[str] = set()
    if not stop_times_df.empty and "stop_id" in stop_times_df:
        keep_stop_ids = set(stop_times_df["stop_id"].dropna().astype("string").unique().tolist())
    if not trips_df.empty and "route_id" in trips_df:
        keep_route_ids = set(trips_df["route_id"].dropna().astype("string").unique().tolist())

    progress.progress(60, text="Chargement des stops (filtré)...")
    stops_df = load_stops_filtered(gtfs_bytes, keep_stop_ids)

    progress.progress(75, text="Chargement des routes (filtré)...")
    routes_df = load_routes_filtered(gtfs_bytes, keep_route_ids)

    # (Optionnel) calendar / calendar_dates pour analyses par date (on les charge sans filtrage car petits d'habitude)
    progress.progress(85, text="Chargement calendar/calendar_dates...")
    cal_df, cd_df = load_calendar(gtfs_bytes)

    # Jointures / Analyses
    progress.progress(92, text="Jointure Schedule vs RT...")
    sched_vs_rt = compute_schedule_vs_rt(rt_su, stop_times_df, stops_df)

    progress.progress(98, text="Synthèse...")
    summary_df = summarize_rt(rt_trips, rt_su, trips_df, routes_df)

    progress.progress(100, text="Terminé ✅")
    st.success("Analyse terminée.")

    # -----------------------------
    # Affichage résultats
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Trips dans RT", f"{rt_trips['trip_id'].nunique() if not rt_trips.empty else 0:,}")
    with c2:
        st.metric("Mises à jour d'arrêts", f"{len(rt_su):,}")
    with c3:
        st.metric("Trips chargés (GTFS)", f"{trips_df['trip_id'].nunique() if not trips_df.empty else 0:,}")
    with c4:
        st.metric("Routes concernées", f"{routes_df['route_id'].nunique() if not routes_df.empty else 0:,}")

    st.subheader("Résumé par voyage")
    if summary_df.empty:
        st.info("Aucun résumé disponible (vérifie le contenu du TripUpdate).")
    else:
        # Colonnes pertinentes et tri
        order_cols = [c for c in ["trip_id","route_id","route_short_name","route_long_name","trip_headsign","direction_id","updates_count","avg_delay_sec","start_date","start_time","rt_timestamp"] if c in summary_df.columns]
        st.dataframe(summary_df[order_cols].sort_values(["route_id","trip_id"], na_position="last"), use_container_width=True, height=360)

        csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le résumé CSV", data=csv, file_name="resume_trips_rt.csv", mime="text/csv")

    st.subheader("Détails arrêt par arrêt (Schedule vs RT)")
    if sched_vs_rt.empty:
        st.info("Aucun détail d'arrêt joint — possible si les clés ne correspondent pas (stop_id/stop_sequence) ou si le TripUpdate est vide.")
    else:
        # Sélecteur de trip
        trip_choices = sorted(set(sched_vs_rt["trip_id"].dropna().astype(str).unique()))
        selected_trip = st.selectbox("Sélectionne un trip pour voir le détail", options=trip_choices)
        det = sched_vs_rt[sched_vs_rt["trip_id"] == selected_trip].copy()

        # Tri par stop_sequence si dispo, sinon par stop_id
        if "stop_sequence" in det.columns:
            det = det.sort_values("stop_sequence")
        elif "stop_id" in det.columns:
            det = det.sort_values("stop_id")

        # Colonnes d'intérêt
        cols = [c for c in [
            "trip_id",
            "stop_sequence",
            "stop_id",
            "stop_name",
            "arrival_time_sec",
            "departure_time_sec",
            "arrival_delay",
            "departure_delay",
            "arrival_time",
            "departure_time",
        ] if c in det.columns]

        st.dataframe(det[cols], use_container_width=True, height=400)

        csvd = det.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les détails (trip sélectionné)", data=csvd, file_name=f"details_trip_{selected_trip}.csv", mime="text/csv")

    # Debug / Tables brutes
    if show_raw_tables:
        with st.expander("RT - Trips (brut)"):
            st.dataframe(rt_trips, use_container_width=True, height=250)
        with st.expander("RT - Stop Updates (brut)"):
            st.dataframe(rt_su, use_container_width=True, height=250)
        with st.expander("GTFS - trips (filtré)"):
            st.dataframe(trips_df, use_container_width=True, height=250)
        with st.expander("GTFS - stop_times (filtré)"):
            st.dataframe(stop_times_df.head(1000), use_container_width=True, height=250)
        with st.expander("GTFS - stops (filtré)"):
            st.dataframe(stops_df, use_container_width=True, height=250)
        with st.expander("GTFS - routes (filtré)"):
            st.dataframe(routes_df, use_container_width=True, height=250)
        with st.expander("GTFS - calendar / calendar_dates"):
            st.dataframe(cal_df, use_container_width=True, height=250)
            st.dataframe(cd_df, use_container_width=True, height=250)

    # Libération mémoire (utile pour gros jeux)
    del gtfs_bytes, rt_bytes
    gc.collect()


if __name__ == "__main__":
    main()

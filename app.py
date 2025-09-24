# -*- coding: utf-8 -*-
import os
import io
import json
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import altair as alt

from gtfsrt_tripupdates_report import analyze_tripupdates, load_static_gtfs

st.set_page_config(page_title="🚌 Analyseur TripUpdates", layout="wide")

st.title("🚌 Analyseur GTFS‑realtime : TripUpdates")
st.write(
    "Charge un fichier **TripUpdates .pb** (Protocol Buffer) et, optionnellement, un **GTFS statique** pour des validations avancées et la comparaison au planifié. "
    "Utilise les filtres pour explorer, et télécharge les résultats."
)

# ------------------------------
# Uploaders & options (sidebar)
# ------------------------------
with st.sidebar:
    st.header("Fichiers")
    tu_file = st.file_uploader("Fichier TripUpdates (.pb)", type=["pb", "bin"])
    gtfs_file = st.file_uploader("GTFS statique (zip) (optionnel)", type=["zip"])
    st.divider()
    st.header("Options d'affichage")
    default_tz = "America/Toronto"
    tz_input = st.text_input("Fuseau horaire (IANA)", value=default_tz, help="Ex.: America/Toronto, America/Montreal, UTC")
    st.caption("Astuce : pour de gros fichiers, augmentez la taille via `.streamlit/config.toml` → [server] maxUploadSize = 200")
    run_button = st.button("Analyser", type="primary")

# ------------------------------
# Cache d'analyse
# ------------------------------
@st.cache_data(show_spinner=False)
def run_analysis_cached(tu_bytes: bytes, gtfs_bytes: bytes | None):
    """
    Exécute l'analyse en environnement temporaire et met en cache le résultat pour
    ne pas recalculer à chaque rechargement de l'app.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tu_path = os.path.join(tmpdir, "tripupdates.pb")
        with open(tu_path, "wb") as f:
            f.write(tu_bytes)

        gtfs_path = None
        static_gtfs = {
            "stops": pd.DataFrame(),
            "trips": pd.DataFrame(),
            "stop_times": pd.DataFrame(),
            "routes": pd.DataFrame(),
            "agency": pd.DataFrame()
        }
        if gtfs_bytes:
            gtfs_path = os.path.join(tmpdir, "gtfs.zip")
            with open(gtfs_path, "wb") as f:
                f.write(gtfs_bytes)
            static_gtfs = load_static_gtfs(gtfs_path)

        analysis = analyze_tripupdates(tu_path, static_gtfs)
        return analysis, static_gtfs

# ------------------------------
# Helpers UI / Data
# ------------------------------
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
    Sélectionne l'horodatage d'événement pour les analyses temporelles :
    priorise departure_time, sinon arrival_time.
    """
    if stu_df.empty:
        return pd.Series(dtype="float")
    t = stu_df["departure_time"].where(stu_df["departure_time"].notna(), stu_df["arrival_time"])
    return pd.to_numeric(t, errors="coerce")

def _add_local_hour(df: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Ajoute 'hour_local' (0..23) selon le fuseau local.
    (Conservée si tu veux une vue par heure ; non utilisée pour la heatmap 10 min.)
    """
    s = _event_epoch_series(df)
    if s.empty:
        df = df.copy()
        df["hour_local"] = pd.NA
        return df
    try:
        dt = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(ZoneInfo(tz_str))
    except Exception:
        dt = pd.to_datetime(s, unit="s", utc=True)  # fallback UTC
    out = df.copy()
    out["hour_local"] = dt.dt.hour
    return out

def _add_local_bin10(df: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Ajoute deux colonnes :
      - bin10_minute : minute depuis minuit locale arrondie à 10 min (0, 10, 20, …, 1430)
      - bin10_label  : libellé 'HH:MM' de la tranche (08:30, 08:40, …)
    Priorise departure_time, sinon arrival_time pour l'événement temporel.
    """
    s = _event_epoch_series(df)  # série de timestamps (epoch s)
    out = df.copy()
    if s.empty:
        out["bin10_minute"] = pd.NA
        out["bin10_label"] = pd.NA
        return out

    # UTC → fuseau local (IANA)
    try:
        dt_local = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(ZoneInfo(tz_str))
    except Exception:
        # fallback : reste en UTC si tz invalide
        dt_local = pd.to_datetime(s, unit="s", utc=True)

    minute_of_day = dt_local.dt.hour * 60 + dt_local.dt.minute
    bin10 = (minute_of_day // 10) * 10  # arrondi inférieur à 10 min

    out["bin10_minute"] = bin10

    # libellé 'HH:MM'
    hh = (bin10 // 60).astype(int)
    mm = (bin10 % 60).astype(int)
    out["bin10_label"] = hh.map("{:02d}".format) + ":" + mm.map("{:02d}".format)

    return out

def _build_html_report(analysis: dict) -> str:
    """Construit un HTML autonome (texte + tableaux) téléchargeable."""
    payload = {
        "meta": analysis["meta"],
        "summary": analysis["summary"],
        "timestamp_quality": analysis["ts_quality"],
        "schedule_compare": analysis.get("schedule_stats", {}),
        "counts": {
            "trip_rows": int(len(analysis["trips_df"])),
            "stop_time_updates": int(len(analysis["stu_df"])),
            "anomalies": int(len(analysis["anomalies"])),
            "schedule_compare_rows": int(len(analysis.get("schedule_compare_df", pd.DataFrame())))
        }
    }
    # On réutilise la fabrique HTML du module pour rester cohérents
    from gtfsrt_tripupdates_report import _build_summary_html
    return _build_summary_html({
        "meta": payload["meta"],
        "summary": payload["summary"],
        "timestamp_quality": payload["timestamp_quality"],
        "schedule_compare": payload["schedule_compare"]
    })

# ------------------------------
# Analyse
# ------------------------------
if run_button and tu_file is not None:
    with st.spinner("Analyse en cours…"):
        analysis, static_gtfs = run_analysis_cached(tu_file.getvalue(), gtfs_file.getvalue() if gtfs_file else None)

    st.success("Analyse terminée ✅")

    # Détecte tz par défaut à partir du GTFS s'il existe
    if static_gtfs and not static_gtfs.get("agency", pd.DataFrame()).empty:
        try:
            tz_detected = str(static_gtfs["agency"].iloc[0].get("agency_timezone")) or tz_input
            tz_input = tz_detected
        except Exception:
            pass

    trips_df = analysis["trips_df"]
    stu_df = analysis["stu_df"]
    anomalies_df = analysis["anomalies"]
    sched_df = analysis.get("schedule_compare_df", pd.DataFrame())
    schedule_stats = analysis.get("schedule_stats", {})

    # ------------------ Filtres ------------------
    st.sidebar.header("Filtres")

    # Options route_id
    route_opts = sorted([r for r in trips_df["route_id"].dropna().unique().tolist() if r != ""])
    route_sel = st.sidebar.multiselect("Filtrer par route_id", options=route_opts, default=[])

    # TripDescriptor schedule_relationship
    map_trip = _sr_label_map_trip()
    trip_types = [(0, "SCHEDULED"), (1, "ADDED"), (2, "UNSCHEDULED"), (3, "CANCELED")]
    trip_sel_labels = st.sidebar.multiselect(
        "Types de voyages", options=[lbl for _, lbl in trip_types], default=[lbl for _, lbl in trip_types]
    )
    trip_sel_ids = {k for k, v in trip_types if v in set(trip_sel_labels)}

    # StopTimeUpdate schedule_relationship
    map_stu = _sr_label_map_stu()
    stu_types = [(0, "SCHEDULED"), (1, "SKIPPED"), (2, "NO_DATA")]
    stu_sel_labels = st.sidebar.multiselect(
        "Types d'arrêts (STU)", options=[lbl for _, lbl in stu_types], default=[lbl for _, lbl in stu_types]
    )
    stu_sel_ids = {k for k, v in stu_types if v in set(stu_sel_labels)}

    trip_id_query = st.sidebar.text_input("Recherche trip_id (contient)", value="")

    # Applique filtres
    trips_view = trips_df.copy()
    if route_sel:
        trips_view = trips_view[trips_view["route_id"].isin(route_sel)]
    if trip_sel_ids:
        trips_view = trips_view[trips_view["trip_schedule_relationship"].isin(trip_sel_ids)]
    if trip_id_query:
        trips_view = trips_view[trips_view["trip_id"].fillna("").str.contains(trip_id_query, case=False)]

    # Pour filtrer les STU selon route/trip
    stu_view = stu_df.merge(
        trips_df[["trip_key", "route_id", "trip_id"]], on="trip_key", how="left", suffixes=("", "_t")
    )
    if route_sel:
        stu_view = stu_view[stu_view["route_id"].isin(route_sel)]
    if trip_id_query:
        stu_view = stu_view[stu_view["trip_id_t"].fillna("").str.contains(trip_id_query, case=False)]
    if stu_sel_ids:
        stu_view = stu_view[stu_view["stu_schedule_relationship"].isin(stu_sel_ids)]

    # ------------------ KPIs ------------------
    st.subheader("Résumé")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_trips = int(trips_view["trip_key"].nunique())
    canceled_trips = int((trips_view["trip_schedule_relationship"] == 3).sum())  # CANCELED
    added_trips = int((trips_view["trip_schedule_relationship"] == 1).sum())     # ADDED
    unscheduled_trips = int((trips_view["trip_schedule_relationship"] == 2).sum())  # UNSCHEDULED

    # Partially canceled = trips avec ≥1 SKIPPED mais pas CANCELED
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

    # ------------------ Graphiques récapitulatifs ------------------
    st.markdown("### Graphiques récapitulatifs")
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

    # ------------------ Top 20 arrêts annulés ------------------
    st.markdown("### Top 20 des arrêts annulés (SKIPPED)")
    if not sk.empty:
        top_sk = (
            sk.groupby("stop_id")
            .size().reset_index(name="compte")
            .sort_values("compte", ascending=False)
            .head(20)
        )
        # Ajoute stop_name si disponible
        stops_static = static_gtfs.get("stops", pd.DataFrame())
        if not stops_static.empty and "stop_name" in stops_static.columns:
            top_sk = top_sk.merge(stops_static[["stop_id", "stop_name"]], on="stop_id", how="left")
        chart_top = (
            alt.Chart(top_sk)
            .mark_bar()
            .encode(
                x=alt.X("compte:Q", title="Nombre de SKIPPED"),
                y=alt.Y("stop_name:N", sort="-x", title="Arrêt").axis(labelLimit=250)
                if "stop_name" in top_sk.columns else alt.Y("stop_id:N", sort="-x", title="stop_id"),
                tooltip=top_sk.columns.tolist()
            )
            .properties(height=400)
        )
        st.altair_chart(chart_top, use_container_width=True)
    else:
        st.info("Aucun arrêt annulé dans la sélection.")

    # ------------------ Heatmap par tranches de 10 minutes ------------------
    st.markdown("### Heatmap par tranches de 10 minutes et type de passage")
    if not stu_view.empty:
        try:
            # Ajoute les tranches 10 min locales
            stu_10 = _add_local_bin10(stu_view, tz_input)
        except Exception:
            stu_10 = _add_local_bin10(stu_view, "UTC")

        # Map du type lisible
        stu_10 = stu_10.assign(
            type=lambda d: d["stu_schedule_relationship"].map(_sr_label_map_stu()).fillna("AUTRE")
        )

        # Agrégation par (tranche 10 min, type)
        dist_10 = (
            stu_10
            .dropna(subset=["bin10_minute"])
            .groupby(["bin10_minute", "bin10_label", "type"])
            .size()
            .reset_index(name="compte")
            .sort_values(["bin10_minute", "type"])
        )

        if not dist_10.empty:
            heat10 = (
                alt.Chart(dist_10)
                .mark_rect()
                .encode(
                    # Trie X par la valeur numérique de la tranche (évite l'ordre alpha '01:00' vs '10:00')
                    x=alt.X("bin10_label:N",
                            sort=alt.SortField(field="bin10_minute", order="ascending"),
                            title="Heure locale (tranches de 10 min)"),
                    y=alt.Y("type:N", title="Type de passage"),
                    color=alt.Color("compte:Q", title="Compte", scale=alt.Scale(scheme="blues")),
                    tooltip=["bin10_label", "type", "compte"]
                )
                .properties(height=320)
            )
            st.altair_chart(heat10, use_container_width=True)
        else:
            st.info("Pas de données suffisantes pour la heatmap en tranches de 10 minutes.")
    else:
        st.info("Aucun STU à représenter pour la heatmap.")

    # ------------------ Comparaison au planifié ------------------
    st.markdown("### Écart vs horaire planifié (si GTFS fourni)")
    if not sched_df.empty and schedule_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes comparées", f"{schedule_stats.get('rows_compared', 0):,}".replace(",", " "))
        arr = schedule_stats.get("arrival", {})
        dep = schedule_stats.get("departure", {})
        c2.metric("Arrival — médiane (s)", f"{arr.get('median_signed_sec', 0):.0f}")
        c3.metric("Departure — médiane (s)", f"{dep.get('median_signed_sec', 0):.0f}")
        c4.metric("≤ 5 min (arrival)", f"{arr.get('within_5_min_pct', 0):.1f}%")

        # Histogramme des deltas (en minutes)
        sched_plot = sched_df.copy()
        sched_plot["arr_delta_min"] = pd.to_numeric(sched_plot["arr_delta_sec"], errors="coerce") / 60.0
        sched_plot["dep_delta_min"] = pd.to_numeric(sched_plot["dep_delta_sec"], errors="coerce") / 60.0

        charts = st.tabs(["Histogramme arrival", "Histogramme departure"])
        with charts[0]:
            s = sched_plot["arr_delta_min"].dropna()
            if not s.empty:
                chart_arr = (
                    alt.Chart(pd.DataFrame({"delta_min": s}))
                    .mark_bar()
                    .encode(x=alt.X("delta_min:Q", bin=alt.Bin(maxbins=60), title="Écart (minutes, +retard / -avance)"),
                            y=alt.Y("count():Q", title="Nombre"))
                    .properties(height=300)
                )
                st.altair_chart(chart_arr, use_container_width=True)
            else:
                st.info("Aucun delta arrival disponible.")
        with charts[1]:
            s = sched_plot["dep_delta_min"].dropna()
            if not s.empty:
                chart_dep = (
                    alt.Chart(pd.DataFrame({"delta_min": s}))
                    .mark_bar()
                    .encode(x=alt.X("delta_min:Q", bin=alt.Bin(maxbins=60), title="Écart (minutes)"),
                            y=alt.Y("count():Q", title="Nombre"))
                    .properties(height=300)
                )
                st.altair_chart(chart_dep, use_container_width=True)
            else:
                st.info("Aucun delta departure disponible.")
    else:
        st.info("Aucune comparaison planifié vs prédiction (ajoute un GTFS statique pour activer).")

    # ------------------ Détails JSON ------------------
    st.markdown("### Détails (JSON)")
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

    # ------------------ Téléchargements ------------------
    st.markdown("### Téléchargements")
    cdl1, cdl2, cdl3, cdl4, cdl5 = st.columns(5)
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
        st.download_button("⬇️ summary.json",
                           json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                           "summary.json", mime="application/json")
    with cdl5:
        html_bytes = _build_html_report(analysis).encode("utf-8")
        st.download_button("⬇️ summary.html", html_bytes, "summary.html", mime="text/html")

    # ------------------ Tables ------------------
    st.markdown("### Tables")
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
        sv["stu_type"] = sv["stu_schedule_relationship"].map(_sr_label_map_stu()).fillna("SCHEDULED")
        st.dataframe(sv, use_container_width=True)
    if show_anom_flag:
        st.subheader("Anomalies")
        st.dataframe(anomalies_df, use_container_width=True)

else:
    st.info("Charge au moins un fichier **TripUpdates .pb** puis clique **Analyser** dans la barre latérale.")

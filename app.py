def _add_local_bin10(df: pd.DataFrame, tz_str: str) -> pd.DataFrame:
    """
    Ajoute deux colonnes :
      - bin10_minute : minute depuis minuit locale arrondie à 10 min (0, 10, 20, …, 1430) [dtype: Int64]
      - bin10_label  : libellé 'HH:MM' de la tranche (08:30, 08:40, …) [dtype: string]
    Priorise departure_time, sinon arrival_time pour l'événement temporel.
    """
    s = _event_epoch_series(df)  # timestamps epoch (secondes), NaN possible
    out = df.copy()

    if s.empty:
        # Crée des colonnes vides avec dtypes “nullable”
        out["bin10_minute"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["bin10_label"] = pd.Series([pd.NA] * len(out), dtype="string")
        return out

    # UTC → fuseau local (IANA)
    try:
        dt_local = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(ZoneInfo(tz_str))
    except Exception:
        # fallback : reste en UTC si tz invalide
        dt_local = pd.to_datetime(s, unit="s", utc=True)

    # Arrondi à la tranche 10 min (évite nos propres divisions + gère NaT proprement)
    dt10 = dt_local.dt.floor("10min")  # NaT reste NaT

    # minute depuis minuit locale
    # Utilise les dtypes “nullable” pour accepter les NaT → NA
    minute_of_day = dt10.dt.hour * 60 + dt10.dt.minute
    out["bin10_minute"] = pd.to_numeric(minute_of_day, errors="coerce").astype("Int64")

    # Libellé de tranche
    out["bin10_label"] = dt10.dt.strftime("%H:%M").astype("string")

    return out

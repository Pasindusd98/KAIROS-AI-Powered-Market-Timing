from nixtla import NixtlaClient
import pandas as pd

client = NixtlaClient(api_key="nixak-bd7233b7bbf65ccdf97178da5ef5409e849f6f64cc3419542240012b5672d96101c617ae9eeb301c")

def timegpt_forecast(df, horizon=5, timeframe="1h"):
    """
    TimeGPT-compatible forecast with proper column selection
    """

    # ---- Frequency mapping ----
    freq_map = {
        "1m": "T",
        "1h": "H",
        "4h": "4H"
    }
    freq = freq_map.get(timeframe, "H")

    # ---- CLEAN TIME SERIES ----
    ts = df.copy()
    ts = ts.sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]

    # Keep ONLY close price
    ts = ts[["close"]]

    # Force regular frequency
    full_index = pd.date_range(
        start=ts.index.min(),
        end=ts.index.max(),
        freq=freq
    )

    ts = ts.reindex(full_index)
    ts["close"] = ts["close"].ffill().bfill()

    # Build TimeGPT dataframe
    ts = ts.reset_index()
    ts.columns = ["ds", "y"]

    # ---- FORECAST ----
    forecast = client.forecast(
        df=ts,
        h=horizon,
        freq=freq,
        model="timegpt-1"
    )

    last_price = ts["y"].iloc[-1]
    future_price = forecast["TimeGPT"].iloc[-1]

    if future_price > last_price:
        return "BUY", float(future_price - last_price)
    elif future_price < last_price:
        return "SELL", float(last_price - future_price)
    else:
        return "HOLD", 0.0


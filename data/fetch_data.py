import requests
import pandas as pd
from data.symbols import SYMBOL_MAP

TWELVE_DATA_KEY = "bbbcc9dd917746a688d87f8c7da7da98"

def fetch_forex(pair, interval):
    symbol = SYMBOL_MAP[pair]

    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={symbol}"
        f"&interval={interval}"
        f"&outputsize=500"
        f"&apikey={TWELVE_DATA_KEY}"
    )

    r = requests.get(url).json()

    if "values" not in r:
        raise ValueError(f"TwelveData error: {r}")

    df = pd.DataFrame(r["values"])

    # ✅ Convert datetime FIRST
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # ✅ Convert ONLY numeric columns
    numeric_cols = ["open", "high", "low", "close"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df.attrs["symbol"] = pair

    return df.sort_index()


def resample_4h(df):
    return df.resample("4H").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()
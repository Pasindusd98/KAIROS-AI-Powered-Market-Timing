import ta

def add_indicators(df):
    df["roc"] = df["close"].pct_change(3)
    df["volatility"] = df["close"].rolling(10).std()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"]
    ).average_true_range()
    df.dropna(inplace=True)
    return df

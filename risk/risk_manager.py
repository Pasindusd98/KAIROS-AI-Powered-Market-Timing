def calculate_sl_tp(price, atr, signal):
    if signal == "BUY":
        return price - atr*1.5, price + atr*3
    if signal == "SELL":
        return price + atr*1.5, price - atr*3
    return None, None

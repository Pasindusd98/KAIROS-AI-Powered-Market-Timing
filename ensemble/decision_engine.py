import math

def ensemble_decision(xgb_prob, timegpt_output=None):
    """
    xgb_prob: [sell, hold, buy]
    timegpt_output: ("BUY"/"SELL"/"HOLD", confidence)
    """

    # ---------- XGBoost ----------
    if len(xgb_prob) == 3:
        sell, hold, buy = xgb_prob
        xgb_score = buy - sell            # [-1, 1]
        xgb_conf = max(xgb_prob)          # [0, 1]
    else:
        xgb_score = float(xgb_prob[0])
        xgb_conf = abs(xgb_score)

    # ---------- TimeGPT ----------
    tg_score = 0.0
    tg_conf = 0.0

    if timegpt_output:
        tg_signal, tg_conf = timegpt_output

        signal_map = {
            "BUY": 1.0,
            "HOLD": 0.0,
            "SELL": -1.0
        }

        tg_score = signal_map.get(tg_signal, 0.0)
        tg_conf = min(max(float(tg_conf), 0.0), 1.0)

    # ---------- Ensemble ----------
    raw_score = (
        0.65 * xgb_score * xgb_conf +
        0.35 * tg_score * tg_conf
    )

    # ---------- FX-aware decision ----------
    THRESHOLD = 0.10   # LOWERED from 0.20

    if raw_score > THRESHOLD:
        signal = "BUY"
    elif raw_score < -THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ---------- Confidence ----------
    if signal == "HOLD":
        confidence = 1.0 - abs(math.tanh(raw_score))
    else:
        confidence = abs(math.tanh(raw_score))

    return signal, round(confidence, 3)

import numpy as np
from xgboost import XGBClassifier


def get_threshold(symbol: str) -> float:
    """
    Asset-aware threshold
    """
    if "XAU" in symbol:
        return 0.002      # Gold
    else:
        return 0.0003     # Forex


def train_xgb(df):
    """
    Train XGBoost classifier for BUY / HOLD / SELL
    """

    # ----------------------------
    # 1. Threshold
    # ----------------------------
    symbol = df.attrs.get("symbol", "")
    threshold = get_threshold(symbol)

    # ----------------------------
    # 2. Create labels
    # ----------------------------
    df = df.copy()

    df["future_return"] = df["close"].shift(-1) / df["close"] - 1

    df["y"] = 0
    df.loc[df["future_return"] > threshold, "y"] = 1
    df.loc[df["future_return"] < -threshold, "y"] = -1

    df = df.dropna()

    # ----------------------------
    # 3. Features
    # ----------------------------
    FEATURES = [
        "rsi",
        "ema20",
        "ema50",
        "atr",
        "roc",
        "volatility",
    ]

    X = df[FEATURES]
    y = df["y"]

    # ----------------------------
    # 4. SAFETY: class check
    # ----------------------------
    unique_classes = np.unique(y)

    if len(unique_classes) < 2:
        # Return a neutral dummy model (no crash, no fake confidence)
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            base_score=1 / 3,
            n_estimators=1
        )
        model.fit(X.iloc[:3], [0, 1, 2])
        return model

    # ----------------------------
    # 5. Train model (CORRECT)
    # ----------------------------
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        base_score=1 / 3,     
        random_state=42,
        tree_method="hist"
    )

    # shift labels from (-1, 0, 1) → (0, 1, 2)
    model.fit(X, y + 1)

    return model


def predict_xgb(model, X_last):
    """
    Returns probabilities: [sell, hold, buy]
    """
    prob = model.predict_proba(X_last)[0]
    return prob

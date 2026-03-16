import streamlit as st
from models.lstm_model import predict_future
from config import FOREX_PAIRS, TIMEFRAMES
from data.fetch_data import fetch_forex, resample_4h
from features.indicators import add_indicators
from models.xgb_model import train_xgb, predict_xgb
from api.timegpt_model import timegpt_forecast
from ensemble.decision_engine import ensemble_decision
from risk.risk_manager import calculate_sl_tp 
from visualization.charts import plot_chart


# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Project KAIROS 🚀 Forex & Gold AI Signal Platform")

pair = st.selectbox("Currency Pair", FOREX_PAIRS)
tf = st.selectbox("Timeframe", TIMEFRAMES.keys())


# --------------------------------------------------
# Data
# --------------------------------------------------
@st.cache_data(ttl=60)
def load_data(pair, tf):
    return fetch_forex(pair, tf)

df = load_data(pair, TIMEFRAMES[tf])

if tf == "4h":
    df = resample_4h(df)

df = add_indicators(df)


# --------------------------------------------------
# XGBoost
# --------------------------------------------------
xgb_model = train_xgb(df)

xgb_prob = predict_xgb(
    xgb_model,
    df[["rsi", "ema20", "ema50", "atr","roc","volatility"]].iloc[-1:]
)
# xgb_prob must be array-like


# --------------------------------------------------
# TimeGPT (skip for 1m)
# --------------------------------------------------
if tf != "1m":
    timegpt_signal, timegpt_conf = timegpt_forecast(df, timeframe=tf)
    timegpt_output = (timegpt_signal, float(timegpt_conf))
else:
    timegpt_output = None


# --------------------------------------------------
# Ensemble Decision
# --------------------------------------------------
signal, confidence = ensemble_decision(
    xgb_prob=xgb_prob,
    timegpt_output=timegpt_output
)


# --------------------------------------------------
# Risk Management
# --------------------------------------------------
sl, tp = calculate_sl_tp(
    df["close"].iloc[-1],
    df["atr"].iloc[-1],
    signal
)


# --------------------------------------------------
# Forecast for plotting
# --------------------------------------------------
future = predict_future(df)  # or None if you want

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(
        plot_chart(df, future=future),
        use_container_width=True
    )

with col2:
    st.metric("Signal", signal)
    st.metric("Confidence", round(confidence, 2))
    st.metric("Stop Loss", sl)
    st.metric("Take Profit", tp)

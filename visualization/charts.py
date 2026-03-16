import plotly.graph_objects as go

def plot_chart(df, future):
    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )

    fig.add_scatter(
        x=list(range(len(df), len(df)+len(future))),
        y=future,
        mode="lines",
        name="Prediction"
    )

    fig.update_layout(height=600)
    return fig

import pandas as pd
import matplotlib.pyplot as plt


def plot_chart(df: pd.DataFrame):
    df.index = pd.to_datetime(df["Date"])

    # Buy Sell decisions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"], label="Equity Curve", color="green")
    plt.scatter(
        df.index[df["signal"] == "BUY"],
        df["Close"][df["signal"] == "BUY"] + 95000,
        color="blue",
        marker="^",
        label="BUY",
    )
    plt.scatter(
        df.index[df["signal"] == "SELL"],
        df["Close"][df["signal"] == "SELL"] + 95000,
        color="red",
        marker="v",
        label="SELL",
    )
    plt.title("Equity Curve with Trades")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.show()

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
ticker = (
    "CSPX.L"  # Yahoo ticker for CSPX on LSE (change if you prefer CSPX, CSPX.AS etc)
)
start = "2021-01-01"
end = "2024-12-31"
annual_investment = 400 * 12  # total money invested per year (adjust as you like)
# -------------------------


# helper: week_of_month (Mon-start)
def week_of_month(dt: pd.Timestamp):
    first = dt.replace(day=1)
    return int(np.ceil((dt.day + first.weekday()) / 7.0))


# XIRR: Newton method
def xirr(cashflows_dates_amounts, guess=0.1):
    # cashflows_dates_amounts: list of (date, amount) where negative = contribution, positive = final value
    dates = np.array(
        [(d - cashflows_dates_amounts[0][0]).days for d, _ in cashflows_dates_amounts],
        dtype=float,
    )
    amounts = np.array([a for _, a in cashflows_dates_amounts], dtype=float)

    def npv(rate):
        return np.sum(amounts / ((1 + rate) ** (dates / 365.0)))

    # derivative approx
    rate = guess
    for _ in range(100):
        f = npv(rate)
        # numeric derivative
        eps = 1e-6
        f1 = npv(rate + eps)
        df = (f1 - f) / eps
        if df == 0:
            break
        new_rate = rate - f / df
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate
    return rate


# -------------------------
# Download
# -------------------------
df = yf.download(
    ticker, start=start, end=end, progress=False, auto_adjust=True, group_by="ticker"
)
df = df.sort_index()
# prices = df.copy()  # series indexed by trading date
prices = df[ticker]["Low"]

# -------------------------
# Build buy-date lists
# -------------------------
all_dates = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="D")

# Strategy A: 8th and 22nd each month => per-buy amount = annual_investment / 24
per_buy_A = annual_investment / 24.0
buy_dates_A = []
for y in range(prices.index.min().year, prices.index.max().year + 1):
    for m in range(1, 13):
        for day in (8, 22):
            try:
                candidate = pd.Timestamp(year=y, month=m, day=day)
            except ValueError:
                continue
            # only within range
            if candidate < prices.index.min() or candidate > prices.index.max():
                continue
            # if not trading day, forward roll to next trading day
            if candidate not in prices.index:
                # find next trading day >= candidate
                nxt = prices.index[prices.index >= candidate]
                if len(nxt) == 0:
                    continue
                candidate = nxt[0]
            buy_dates_A.append(candidate)
buy_dates_A = sorted(set(buy_dates_A))

# Strategy B: quarterly on Monday of 2nd week of Jan,Apr,Jul,Oct => per-buy amount = annual_investment / 4 (per quarter)
per_buy_B = annual_investment / 4.0
quarter_months = [1, 4, 7, 10]
buy_dates_B = []
for y in range(prices.index.min().year, prices.index.max().year + 1):
    for m in quarter_months:
        # find dates in that month
        month_rng = pd.date_range(
            start=pd.Timestamp(y, m, 1),
            end=pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0),
        )
        # find the Monday in week_of_month==2
        found = None
        for d in month_rng:
            if week_of_month(d) == 2 and d.weekday() == 0:
                found = d
                break
        if found is None:
            # fallback: find the first Monday in the month and add 7 days
            mondays = [d for d in month_rng if d.weekday() == 0]
            if len(mondays) >= 1:
                candidate = mondays[0] + pd.Timedelta(days=7)
                if candidate.month == m:
                    found = candidate
        if found is None:
            continue
        # forward-roll to trading day if needed
        if found not in prices.index:
            nxt = prices.index[prices.index >= found]
            if len(nxt) == 0:
                continue
            found = nxt[0]
        if found < prices.index.min() or found > prices.index.max():
            continue
        buy_dates_B.append(found)
buy_dates_B = sorted(set(buy_dates_B))

# Adjust per-buy amounts so total invested period-by-period matches:
# We'll simulate over the full period and let per_buy amounts remain as above (annual_investment parity).
# Note: last partial year might have fewer buys; that's fine.


# -------------------------
# Simulation function
# -------------------------
def simulate(buy_dates, per_buy_amount, prices):
    shares = 0.0
    cashflows = []  # (date, amount) negative for buys
    timeline = (
        []
    )  # list of (date, cumulative_shares, cumulative_invested, portfolio_value)
    invested = 0.0
    for d in sorted(buy_dates):
        price = prices.loc[d]
        qty = per_buy_amount / price
        shares += qty
        invested += per_buy_amount
        cashflows.append((d.to_pydatetime(), -per_buy_amount))
        val = shares * prices.loc[d]
        timeline.append((d, shares, invested, val))
    # final valuation on last available price
    final_date = prices.index.max()
    final_value = shares * prices.loc[final_date]
    cashflows.append((final_date.to_pydatetime(), final_value))
    # compute XIRR
    irr = xirr(cashflows)
    # create dataframe timeline
    timeline_df = pd.DataFrame(
        timeline, columns=["date", "shares", "invested", "value"]
    ).set_index("date")
    return {
        "shares": shares,
        "invested": invested,
        "final_value": final_value,
        "absolute_return": (
            (final_value - invested) / invested if invested > 0 else np.nan
        ),
        "xirr": irr,
        "timeline": timeline_df,
        "cashflows": cashflows,
    }


res_A = simulate(buy_dates_A, per_buy_A, prices)
res_B = simulate(buy_dates_B, per_buy_B, prices)


# -------------------------
# Print results
# -------------------------
def print_summary(name, res):
    print(f"--- {name} ---")
    print(f"Total buys: {len(res['timeline'])}")
    print(f"Total invested: ${res['invested']:.2f}")
    print(
        f"Final portfolio value (on {prices.index.max().date()}): ${res['final_value']:.2f}"
    )
    print("Earnings: ${:.2f}".format(res["final_value"] - res["invested"]))
    print(f"Absolute return: {res['absolute_return']*100:.2f}%")
    if not np.isnan(res["xirr"]):
        print(f"XIRR (annualised): {res['xirr']*100:.2f}%")
    else:
        print("XIRR: n/a")
    print()


print_summary("Bi-weekly (8th & 22nd)", res_A)
print_summary("Quarterly (Mon of 2nd week in Jan/Apr/Jul/Oct)", res_B)

# Create a unified date range for plotting
all_dates = pd.date_range(start=prices.index.min(), end=prices.index.max(), freq="D")

# Use reindex to align the data and ffill to forward-fill values
timeline_A = res_A["timeline"].reindex(all_dates, method="ffill").fillna(0)
timeline_B = res_B["timeline"].reindex(all_dates, method="ffill").fillna(0)

# -------------------------
# Quick plot
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(timeline_A.index, timeline_A["value"], label="Bi-weekly portfolio value")
plt.plot(timeline_B.index, timeline_B["value"], label="Quarterly portfolio value")
plt.plot(
    timeline_A.index,
    timeline_A["invested"],
    "--",
    alpha=0.4,
    label="Bi-weekly invested",
)
plt.plot(
    timeline_B.index,
    timeline_B["invested"],
    "--",
    alpha=0.4,
    label="Quarterly invested",
)
plt.legend()
plt.title(f"Portfolio value over time - {ticker}")
plt.ylabel("USD")
plt.tight_layout()
plt.show()

import datetime


def print_signals_multi_tf(results: list[dict], scores_only: bool = False):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"TA Multi-Timeframe Signal Report — generated {now}\n")
    for r in results:
        c = r["combined"]
        d = r["daily"]
        w = r["weekly"]
        print(
            f"{c.short_name} [{c.ticker}]: {c.signal}  (Daily: {d.signal}, Weekly: {w.signal})"
        )
        print(f"  Last close: {c.last_close}, ATR: {c.atr}")
        print(
            f"  Suggested entry range (from weekly): {c.entry_range[0]:.2f} — {c.entry_range[1]:.2f}"
        )
        if not scores_only:
            print("  Daily reasons:")
            for reason in d.reasons:
                print(f"    - {reason}")
            print("  Weekly reasons:")
            for reason in w.reasons:
                print(f"    - {reason}")
        print("")

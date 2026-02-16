import datetime


def print_signals_multi_tf(results: list[dict], scores_only: bool = False):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"TA Multi-Timeframe Signal Report — generated {now}\n")
    for r in results:
        c = r["combined"]
        d = r["daily"]
        w = r["weekly"]
        ta_signal = r.get("trading_agents")
        
        # Display technical analysis signal
        print(
            f"{c.short_name} [{c.ticker}]: {c.signal}  (Daily: {d.signal}, Weekly: {w.signal})"
        )
        print(f"  Last close: {c.last_close}, ATR: {c.atr}")
        print(
            f"  Suggested entry range (from weekly): {c.entry_range[0]:.2f} — {c.entry_range[1]:.2f}"
        )
        
        # Display TradingAgents LLM signal if available
        if ta_signal is not None:
            # Check for agreement/disagreement
            agreement = ""
            if ta_signal.signal == c.signal:
                agreement = " ✓ AGREES"
            elif (ta_signal.signal == "BUY" and c.signal == "SELL") or (ta_signal.signal == "SELL" and c.signal == "BUY"):
                agreement = " ⚠ CONFLICT"
            else:
                agreement = " ⚡ DIFFERENT"
            
            print(f"  TradingAgents LLM: {ta_signal.signal}{agreement}")
        
        if not scores_only:
            print("  Daily reasons:")
            for reason in d.reasons:
                print(f"    - {reason}")
            print("  Weekly reasons:")
            for reason in w.reasons:
                print(f"    - {reason}")
            
            # Display TradingAgents reasons if available
            if ta_signal is not None and not scores_only:
                print("  TradingAgents LLM reasons:")
                for reason in ta_signal.reasons[:5]:  # Show first 5 reasons
                    print(f"    - {reason}")
        print("")

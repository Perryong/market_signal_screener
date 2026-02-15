from dataclasses import dataclass


@dataclass
class Signal:
    ticker: str
    signal: str  # BUY / HOLD / SELL
    reasons: list[str]
    entry_range: tuple[float, float]  # (low, high)
    last_close: float
    atr: float
    short_name: str = ""

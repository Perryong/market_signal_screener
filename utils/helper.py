import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def last_valid(series: pd.Series) -> float:
    # Get last non-NaN value
    if series.dropna().empty:
        raise ValueError(f"Series '{series.name}' has no valid values.")
    return float(series.dropna().iat[-1])

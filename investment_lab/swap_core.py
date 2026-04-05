"""Core helper utilities for variance/gamma/skew/M4 swap tooling."""

import numpy as np
import pandas as pd

from investment_lab.option_selection import select_closest_maturity
from investment_lab.option_trade import VarianceSwap

_compute_strike_widths = VarianceSwap._compute_strike_widths

def _ensure_ticker(df_options: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add a synthetic ticker when the source contains a single underlier only."""
    df = df_options.copy()
    ticker_was_missing = "ticker" not in df.columns
    if ticker_was_missing:
        df["ticker"] = "SPY"
    return df, ticker_was_missing


def _select_target_maturity_slice(
    df_options: pd.DataFrame,
    dte_lo: int,
    dte_hi: int,
) -> tuple[pd.DataFrame, bool]:
    """
    Keep one expiry slice per date/ticker, chosen as the closest maturity to the
    midpoint of the requested DTE window.
    """
    df, ticker_was_missing = _ensure_ticker(df_options)
    dte_target = int(round((dte_lo + dte_hi) / 2))
    df_filtered = df[df["day_to_expiration"].between(dte_lo, dte_hi)].copy()
    if df_filtered.empty:
        return df_filtered, ticker_was_missing
    return select_closest_maturity(df_filtered, day_to_expiry_target=dte_target), ticker_was_missing


def _finalize_grouped_output(df_result: pd.DataFrame, ticker_was_missing: bool) -> pd.DataFrame:
    if ticker_was_missing and "ticker" in df_result.columns:
        return df_result.drop(columns=["ticker"])
    return df_result


def _finalize_grouped_series(df_result: pd.DataFrame, value_col: str, ticker_was_missing: bool) -> pd.Series:
    df_result = _finalize_grouped_output(df_result, ticker_was_missing)
    index_cols = ["date"] if "ticker" not in df_result.columns else ["date", "ticker"]
    return df_result.set_index(index_cols)[value_col]


def _infer_forward_from_parity(grp: pd.DataFrame, r: float, ttm: float) -> float:
    """
    Infer the forward from call-put parity around the ATM strike when possible.
    Fallback to spot growth when no common strikes are available.
    """
    cp = (
        grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
        .dropna(subset=["C", "P"], how="any")
        .reset_index()
    )
    spot = float(grp["spot"].iloc[0])
    if cp.empty:
        return spot * np.exp(r * ttm)

    cp["parity_gap"] = (cp["C"] - cp["P"]).abs()
    parity_row = cp.sort_values(["parity_gap", "strike"]).iloc[0]
    return float(parity_row["strike"] + np.exp(r * ttm) * (parity_row["C"] - parity_row["P"]))




"""Time-series computations for variance-family swap analytics."""

import numpy as np
import pandas as pd

from investment_lab.constants import TRADING_DAYS_PER_YEAR
from investment_lab.option_selection import select_closest_maturity
from investment_lab.swap_core import (
    _compute_strike_widths,
    _finalize_grouped_output,
    _finalize_grouped_series,
    _infer_forward_from_parity,
    _select_target_maturity_slice,
)

def compute_realized_variance_series(
    returns: pd.Series,
    window: int = 21,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling realized variance using the zero-mean variance swap convention:

        var(t) = (ann_factor / window) * sum(r_i^2)

    Differs from metrics.volatility.rolling_realized_volatility which uses std
    (mean-subtracted). For variance swaps, drift is excluded by market convention.

    Args:
        returns: Daily log-return series.
        window: Rolling window in trading days.
        ann_factor: Annualization factor (default 252).

    Returns:
        Rolling annualized realized variance series.
    """
    return returns.rolling(window).apply(
        lambda x: np.sum(x**2) * ann_factor / len(x), raw=True
    )


def get_atm_iv_series(
    df_options: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
) -> pd.DataFrame:
    """
    Extract ATM implied volatility per date from a raw options DataFrame.

    Selects the call option with moneyness closest to 1.0 within the DTE range.

    Args:
        df_options: Options DataFrame with columns: date, day_to_expiration,
                    implied_volatility, moneyness, call_put.
        dte_lo: Lower bound on days to expiration (inclusive).
        dte_hi: Upper bound on days to expiration (inclusive).

    Returns:
        DataFrame with columns: date, atm_iv, atm_var.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 3.0)
    ].copy()

    def _pick_atm(grp: pd.DataFrame) -> pd.Series:
        calls = grp[grp["call_put"] == "C"]
        if calls.empty:
            return pd.Series({"atm_iv": np.nan})
        idx = (calls["moneyness"] - 1.0).abs().idxmin()
        return pd.Series({"atm_iv": calls.loc[idx, "implied_volatility"]})

    group_cols = ["date", "ticker"]
    df_atm = df_filtered.groupby(group_cols).apply(_pick_atm, include_groups=False).reset_index()
    df_atm["atm_var"] = df_atm["atm_iv"] ** 2
    return _finalize_grouped_output(df_atm, ticker_was_missing)


def get_iv_at_moneyness(
    df_options: pd.DataFrame,
    moneyness_target: float,
    dte_lo: int = 21,
    dte_hi: int = 35,
    tol: float = 0.03,
    call_or_put: str = "P",
) -> pd.Series:
    """
    Extract implied volatility at a target moneyness level per date.

    Args:
        df_options: Options DataFrame.
        moneyness_target: Target moneyness (e.g. 0.90 for 90% put).
        dte_lo: Lower bound on days to expiration (inclusive).
        dte_hi: Upper bound on days to expiration (inclusive).
        tol: Tolerance band around moneyness_target.
        call_or_put: "P" for puts, "C" for calls.

    Returns:
        Series indexed by date with IV at the target moneyness.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 3.0)
    ].copy()

    def _pick_iv(grp: pd.DataFrame) -> float:
        sub = grp[
            (grp["call_put"] == call_or_put)
            & grp["moneyness"].between(moneyness_target - tol, moneyness_target + tol)
        ]
        if sub.empty:
            return np.nan
        return float(
            sub.loc[(sub["moneyness"] - moneyness_target).abs().idxmin(), "implied_volatility"]
        )

    group_cols = ["date", "ticker"]
    df_iv = df_filtered.groupby(group_cols).apply(_pick_iv, include_groups=False).reset_index(name="iv_target")
    return _finalize_grouped_series(df_iv, "iv_target", ticker_was_missing)


def compute_mfiv_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute Model-Free Implied Variance (MFIV) per date.

    Implements the Carr-Madan static replication formula:

        var_MFIV = (2 * e^(rT) / T) * [integral_0^F P(K)/K^2 dK + integral_F^inf C(K)/K^2 dK]

    Discretized with centered half-interval weights dK_i / K_i^2 (see _compute_strike_widths).

    Args:
        df_options: Raw options DataFrame (unfiltered).
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo / dte_hi: DTE filter bounds.
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price to exclude illiquid options.

    Returns:
        DataFrame with columns: date, mfiv, mfiv_vol.
    """
    df_slice, ticker_was_missing = _select_target_maturity_slice(df_options, dte_lo=dte_lo, dte_hi=dte_hi)
    df_filtered = df_slice[
        df_slice["implied_volatility"].notna()
        & df_slice["implied_volatility"].between(0.02, 2.5)
        & (df_slice["mid"] > min_mid)
    ].copy()
    df_filtered = df_filtered.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_filtered[rate_col] = df_filtered[rate_col].fillna(0.02)

    def _mfiv_one_date(grp: pd.DataFrame) -> float:
        # day_to_expiration is in calendar days; use 365 to match the annualization
        # convention used in the implied_volatility column (Black-Scholes with T=cal/365)
        T = grp["day_to_expiration"].iloc[0] / 365
        r = grp[rate_col].iloc[0]
        F = _infer_forward_from_parity(grp, r=r, ttm=T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty or not {"C", "P"}.intersection(chain.columns):
            return np.nan

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        call_mid = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        put_mid = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < F,
            put_mid.to_numpy(dtype=float),
            np.where(
                strikes > F,
                call_mid.to_numpy(dtype=float),
                0.5 * np.nan_to_num(call_mid.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(put_mid.to_numpy(dtype=float), nan=0.0),
            ),
        )
        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return np.nan

        strike_term = np.sum(dK[valid] * otm_mid[valid] / strikes[valid] ** 2)
        return max(2.0 * np.exp(r * T) / T * strike_term, 0.0)

    group_cols = ["date", "ticker"]
    mfiv = (
        df_filtered.groupby(group_cols)
        .apply(_mfiv_one_date, include_groups=False)
        .reset_index(name="mfiv")
    )
    mfiv["mfiv_vol"] = np.sqrt(mfiv["mfiv"].clip(lower=0))
    mfiv = mfiv.dropna(subset=["mfiv_vol"])
    mfiv = mfiv[mfiv["mfiv_vol"].between(0.02, 2.0)]
    return _finalize_grouped_output(mfiv.reset_index(drop=True), ticker_was_missing)

def compute_gamma_swap_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute the gamma swap implied strike (annualized) per date.

    The gamma swap kernel replaces 1/K^2 with 1/K, reducing the weight on
    deep OTM puts compared to the variance swap.

        var_gamma = (2 * e^(rT) / (T * S_0)) * [integral P(K)/K dK + integral C(K)/K dK]

    Args:
        df_options: Raw SPY options DataFrame.
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo: Lower DTE bound (inclusive).
        dte_hi: Upper DTE bound (inclusive).
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price filter.

    Returns:
        DataFrame with columns: date, gamma_var, gamma_vol.
    """
    dte_target = int(round((dte_lo + dte_hi) / 2))

    df_slice = df_options[
        df_options["day_to_expiration"].between(dte_lo, dte_hi)
        & df_options["implied_volatility"].notna()
        & df_options["implied_volatility"].between(0.02, 2.5)
        & df_options["mid"].gt(min_mid)
    ].copy()

    if "ticker" not in df_slice.columns:
        df_slice["ticker"] = "SPY"

    df_slice = select_closest_maturity(df_slice, day_to_expiry_target=dte_target)
    df_slice = df_slice.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_slice[rate_col] = df_slice[rate_col].fillna(0.02)

    def _gamma_one_group(grp: pd.DataFrame) -> float:
        T = grp["day_to_expiration"].iloc[0] / 365
        r = grp[rate_col].iloc[0]
        s0 = grp["spot"].iloc[0]
        fwd = _infer_forward_from_parity(grp, r=r, ttm=T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty:
            return np.nan

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        calls = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        puts = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < fwd,
            puts.to_numpy(dtype=float),
            np.where(
                strikes > fwd,
                calls.to_numpy(dtype=float),
                0.5 * np.nan_to_num(calls.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(puts.to_numpy(dtype=float), nan=0.0),
            ),
        )

        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return np.nan

        integral = np.sum(dK[valid] * otm_mid[valid] / strikes[valid])
        return max(2.0 * np.exp(r * T) / (T * s0) * integral, 0.0)

    gamma_swap = (
        df_slice.groupby(["date", "ticker"])
        .apply(_gamma_one_group, include_groups=False)
        .reset_index(name="gamma_var")
    )
    gamma_swap["gamma_vol"] = np.sqrt(gamma_swap["gamma_var"].clip(lower=0))
    gamma_swap = gamma_swap[gamma_swap["gamma_vol"].between(0.02, 2.0)].reset_index(drop=True)
    return _finalize_grouped_output(gamma_swap, "ticker" not in df_options.columns)

def compute_m4_cm_series(
    df_options: pd.DataFrame,
    df_rates: pd.DataFrame,
    dte_lo: int = 21,
    dte_hi: int = 35,
    rate_col: str = "1 Mo",
    min_mid: float = 0.01,
) -> pd.DataFrame:
    """
    Compute Carr-Madan M4 contract family metrics (M2_CM, M4_CM, kappa_CM) per date.

    Both moments are derived from the spanning formula applied to simple
    returns R = S_T/F - 1, ensuring internal consistency:

        f(S_T) = (S_T/F - 1)^n  ->  f''(K) = n(n-1)(K/F - 1)^(n-2) / F^2

    Concretely:

        M2_CM    = e^(rT) * (2/F^2) * integral Q(K) dK        [kernel: 2/F^2]
        M4_CM    = e^(rT) * integral 12(K-F)^2/F^4 * Q(K) dK [kernel: 12(K-F)^2/F^4]
        kappa_CM = M4_CM / M2_CM^2


    Args:
        df_options: Raw SPY options DataFrame.
        df_rates: Rates DataFrame with columns: date, <rate_col>.
        dte_lo: Lower DTE bound (inclusive).
        dte_hi: Upper DTE bound (inclusive).
        rate_col: Risk-free rate column in df_rates.
        min_mid: Minimum mid-price filter.

    Returns:
        DataFrame with columns: date, M2_CM, M4_CM, kappa_CM.
    """
    dte_target = int(round((dte_lo + dte_hi) / 2))

    df_slice = df_options[
        df_options["day_to_expiration"].between(dte_lo, dte_hi)
        & df_options["implied_volatility"].notna()
        & df_options["implied_volatility"].between(0.02, 2.5)
        & df_options["mid"].gt(min_mid)
    ].copy()

    if "ticker" not in df_slice.columns:
        df_slice["ticker"] = "SPY"

    df_slice = select_closest_maturity(df_slice, day_to_expiry_target=dte_target)
    df_slice = df_slice.merge(df_rates[["date", rate_col]], on="date", how="left")
    df_slice[rate_col] = df_slice[rate_col].fillna(0.02)

    def _cm_one_group(grp: pd.DataFrame) -> pd.Series:
        T = grp["day_to_expiration"].iloc[0] / 365.0
        r = grp[rate_col].iloc[0]
        fwd = _infer_forward_from_parity(grp, r=r, ttm=T)
        ert = np.exp(r * T)

        chain = (
            grp.pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first")
            .sort_index()
        )
        if chain.empty:
            return pd.Series({"M2_CM": np.nan, "M4_CM": np.nan, "kappa_CM": np.nan})

        strikes = chain.index.to_numpy(dtype=float)
        dK = _compute_strike_widths(strikes)
        calls = chain["C"] if "C" in chain.columns else pd.Series(np.nan, index=chain.index)
        puts = chain["P"] if "P" in chain.columns else pd.Series(np.nan, index=chain.index)

        otm_mid = np.where(
            strikes < fwd,
            puts.to_numpy(dtype=float),
            np.where(
                strikes > fwd,
                calls.to_numpy(dtype=float),
                0.5 * np.nan_to_num(calls.to_numpy(dtype=float), nan=0.0)
                + 0.5 * np.nan_to_num(puts.to_numpy(dtype=float), nan=0.0),
            ),
        )
        valid = np.isfinite(otm_mid) & (otm_mid > 0)
        if valid.sum() < 3:
            return pd.Series({"M2_CM": np.nan, "M4_CM": np.nan, "kappa_CM": np.nan})

        K = strikes[valid]
        w = dK[valid]
        Q = otm_mid[valid]

        # M2_CM = e^(rT) * (2/F^2) * integral Q(K) dK
        M2_CM = ert * (2.0 / fwd ** 2) * float(np.sum(w * Q))

        # M4_CM = e^(rT) * integral [12(K-F)^2/F^4] * Q(K) dK
        M4_CM = ert * float(np.sum(12.0 * (K - fwd) ** 2 / fwd ** 4 * w * Q))

        # kappa_CM = M4_CM / M2_CM^2  (= 3 for Gaussian)
        kappa_CM = float(M4_CM / M2_CM ** 2) if M2_CM > 1e-12 else np.nan

        return pd.Series({"M2_CM": M2_CM, "M4_CM": M4_CM, "kappa_CM": kappa_CM})

    result = (
        df_slice.groupby(["date", "ticker"])
        .apply(_cm_one_group, include_groups=False)
        .reset_index()
    )
    result = result.dropna(subset=["kappa_CM"]).reset_index(drop=True)
    return _finalize_grouped_output(result, "ticker" not in df_options.columns)


class SwapSeriesCalculator:
    """OO namespace for all swap-related time series computations."""

    @staticmethod
    def compute_realized_variance_series(
        returns: pd.Series,
        window: int = 21,
        ann_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> pd.Series:
        return compute_realized_variance_series(returns, window=window, ann_factor=ann_factor)

    @staticmethod
    def get_atm_iv_series(
        df_options: pd.DataFrame,
        dte_lo: int = 21,
        dte_hi: int = 35,
    ) -> pd.DataFrame:
        return get_atm_iv_series(df_options=df_options, dte_lo=dte_lo, dte_hi=dte_hi)

    @staticmethod
    def get_iv_at_moneyness(
        df_options: pd.DataFrame,
        moneyness_target: float,
        dte_lo: int = 21,
        dte_hi: int = 35,
        tol: float = 0.03,
        call_or_put: str = "P",
    ) -> pd.Series:
        return get_iv_at_moneyness(
            df_options=df_options,
            moneyness_target=moneyness_target,
            dte_lo=dte_lo,
            dte_hi=dte_hi,
            tol=tol,
            call_or_put=call_or_put,
        )

    @staticmethod
    def compute_mfiv_series(
        df_options: pd.DataFrame,
        df_rates: pd.DataFrame,
        dte_lo: int = 21,
        dte_hi: int = 35,
        rate_col: str = "1 Mo",
        min_mid: float = 0.01,
    ) -> pd.DataFrame:
        return compute_mfiv_series(
            df_options=df_options,
            df_rates=df_rates,
            dte_lo=dte_lo,
            dte_hi=dte_hi,
            rate_col=rate_col,
            min_mid=min_mid,
        )

    @staticmethod
    def compute_gamma_swap_series(
        df_options: pd.DataFrame,
        df_rates: pd.DataFrame,
        dte_lo: int = 21,
        dte_hi: int = 35,
        rate_col: str = "1 Mo",
        min_mid: float = 0.01,
    ) -> pd.DataFrame:
        return compute_gamma_swap_series(
            df_options=df_options,
            df_rates=df_rates,
            dte_lo=dte_lo,
            dte_hi=dte_hi,
            rate_col=rate_col,
            min_mid=min_mid,
        )

    @staticmethod
    def compute_m4_cm_series(
        df_options: pd.DataFrame,
        df_rates: pd.DataFrame,
        dte_lo: int = 21,
        dte_hi: int = 35,
        rate_col: str = "1 Mo",
        min_mid: float = 0.01,
    ) -> pd.DataFrame:
        return compute_m4_cm_series(
            df_options=df_options,
            df_rates=df_rates,
            dte_lo=dte_lo,
            dte_hi=dte_hi,
            rate_col=rate_col,
            min_mid=min_mid,
        )



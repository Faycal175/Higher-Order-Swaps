"""Kernel diagnostics plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from investment_lab.swap_core import _compute_strike_widths

def get_kernel_data(
    df_spy: pd.DataFrame,
    df_rates: pd.DataFrame,
    date,
) -> "dict | None":
    """Return puts/calls, kernel weights and contributions for a given date.

    Filters the option chain to 21-35 DTE near-the-money options, splits into
    puts (K < F) and calls (K >= F), and returns centred strike widths and the
    variance-kernel weights DeltaK/K^2.

    Returns ``None`` if the chain is empty after filtering.
    """
    samp = df_spy[
        df_spy["date"].eq(date)
        & df_spy["day_to_expiration"].between(21, 35)
        & df_spy["implied_volatility"].notna()
        & df_spy["mid"].gt(0.10)
    ].copy().merge(df_rates[["date", "1 Mo"]], on="date", how="left")
    if samp.empty:
        return None
    T = samp["day_to_expiration"].mean() / 365
    r = samp["1 Mo"].fillna(0.02).iloc[0]
    F = samp["spot"].iloc[0] * np.exp(r * T)
    puts  = samp[(samp["call_put"] == "P") & (samp["strike"] < F)].sort_values("strike")
    calls = samp[(samp["call_put"] == "C") & (samp["strike"] >= F)].sort_values("strike")
    dK_p = _compute_strike_widths(puts["strike"].to_numpy(dtype=float))
    dK_c = _compute_strike_widths(calls["strike"].to_numpy(dtype=float))
    w_p = dK_p / puts["strike"].to_numpy(dtype=float) ** 2
    w_c = dK_c / calls["strike"].to_numpy(dtype=float) ** 2
    return dict(puts=puts, calls=calls, dK_p=dK_p, dK_c=dK_c, w_p=w_p, w_c=w_c, F=F)


def plot_kernel_bars(
    ax: "plt.Axes",
    st: "np.ndarray",
    weights: "np.ndarray",
    dK: "np.ndarray",
    put_mask: "np.ndarray",
    fwd: float,
    title: str,
) -> None:
    """Bar chart of kernel weights - shared by variance, gamma, skew and M4 cells.

    Puts are drawn in blue, calls in red, y-axis scaled by x1e6.
    """
    call_mask = ~put_mask
    ax.bar(st[put_mask],  weights[put_mask]  * 1e6, width=dK[put_mask],  alpha=0.75, color="tab:blue", label="Puts OTM")
    ax.bar(st[call_mask], weights[call_mask] * 1e6, width=dK[call_mask], alpha=0.75, color="tab:red",  label="Calls OTM")
    ax.axvline(fwd, color="black", ls=":", lw=1.2, label=f"F={fwd:.0f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel(r"Poids ($\times 10^6$)")
    ax.legend(fontsize=8)


def plot_kernel_contrib(
    ax: "plt.Axes",
    st: "np.ndarray",
    weights: "np.ndarray",
    mid: "np.ndarray",
    dK: "np.ndarray",
    put_mask: "np.ndarray",
    fwd: float,
    title: str,
) -> None:
    """Bar chart of price x weight contribution - same visual convention as ``plot_kernel_bars``."""
    call_mask = ~put_mask
    contrib = weights * mid
    ax.bar(st[put_mask],  contrib[put_mask]  * 1e3, width=dK[put_mask],  alpha=0.75, color="tab:blue", label="Puts OTM")
    ax.bar(st[call_mask], contrib[call_mask] * 1e3, width=dK[call_mask], alpha=0.75, color="tab:red",  label="Calls OTM")
    ax.axvline(fwd, color="black", ls=":", lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel(r"Contribution ($\times 10^3$)")
    ax.legend(fontsize=8)


def plot_kernel(ax_w: "plt.Axes", ax_c: "plt.Axes", kd: dict, title_suffix: str) -> None:
    """Variance kernel panel: weight bars (ax_w) + contribution bars (ax_c).

    ``kd`` is the dict returned by :func:`get_kernel_data`.
    """
    st  = np.concatenate([kd["puts"]["strike"].to_numpy(dtype=float),
                          kd["calls"]["strike"].to_numpy(dtype=float)])
    w   = np.concatenate([kd["w_p"], kd["w_c"]])
    dk  = np.concatenate([kd["dK_p"], kd["dK_c"]])
    mid = np.concatenate([kd["puts"]["mid"].to_numpy(dtype=float),
                          kd["calls"]["mid"].to_numpy(dtype=float)])
    pm  = np.array([True] * len(kd["puts"]) + [False] * len(kd["calls"]), dtype=bool)
    plot_kernel_bars(ax_w, st, w, dk, pm, kd["F"],
                     f"Poids \u0394K/K\u00b2 \u2014 {title_suffix}")
    plot_kernel_contrib(ax_c, st, w, mid, dk, pm, kd["F"],
                        f"Contribution prix \u00d7 poids \u2014 {title_suffix}")




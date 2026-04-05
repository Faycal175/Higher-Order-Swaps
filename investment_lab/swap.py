"""Public API for variance-family swap utilities.

This module is a compatibility facade that re-exports functions/classes from
smaller thematic modules:
- swap_core.py
- swap_series.py
- swap_positions.py
- swap_backtester.py
- swap_plotting.py
"""

from investment_lab.swap_backtester import PreloadedBacktester
from investment_lab.swap_core import (
    _compute_strike_widths,
    _ensure_ticker,
    _finalize_grouped_output,
    _finalize_grouped_series,
    _infer_forward_from_parity,
    _select_target_maturity_slice,
)
from investment_lab.swap_plotting import (
    get_kernel_data,
    plot_kernel,
    plot_kernel_bars,
    plot_kernel_contrib,
)
from investment_lab.swap_positions import (
    GammaSwapBuilder,
    KernelSwapBuilder,
    M4SwapBuilder,
    SkewSwapBuilder,
    VarianceSwapBuilder,
)
from investment_lab.swap_series import (
    SwapSeriesCalculator,
    TRADING_DAYS_PER_YEAR,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PreloadedBacktester",
    "_compute_strike_widths",
    "_ensure_ticker",
    "_finalize_grouped_output",
    "_finalize_grouped_series",
    "_infer_forward_from_parity",
    "_select_target_maturity_slice",
    "GammaSwapBuilder",
    "KernelSwapBuilder",
    "M4SwapBuilder",
    "SkewSwapBuilder",
    "VarianceSwapBuilder",
    "SwapSeriesCalculator",
    "get_kernel_data",
    "plot_kernel",
    "plot_kernel_bars",
    "plot_kernel_contrib",
]


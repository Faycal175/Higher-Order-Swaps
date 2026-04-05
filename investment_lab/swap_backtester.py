"""Backtester adapters for preloaded option universes."""

import pandas as pd

from investment_lab.backtest import StrategyBacktester
from investment_lab.util import ffill_options_data

class PreloadedBacktester(StrategyBacktester):
    """
    StrategyBacktester variant that uses pre-loaded option data
    instead of reloading from the parquet database.

    Required for SPYOptionLoader data (spy_2020_2022.parquet), which lacks
    the 'ticker' column expected by StrategyBacktester._preprocess_positions().

    Usage:
        positions = VarianceSwapBuilder.build_positions(df_spy)
        bt = PreloadedBacktester(positions, df_spy)
        bt.compute_backtest()
        bt.nav   # NAV series
        bt.pnl   # P&L decomposition (delta, gamma, theta, vega)
    """

    def __init__(self, df_positions: pd.DataFrame, df_options: pd.DataFrame) -> None:
        super().__init__(df_positions)
        self._df_options = df_options.copy()
        if "ticker" not in self._df_options.columns:
            self._df_options["ticker"] = "SPY"

    def _preprocess_positions(self, df_positions: pd.DataFrame) -> pd.DataFrame:
        """Override: merge with pre-loaded data instead of reloading from file."""
        df_options = self._df_options

        # Spot rows - mirrors StrategyBacktester._preprocess_positions logic
        df_spot = (
            df_options.groupby(["date", "ticker"], as_index=False)
            .agg(spot=("spot", "first"))
            .assign(
                option_id=lambda x: x["ticker"],
                bid=lambda x: x["spot"],
                ask=lambda x: x["spot"],
                mid=lambda x: x["spot"],
                delta=1.0,
            )
        )
        df_options_with_spot = pd.concat([df_options, df_spot], ignore_index=True)

        df_extended = df_positions.merge(
            df_options_with_spot, how="left", on=["ticker", "option_id", "date"]
        )
        df_extended = df_extended[
            (df_extended["date"] <= df_extended["expiration"])
            | df_extended["expiration"].isna()
        ]
        return ffill_options_data(df_extended)




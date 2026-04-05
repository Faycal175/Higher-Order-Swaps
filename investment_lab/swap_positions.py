"""Object-oriented position builders for variance, gamma, skew, and M4 swap replications."""

import numpy as np
import pandas as pd

from investment_lab.dataclass import VarianceSwapLegSpec
from investment_lab.option_selection import select_closest_maturity
from investment_lab.option_trade import VarianceSwap
from investment_lab.swap_core import _compute_strike_widths, _infer_forward_from_parity


class VarianceSwapBuilder:
    """Build positions for the standard variance swap replication."""

    @classmethod
    def build_positions(
        cls,
        df_spy: "pd.DataFrame | None" = None,
        weight: float = -1.0,
        day_to_expiry_target: int = 21,
        rebal_week_day: "list[int] | None" = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Build rolling variance swap positions from pre-loaded options data."""
        # Backward-compat: allow alias df_options
        if df_spy is None:
            df_spy = kwargs.pop("df_options", None)
        else:
            kwargs.pop("df_options", None)

        if kwargs:
            bad = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {bad}")
        if df_spy is None:
            raise TypeError(
                "VarianceSwapBuilder.build_positions requires `df_spy` (or alias `df_options`)."
            )

        if rebal_week_day is None:
            rebal_week_day = [1]

        df = df_spy.copy()
        if "ticker" not in df.columns:
            df["ticker"] = "SPY"

        # Leg configuration used by existing VarianceSwap selection logic
        leg: VarianceSwapLegSpec = {
            "day_to_expiry_target": day_to_expiry_target,
            "strike_spacing": 2.0,
            "weight": weight,
            "rebal_week_day": rebal_week_day,
        }

        trades = VarianceSwap._select_options(df, legs=[leg])
        return VarianceSwap._convert_trades_to_timeseries(trades)


class KernelSwapBuilder:
    """Generic OTM kernel-based builder for replicated swap positions."""

    LEG_NAME = "KERNEL SWAP"

    @classmethod
    def kernel(cls, strikes: np.ndarray, fwd: float) -> np.ndarray:
        """Return kernel values as a function of strike K and forward F."""
        raise NotImplementedError

    @classmethod
    def filter_weighted_side(cls, side_df: pd.DataFrame) -> pd.DataFrame:
        """Optional post-processing after side-level weight computation."""
        return side_df

    @classmethod
    def infer_forward(cls, grp: pd.DataFrame) -> float:
        """Infer forward from put-call parity with short-horizon time-to-maturity."""
        ttm = grp["day_to_expiration"].iloc[0] / 365.0
        return _infer_forward_from_parity(grp, r=0.0, ttm=ttm)

    @classmethod
    def prepare_selected(
        cls,
        df_options: pd.DataFrame,
        day_to_expiry_target: int,
        rebal_week_day: "list[int] | None",
    ) -> pd.DataFrame:
        """Select maturity, keep OTM options, and apply rebalance weekday filter."""
        if rebal_week_day is None:
            rebal_week_day = [1]

        df = df_options.copy()
        if "ticker" not in df.columns:
            df["ticker"] = "SPY"

        selected = select_closest_maturity(df, day_to_expiry_target=day_to_expiry_target)
        selected = selected.loc[
            ((selected["call_put"] == "P") & (selected["moneyness"] <= 1.0))
            | ((selected["call_put"] == "C") & (selected["moneyness"] >= 1.0))
        ].copy()
        selected["leg_name"] = cls.LEG_NAME
        selected = selected.loc[selected["date"].dt.day_of_week.isin(rebal_week_day)]
        return selected

    @classmethod
    def compute_group_weights(cls, grp: pd.DataFrame, strike_spacing: float) -> pd.DataFrame:
        """Compute strike weights for one (date, expiration) group."""
        fwd = cls.infer_forward(grp)
        grp = VarianceSwap._thin_strikes(grp.copy(), strike_spacing=strike_spacing)

        weighted_groups: list[pd.DataFrame] = []
        for call_put in ("P", "C"):
            side_df = grp[grp["call_put"] == call_put].copy().sort_values("strike")
            if side_df.empty:
                continue

            strikes = side_df["strike"].to_numpy(dtype=float)
            widths = _compute_strike_widths(strikes)

            # Generic discretized weight = kernel(K, F) * dK
            side_df["weight"] = cls.kernel(strikes, fwd) * widths
            side_df = cls.filter_weighted_side(side_df)

            if not side_df.empty:
                weighted_groups.append(side_df)

        if not weighted_groups:
            grp["weight"] = 0.0
            return grp

        return pd.concat(weighted_groups, ignore_index=True)

    @classmethod
    def build_positions(
        cls,
        df_options: pd.DataFrame,
        weight: float = -1.0,
        day_to_expiry_target: int = 21,
        rebal_week_day: "list[int] | None" = None,
        strike_spacing: float = 2.0,
    ) -> pd.DataFrame:
        """Build rolling positions using the class kernel."""
        selected = cls.prepare_selected(
            df_options=df_options,
            day_to_expiry_target=day_to_expiry_target,
            rebal_week_day=rebal_week_day,
        )
        if selected.empty:
            return selected.assign(entry_date=pd.NaT).reindex(
                columns=["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]
            )

        weighted = pd.concat(
            [
                cls.compute_group_weights(grp.copy(), strike_spacing=strike_spacing)
                for _, grp in selected.groupby(["date", "expiration"], sort=False)
            ],
            ignore_index=True,
        )

        # Normalize each date group to target notional weight
        normalized = pd.concat(
            [
                VarianceSwap._normalize_strike_weights(grp.copy(), target_weight=weight)
                for _, grp in weighted.groupby("date", sort=False)
            ],
            ignore_index=True,
        )

        trades = normalized.rename(columns={"date": "entry_date"})[
            ["entry_date", "option_id", "expiration", "leg_name", "weight", "ticker"]
        ]
        return VarianceSwap._convert_trades_to_timeseries(trades)


class GammaSwapBuilder(KernelSwapBuilder):
    """Build positions for gamma swap replication using kernel 1/K."""

    LEG_NAME = "GAMMA SWAP"

    @classmethod
    def kernel(cls, strikes: np.ndarray, fwd: float) -> np.ndarray:
        del fwd
        return 1.0 / strikes


class M4SwapBuilder(KernelSwapBuilder):
    """Build positions for M4 swap replication using kernel 12 * (K - F)^2 / F^4."""

    LEG_NAME = "M4 SWAP"

    @classmethod
    def kernel(cls, strikes: np.ndarray, fwd: float) -> np.ndarray:
        return 12.0 * (strikes - fwd) ** 2 / fwd**4

    @classmethod
    def filter_weighted_side(cls, side_df: pd.DataFrame) -> pd.DataFrame:
        # Keep strictly positive practical weights to avoid numerical noise
        return side_df[side_df["weight"] > 1e-14].copy()


class SkewSwapBuilder:
    """Build skew swap proxy positions: 0.5 * (variance_weights - gamma_weights)."""

    @classmethod
    def build_positions(
        cls,
        df_options: pd.DataFrame,
        weight: float = -1.0,
        day_to_expiry_target: int = 21,
        rebal_week_day: "list[int] | None" = None,
        strike_spacing: float = 2.0,
    ) -> pd.DataFrame:
        # Build both legs on the same universe and merge by option/date keys
        var_pos = VarianceSwapBuilder.build_positions(
            df_spy=df_options,
            weight=weight,
            day_to_expiry_target=day_to_expiry_target,
            rebal_week_day=rebal_week_day,
        )
        gamma_pos = GammaSwapBuilder.build_positions(
            df_options=df_options,
            weight=weight,
            day_to_expiry_target=day_to_expiry_target,
            rebal_week_day=rebal_week_day,
            strike_spacing=strike_spacing,
        )

        key_cols = ["date", "entry_date", "ticker", "option_id"]
        skew = (
            var_pos[key_cols + ["weight"]]
            .rename(columns={"weight": "weight_var"})
            .merge(
                gamma_pos[key_cols + ["weight"]].rename(columns={"weight": "weight_gamma"}),
                on=key_cols,
                how="outer",
            )
            .fillna(0.0)
        )

        skew["weight"] = 0.5 * (skew["weight_var"] - skew["weight_gamma"])
        skew = skew[skew["weight"].abs() > 1e-14].copy()
        skew["leg_name"] = "SKEW SWAP"

        return (
            skew[["date", "option_id", "entry_date", "leg_name", "weight", "ticker"]]
            .sort_values(["entry_date", "date", "option_id"])
            .reset_index(drop=True)
        )

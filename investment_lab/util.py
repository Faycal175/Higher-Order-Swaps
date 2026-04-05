import logging
from typing import Optional

import pandas as pd


def check_is_true(condition: bool, message: Optional[str] = None) -> None:
    if not condition:
        raise ValueError(message or "Condition is not true.")


def ffill_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill option data based on dates and id.

    Args:
        df (pd.DataFrame): dataframe of options containing at least
        the columns: date and option_id

    Returns:
        pd.DataFrame: _description_
    """
    missing_cols = set(["option_id", "date"]).difference(df.columns)
    check_is_true(
        len(missing_cols) == 0, f"Data is missing required columns: {missing_cols}"
    )
    logging.info("Forward filling option data for df")
    df_sorted = df.sort_values(by=["option_id", "date"]).copy()
    cols_to_fill = [col for col in df_sorted.columns if col != "option_id"]
    df_sorted[cols_to_fill] = df_sorted.groupby("option_id", group_keys=False)[cols_to_fill].ffill()
    return df_sorted

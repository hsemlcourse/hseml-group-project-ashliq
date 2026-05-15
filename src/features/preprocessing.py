"""Reusable preprocessing helpers for the movies project."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def safe_list(value: Any) -> list[str]:
    """Convert a list-like cell from the raw dataset into a Python list of strings.

    The source dataset contains columns such as genre, companies, countries and languages.
    Depending on export format, values may already be lists, stringified lists, comma-separated
    strings or missing values. This helper normalizes those cases.
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        cleaned_value = value.strip()
        if not cleaned_value:
            return []
        try:
            parsed = ast.literal_eval(cleaned_value)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

        return [item.strip() for item in cleaned_value.split(",") if item.strip()]

    return []


def clip_outliers_by_quantile(
    data: pd.DataFrame,
    columns: list[str],
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Clip selected numeric columns by an upper quantile and return a copy."""
    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must be in the interval (0, 1].")

    result = data.copy()
    for column in columns:
        if column not in result.columns:
            raise KeyError(f"Column '{column}' is absent from the dataframe.")
        upper_value = result[column].quantile(upper_quantile)
        result[column] = result[column].clip(upper=upper_value)
    return result

import pandas as pd
import pytest
from src.features.preprocessing import clip_outliers_by_quantile, safe_list


def test_safe_list_handles_missing_values():
    assert safe_list(None) == []
    assert safe_list("") == []


def test_safe_list_handles_stringified_list():
    assert safe_list("['Drama', 'Comedy']") == ["Drama", "Comedy"]


def test_safe_list_handles_comma_separated_values():
    assert safe_list("Drama, Comedy") == ["Drama", "Comedy"]


def test_clip_outliers_by_quantile_clips_upper_tail():
    data = pd.DataFrame({"value": [1, 2, 3, 100]})
    result = clip_outliers_by_quantile(data, ["value"], upper_quantile=0.75)
    assert result["value"].max() <= data["value"].quantile(0.75)


def test_clip_outliers_by_quantile_raises_for_absent_column():
    data = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(KeyError):
        clip_outliers_by_quantile(data, ["missing"])

import pandas as pd
import pytest

from hello_world.features import dummy, min_max_scale, one_hot, ordinal_encode, standard_scale

EDUCATION_ORDER = ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]


def test_one_hot_creates_one_column_per_category():
    df = pd.DataFrame({"country": ["UZ", "US", "UK", "RU"]})

    result = one_hot(df, "country")

    assert sorted(result.columns) == ["country_RU", "country_UK", "country_US", "country_UZ"]
    assert result.isin([0, 1]).all().all()
    assert (result.sum(axis=1) == 1).all()


def test_dummy_creates_one_column_per_category_minus_first():
    df = pd.DataFrame({"country": ["UZ", "US", "UK", "RU"]})

    result = dummy(df, "country")

    assert sorted(result.columns) == ["country_UK", "country_US", "country_UZ"]
    assert result.isin([0, 1]).all().all()
    assert (result.sum(axis=1) <= 1).all()


def test_ordinal_encode_encodes_correctly():
    df = pd.DataFrame({"level of education": ["high school", "bachelor's degree", "master's degree"]})

    result = ordinal_encode(df, "level of education", EDUCATION_ORDER)

    assert result["level of education"].tolist() == [0, 3, 4]


def test_standard_scale_scales_correctly():
    df = pd.DataFrame({"salary": [5000, 6000, 7000], "sales": [100, 200, 300]})

    result = standard_scale(df, ["salary", "sales"])

    assert result["salary"].mean() == pytest.approx(0)
    assert result["sales"].mean() == pytest.approx(0)
    assert result["salary"].std() == pytest.approx(1)
    assert result["sales"].std() == pytest.approx(1)


def test_min_max_scale_scales_correctly():
    df = pd.DataFrame({"salary": [5000, 6000, 17000], "sales": [100, 200, 300]})

    result = min_max_scale(df, ["salary", "sales"])

    assert result["salary"].min() == pytest.approx(0)
    assert result["salary"].max() == pytest.approx(1)
    assert result["sales"].min() == pytest.approx(0)
    assert result["sales"].max() == pytest.approx(1)


def test_scale_constant_column_gives_zeros():
    df = pd.DataFrame({"constant": [5, 5, 5]})

    assert standard_scale(df, ["constant"])["constant"].tolist() == [0.0, 0.0, 0.0]
    assert min_max_scale(df, ["constant"])["constant"].tolist() == [0.0, 0.0, 0.0]


def test_input_not_modified():
    df = pd.DataFrame({"salary": [5000, 6000, 78000], "sales": [100, 200, 300]})
    original = df.copy()

    standard_scale(df, ["salary", "sales"])

    pd.testing.assert_frame_equal(df, original)


def test_ordinal_encode_unknown_value_raises_value_error():
    df = pd.DataFrame({"level of education": ["high school", "PhD", "bachelor's degree"]})

    with pytest.raises(ValueError):
        ordinal_encode(df, "level of education", EDUCATION_ORDER)


def test_missing_column_raises_key_error():
    df = pd.DataFrame({"level of education": ["high school", "bachelor's degree"]})

    with pytest.raises(KeyError):
        ordinal_encode(df, "sales", EDUCATION_ORDER)

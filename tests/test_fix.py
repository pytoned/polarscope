"""Tests for ps.fix() and the case-style column renaming."""

from __future__ import annotations

import polars as pl
import pytest

import polarscope as ps
from polarscope.clean import clean_column_names


def _messy_df() -> pl.DataFrame:
    return pl.DataFrame({
        "Kunde Navn": ["  Alice ", "Bob", "", "Bob", None],
        "Årlig Inntekt (kr)": [50000, 60000, 55000, 60000, None],
        "myColumnName": [1.0, 2.0, 3.0, 2.0, 1.0],
        "ratio": [0.1, 0.123456789012, 1e-60, 0.123456789012, None],
        "empty_col": pl.Series([None] * 5, dtype=pl.String),
        "const": ["x", "x", "x", "x", "x"],
    })


class TestCaseStyles:
    def test_snake_splits_camel_and_folds_ascii(self):
        df = pl.DataFrame({"Kunde Navn": [1], "myColumnName": [2], "Årlig Inntekt (kr)": [3]})
        out = clean_column_names(df, case="snake")
        assert out.columns == ["kunde_navn", "my_column_name", "arlig_inntekt_kr"]

    def test_camel_pascal_kebab(self):
        df = pl.DataFrame({"Kunde Navn": [1], "myColumnName": [2]})
        assert clean_column_names(df, case="camel").columns == ["kundeNavn", "myColumnName"]
        assert clean_column_names(df, case="pascal").columns == ["KundeNavn", "MyColumnName"]
        assert clean_column_names(df, case="kebab").columns == ["kunde-navn", "my-column-name"]

    def test_legacy_lower_upper_unchanged(self):
        # lower/upper keep the historical simple behavior (no camel splitting)
        df = pl.DataFrame({"MiXeD Name": [1], "myColumnName": [2]})
        assert clean_column_names(df, case="lower").columns == ["mixed_name", "mycolumnname"]
        assert clean_column_names(df, case="upper").columns == ["MIXED_NAME", "MYCOLUMNNAME"]

    def test_invalid_case_raises(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="case must be one of"):
            clean_column_names(df, case="bogus")
        with pytest.raises(ValueError, match="case must be one of"):
            ps.fix(df, case="bogus", verbose=False)


class TestFix:
    def test_defaults_are_lossless(self):
        df = _messy_df()
        out = ps.fix(df, verbose=False)

        # snake_case names
        assert out.columns == ["kunde_navn", "arlig_inntekt_kr", "my_column_name", "ratio", "const"]
        # strings stripped, empty -> null
        assert out["kunde_navn"].to_list() == ["Alice", "Bob", None, "Bob", None]
        # empty column dropped, rows untouched
        assert "empty_col" not in out.columns
        assert out.height == df.height
        # dtypes shrunk (with nulls present - regression for the null-skip bug)
        assert out.schema["arlig_inntekt_kr"] == pl.Int32
        assert out.schema["my_column_name"] == pl.Float32  # exactly representable
        # values Float32 cannot hold exactly keep their dtype and every digit
        assert out.schema["ratio"] == pl.Float64
        assert out["ratio"].to_list() == df["ratio"].to_list()
        # original never mutated
        assert df.columns[0] == "Kunde Navn"
        assert df["Kunde Navn"][0] == "  Alice "

    def test_case_none_keeps_names(self):
        out = ps.fix(_messy_df(), case=None, verbose=False)
        assert "Kunde Navn" in out.columns

    def test_opt_in_steps(self):
        df = _messy_df()
        out = ps.fix(
            df,
            drop_duplicate_rows=True,
            drop_constant_columns=True,
            missing_threshold=0.3,
            verbose=False,
        )
        assert "const" not in out.columns          # constant dropped
        assert "kunde_navn" not in out.columns     # 40% null after strip > 30%
        assert "ratio" in out.columns              # 20% null
        assert out.height == 4                     # one duplicate row removed

    def test_duplicates_kept_by_default(self):
        df = pl.DataFrame({"a": [1, 1, 2]})
        assert ps.fix(df, verbose=False).height == 3

    def test_outliers_opt_in(self):
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        assert ps.fix(df, verbose=False)["v"].null_count() == 0
        out = ps.fix(df, outliers="iqr", verbose=False)
        assert out["v"].to_list()[-1] is None
        with pytest.raises(ValueError, match="outliers must be"):
            ps.fix(df, outliers="bogus", verbose=False)
        with pytest.raises(ValueError, match="positive finite"):
            ps.fix(
                df,
                outliers="zscore",
                outlier_threshold=-1,
                verbose=False,
            )

    def test_missing_threshold_validation(self):
        with pytest.raises(ValueError, match="missing_threshold"):
            ps.fix(pl.DataFrame({"a": [1]}), missing_threshold=1.5, verbose=False)

    def test_verbose_report(self, capsys):
        ps.fix(_messy_df())
        captured = capsys.readouterr().out
        assert "ps.fix report" in captured
        assert "Renamed" in captured
        assert "Memory:" in captured

        ps.fix(_messy_df(), verbose=False)
        assert capsys.readouterr().out == ""

    def test_clean_frame_reports_nothing_to_fix(self, capsys):
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int8)})
        ps.fix(df)
        assert "Nothing to fix" in capsys.readouterr().out


class TestSignedIntegerDowncast:
    """Integer downcasting stays signed and no narrower than min_int_bits.

    Polars integer arithmetic wraps on overflow for every integer type, so
    the default floor is Int32. Unsigned dtypes also wrap on subtraction
    (0 - 1 -> 255), so unsigned columns become signed when that does not make
    them wider.
    """

    def test_default_floor_is_int32(self):
        out = ps.convert_datatypes(pl.DataFrame({"score": [10, 20, 30]}))
        assert out.schema["score"] == pl.Int32

    def test_later_arithmetic_does_not_overflow(self):
        # Under the old Int8 downcast, 70 + 90 wrapped to -96.
        df = pl.DataFrame({"age": [30, 70, 100], "score": [120, 90, 110]})
        out = ps.fix(df, verbose=False)
        assert out.select(pl.col("age") + pl.col("score"))["age"].to_list() == [150, 160, 210]

    def test_min_int_bits_allows_narrower_types(self):
        df = pl.DataFrame({"score": [10, 20, 30], "big": [0, 40_000, 1]})
        out = ps.convert_datatypes(df, min_int_bits=8)
        assert out.schema["score"] == pl.Int8
        assert out.schema["big"] == pl.Int32
        assert ps.convert_datatypes(df, min_int_bits=16).schema["score"] == pl.Int16
        with pytest.raises(ValueError, match="min_int_bits"):
            ps.convert_datatypes(df, min_int_bits=12)
        with pytest.raises(ValueError, match="min_int_bits"):
            ps.fix(df, min_int_bits=4, verbose=False)

    def test_downcast_survives_subtraction(self):
        out = ps.convert_datatypes(pl.DataFrame({"score": [10, 20, 30]}))
        assert out.select(pl.col("score") - 100)["score"].to_list() == [-90, -80, -70]

    def test_unsigned_becomes_signed_without_widening(self):
        df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.UInt32)})
        out = ps.convert_datatypes(df)
        assert out.schema["a"] == pl.Int32
        assert out["a"].to_list() == [1, 2, 3]
        df8 = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.UInt8)})
        assert ps.convert_datatypes(df8, min_int_bits=8).schema["a"] == pl.Int8

    def test_unsigned_is_never_widened(self):
        # 255 needs Int16; turning a UInt8 column into Int16 would double it.
        df = pl.DataFrame({"a": pl.Series([0, 255], dtype=pl.UInt8)})
        assert ps.convert_datatypes(df, min_int_bits=8).schema["a"] == pl.UInt8
        # Under the default floor every unsigned type narrower than 32 bits stays.
        assert ps.convert_datatypes(df).schema["a"] == pl.UInt8

    def test_uint64_beyond_int64_range_is_left_unchanged(self):
        big = 2**63 + 5  # does not fit in Int64, so it cannot be made signed
        df = pl.DataFrame({"a": pl.Series([0, big], dtype=pl.UInt64)})
        out = ps.convert_datatypes(df)
        assert out.schema["a"] == pl.UInt64
        assert out["a"].to_list() == [0, big]

    def test_fix_shrinks_to_signed_columns(self):
        out = ps.fix(pl.DataFrame({"Survived": [0, 1, 1, 0]}), verbose=False)
        assert out.schema["survived"] == pl.Int32
        assert out.select(pl.col("survived") - 1)["survived"].to_list() == [-1, 0, 0, -1]


class TestLosslessFloats:
    def test_only_exact_float32_values_are_downcast(self):
        df = pl.DataFrame(
            {
                "exact": [1.5, 0.25, None, float("nan"), float("inf")],
                "tiny": [1e-300, 2e-300, None, 0.0, 1.0],
                "precise": [3.141592653589793, 0.1, None, 2.0, 3.0],
            }
        )
        out = ps.convert_datatypes(df)
        assert out.schema["exact"] == pl.Float32
        assert out.schema["tiny"] == pl.Float64
        assert out.schema["precise"] == pl.Float64
        assert out["tiny"].to_list() == df["tiny"].to_list()

    def test_float16_is_not_widened(self):
        if not hasattr(pl, "Float16"):
            pytest.skip("Float16 needs a newer Polars")
        try:
            df = pl.DataFrame({"h": pl.Series([1.5, 2.5], dtype=pl.Float16)})
        except Exception:
            pytest.skip("this Polars cannot build Float16 columns")
        assert ps.convert_datatypes(df).schema["h"] == pl.Float16


class TestOutliersWithNonFiniteValues:
    @pytest.mark.parametrize("method", ["iqr", "zscore"])
    def test_one_nan_does_not_null_the_column(self, method):
        values = [float(i) for i in range(99)] + [float("nan")]
        out = ps.fix(pl.DataFrame({"x": values}), outliers=method, verbose=False)
        assert out["x"].null_count() == 0
        assert out["x"].is_nan().sum() == 1

    def test_many_nans_iqr_only_nulls_real_outliers(self):
        nan = float("nan")
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, nan, nan, nan, 1000.0]
        out = ps.fix(pl.DataFrame({"x": values}), outliers="iqr", verbose=False)["x"]
        assert out.null_count() == 1
        assert out[-1] is None
        assert out.is_nan().sum() == 3

    @pytest.mark.parametrize("method", ["iqr", "zscore"])
    def test_inf_is_kept_and_does_not_poison_fences(self, method):
        values = [1.0, 2.0, 3.0, 4.0, float("inf"), 5.0, 6.0, float("-inf")]
        out = ps.fix(pl.DataFrame({"x": values}), outliers=method, verbose=False)["x"]
        assert out.null_count() == 0
        assert out.to_list() == values

    def test_zscore_nulls_extreme_values(self):
        values = [10.0] * 30 + [11.0] * 30 + [500.0]
        out = ps.fix(pl.DataFrame({"x": values}), outliers="zscore", verbose=False)["x"]
        assert out.null_count() == 1
        assert out[-1] is None


class TestInputs:
    def test_lazyframes_are_rejected_with_a_clear_error(self):
        lf = pl.LazyFrame({"x": [1]})
        for fn in (ps.fix, ps.convert_datatypes, ps.drop_missing):
            with pytest.raises(TypeError, match="collect"):
                fn(lf)

    def test_clean_column_names_keeps_lazyframes_lazy(self):
        out = clean_column_names(pl.LazyFrame({"Kunde Navn": [1]}), case="snake")
        assert isinstance(out, pl.LazyFrame)
        assert out.collect_schema().names() == ["kunde_navn"]

    def test_pattern_like_column_names(self):
        df = pl.DataFrame({"*": [1, 1, 1], "^a.*$": [None, None, None], "abc": [1, 2, 2]})
        out = ps.fix(df, case=None, drop_constant_columns=True, drop_duplicate_rows=True, verbose=False)
        assert out.columns == ["abc"]
        assert out["abc"].to_list() == [1, 2]
        assert ps.drop_missing(pl.DataFrame({"*": [1, None], "y": [1, 2]}), axis="columns").columns == ["y"]
        assert ps.drop_missing(pl.DataFrame({"*": [1, None], "y": [1, 2]})).rows() == [(1, 1)]
        kept = ps.convert_datatypes(pl.DataFrame({"*": [1, 2], "^x$": [0.5, 1.5]}))
        assert kept.columns == ["*", "^x$"]
        assert kept.schema["^x$"] == pl.Float32

    def test_uncountable_columns_are_kept(self):
        # Object values cannot be counted distinct; the other columns still are.
        df = pl.DataFrame(
            {"o": pl.Series([object(), object()], dtype=pl.Object), "k": [1, 1], "x": [1, 2]}
        )
        out = ps.fix(df, drop_constant_columns=True, verbose=False)
        assert out.columns == ["o", "x"]

    def test_array_columns(self):
        df = pl.DataFrame(
            {
                "a": pl.Series([[1, 2], [1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
                "c": pl.Series([[0, 0]] * 3, dtype=pl.Array(pl.Int64, 2)),
                "x": [1, 1, 2],
            }
        )
        out = ps.fix(df, drop_constant_columns=True, drop_duplicate_rows=True, verbose=False)
        assert out.columns == ["a", "x"]
        assert out.height == 2


class TestBackwardCompat:
    def test_moved_functions_still_importable_from_plots(self):
        from polarscope.plots import convert_datatypes, drop_missing  # noqa: F401

    def test_data_cleaning_removed(self):
        import polarscope.plots as plots

        assert not hasattr(ps, "data_cleaning")
        assert not hasattr(plots, "data_cleaning")

    def test_convert_datatypes_shrinks_columns_with_nulls(self):
        df = pl.DataFrame({"x": [1, 2, None, 200], "s": ["a", "a", None, "b"]})
        out = ps.convert_datatypes(df)
        assert out.schema["x"] == pl.Int32
        assert out.schema["s"] == pl.Categorical
        assert out["x"].null_count() == 1

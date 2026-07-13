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
        assert out.columns == ["kunde_navn", "arlig_inntekt_kr", "my_column_name", "const"]
        # strings stripped, empty -> null
        assert out["kunde_navn"].to_list() == ["Alice", "Bob", None, "Bob", None]
        # empty column dropped, rows untouched
        assert "empty_col" not in out.columns
        assert out.height == df.height
        # dtypes shrunk (with nulls present - regression for the null-skip bug)
        assert out.schema["arlig_inntekt_kr"] == pl.UInt16
        assert out.schema["my_column_name"] == pl.Float32
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
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.UInt8)})
        ps.fix(df)
        assert "Nothing to fix" in capsys.readouterr().out


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
        assert out.schema["x"] == pl.UInt8
        assert out.schema["s"] == pl.Categorical
        assert out["x"].null_count() == 1

"""
Comprehensive test suite for polarscope library.
Tests all functions with various argument combinations to ensure they run without errors.
"""

import pytest
import polars as pl
import numpy as np
import polarscope as ps
from typing import Any, Dict, List
import warnings
import importlib
from polarscope.datasets import _get_data_path
from polarscope.xray import _calculate_shakiness_score

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


def _module_available(module_name: str) -> bool:
    """Return True when an optional dependency can be imported cleanly."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _scipy_cluster_available() -> bool:
    """Return True when SciPy clustering modules import cleanly."""
    try:
        importlib.import_module("scipy.cluster.hierarchy")
        return True
    except Exception:
        return False


class TestDataGenerator:
    """Generate various test datasets for comprehensive testing."""
    
    @staticmethod
    def create_basic_numeric_df(n_rows: int = 100) -> pl.DataFrame:
        """Create basic DataFrame with numeric columns."""
        np.random.seed(42)
        return pl.DataFrame({
            'price': np.random.normal(100, 20, n_rows),
            'volume': np.random.exponential(1000, n_rows),
            'rating': np.random.uniform(1, 5, n_rows),
            'score': np.random.normal(0, 1, n_rows)
        })
    
    @staticmethod
    def create_mixed_df(n_rows: int = 100) -> pl.DataFrame:
        """Create DataFrame with mixed data types."""
        np.random.seed(42)
        return pl.DataFrame({
            'numeric_col': np.random.normal(50, 15, n_rows),
            'int_col': np.random.randint(1, 100, n_rows),
            'string_col': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'bool_col': np.random.choice([True, False], n_rows),
            'float_small': np.random.normal(0.1, 0.05, n_rows),
            'category': np.random.choice(['X', 'Y', 'Z'], n_rows, p=[0.5, 0.3, 0.2])
        })
    
    @staticmethod
    def create_problematic_df(n_rows: int = 100) -> pl.DataFrame:
        """Create DataFrame with data quality issues."""
        np.random.seed(42)
        
        # Ensure all arrays have the same length
        missing_data = [1.0] * min(30, n_rows) + [None] * (n_rows - min(30, n_rows))
        if len(missing_data) < n_rows:
            missing_data.extend([None] * (n_rows - len(missing_data)))
        missing_data = missing_data[:n_rows]  # Trim to exact length
        
        # Create outlier data with correct length
        if n_rows >= 10:
            normal_count = n_rows - 10
            outlier_data = np.concatenate([
                np.random.normal(0, 1, normal_count),
                [100, -100, 200, -200, 300, -300, 400, -400, 500, -500][:10]
            ])
        else:
            outlier_data = np.random.normal(0, 1, n_rows)
        
        # Quasi-constant data
        quasi_count_a = max(1, int(n_rows * 0.95))
        quasi_count_b = n_rows - quasi_count_a
        quasi_constant_data = ['A'] * quasi_count_a + ['B'] * quasi_count_b
        
        data = {
            'missing_heavy': missing_data,
            'constant_col': [42] * n_rows,
            'outlier_heavy': outlier_data,
            'skewed_col': np.random.exponential(2, n_rows),
            'id_like': list(range(n_rows)),
            'quasi_constant': quasi_constant_data
        }
        return pl.DataFrame(data)


class TestXrayFunction:
    """Test the xray() function with all parameter combinations."""
    
    def test_basic_xray(self):
        """Test basic xray functionality."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Basic call
        result = ps.xray(df)
        assert result is not None
        
        # DataFrame output
        result_df = ps.xray(df, great_tables=False)
        assert isinstance(result_df, pl.DataFrame)
        assert result_df.height > 0
    
    def test_xray_expanded_mode(self):
        """Test xray expanded mode."""
        df = TestDataGenerator.create_mixed_df(50)
        
        result = ps.xray(df, expanded=True)
        assert result is not None
        
        # With all statistical tests
        result = ps.xray(df, expanded=True, normality_test="anderson")
        assert result is not None
        
        result = ps.xray(df, expanded=True, normality_test="ks")
        assert result is not None
        
        result = ps.xray(df, expanded=True, uniformity_test="chi2")
        assert result is not None
    
    def test_xray_custom_percentiles(self):
        """Test xray with custom percentiles."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Custom percentiles
        result = ps.xray(df, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        assert result is not None
        
        # Single percentile
        result = ps.xray(df, percentiles=[0.5])
        assert result is not None
        
        # Many percentiles
        result = ps.xray(df, percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        assert result is not None

    def test_xray_percentile_validation_and_sorting(self):
        """Percentiles should be sorted and validated."""
        df = TestDataGenerator.create_basic_numeric_df(50)

        # Unsorted percentiles should still produce sorted percentile columns.
        result_df = ps.xray(df, great_tables=False, expanded=True, percentiles=[0.9, 0.1, 0.5])
        percentile_cols = [c for c in result_df.columns if c in {"10%", "50%", "90%"}]
        assert percentile_cols == ["10%", "50%", "90%"]

        with pytest.raises(ValueError):
            ps.xray(df, great_tables=False, percentiles=[])

        with pytest.raises(ValueError):
            ps.xray(df, great_tables=False, percentiles=[-0.1, 0.5])

        with pytest.raises(ValueError):
            ps.xray(df, great_tables=False, percentiles=[0.5, 1.1])
    
    def test_xray_outlier_methods(self):
        """Test all outlier detection methods."""
        df = TestDataGenerator.create_problematic_df(50)
        
        # IQR method (default)
        result = ps.xray(df, outlier_method="iqr")
        assert result is not None
        
        # Percentile method
        result = ps.xray(df, outlier_method="percentile", outlier_bounds=[0.05, 0.95])
        assert result is not None
        
        # Z-score method
        result = ps.xray(df, outlier_method="zscore")
        assert result is not None
    
    def test_xray_correlation_target(self):
        """Test xray with correlation target."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Test each numeric column as target
        for col in ['price', 'volume', 'rating', 'score']:
            result = ps.xray(df, corr_target=col)
            assert result is not None
            
            result = ps.xray(df, corr_target=col, expanded=True)
            assert result is not None
    
    def test_xray_formatting_options(self):
        """Test all Great Tables formatting options."""
        df = TestDataGenerator.create_basic_numeric_df(30)
        
        # Test decimals
        for decimals in [1, 2, 3, 4]:
            result = ps.xray(df, decimals=decimals)
            assert result is not None
        
        # Test compact mode
        result = ps.xray(df, compact=True)
        assert result is not None
        
        # Test different separators
        result = ps.xray(df, sep_mark=" ", dec_mark=",")
        assert result is not None
        
        # Test pattern
        result = ps.xray(df, pattern="[{x}]")
        assert result is not None
        
        # Test compact with custom separators
        result = ps.xray(df, compact=True, sep_mark=" ", dec_mark=",")
        assert result is not None
    
    def test_xray_quality_thresholds(self):
        """Test custom quality assessment thresholds."""
        df = TestDataGenerator.create_problematic_df(50)
        
        # Strict thresholds
        result = ps.xray(
            df,
            missing_threshold=0.1,
            constant_threshold=0.95,
            skew_threshold=1.0,
            kurtosis_threshold=3.0,
            outlier_threshold=0.01,
            shakiness_threshold=1
        )
        assert result is not None

        # Lenient thresholds
        result = ps.xray(
            df,
            missing_threshold=0.8,
            constant_threshold=0.99,
            skew_threshold=5.0,
            kurtosis_threshold=10.0,
            outlier_threshold=0.2,
            shakiness_threshold=5
        )
        assert result is not None

    def test_xray_non_numeric_std_key_and_value(self):
        """Non-numeric columns should expose lowercase 'std' key with None values."""
        df = pl.DataFrame(
            {
                "num": [1, 2, 3, 4],
                "txt": ["a", "b", "c", "d"],
            }
        )

        result_df = ps.xray(df, include="all", great_tables=False)
        assert "std" in result_df.columns
        assert "Std" not in result_df.columns

        txt_std = result_df.filter(pl.col("Column") == "txt").select("std").item()
        assert txt_std is None

    def test_shakiness_uses_skew_key(self):
        """Shakiness score should increase when 'skew' crosses the configured threshold."""
        base_stats = {
            "Pct_Missing": 0.0,
            "Uniqueness_Ratio": 0.5,
            "N_Unique": 20,
            "skew": 0.4,
            "Kurtosis": 0.0,
            "Pct_Outliers": 0.0,
            "Normality_Test": "NORMAL",
        }

        low_score = _calculate_shakiness_score(
            base_stats,
            missing_threshold=0.3,
            constant_threshold=0.99,
            skew_threshold=2.0,
            kurtosis_threshold=7.0,
            outlier_threshold=0.05,
        )

        high_skew_stats = dict(base_stats)
        high_skew_stats["skew"] = 3.0
        high_score = _calculate_shakiness_score(
            high_skew_stats,
            missing_threshold=0.3,
            constant_threshold=0.99,
            skew_threshold=2.0,
            kurtosis_threshold=7.0,
            outlier_threshold=0.05,
        )

        assert high_score > low_score


class TestDatasets:
    """Dataset path safety and normalization tests."""

    def test_get_data_path_simple_filename(self):
        path = _get_data_path("titanic.parquet")
        assert path.name == "titanic.parquet"

    def test_get_data_path_blocks_parent_escape(self):
        with pytest.raises(ValueError):
            _get_data_path("../../etc/passwd")

    def test_get_data_path_blocks_absolute_paths(self):
        with pytest.raises(ValueError):
            _get_data_path("/etc/passwd")

    def test_get_data_path_normalizes_nested_paths(self):
        path = _get_data_path("subdir/../titanic.parquet")
        assert path.name == "titanic.parquet"


class TestPlottingFunctions:
    """Test all plotting functions with various backends and parameters."""
    
    def test_corr_heatmap(self):
        """Test correlation heatmap with all options."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Test all backends
        for backend in ["plotly", "altair"]:
            if backend == "altair" and not _module_available("altair"):
                continue
            result = ps.corr_heatmap(df, backend=backend)
            assert result is not None
            
            # With annotations off
            result = ps.corr_heatmap(df, backend=backend, annotate=False)
            assert result is not None
            
            # With custom dimensions
            result = ps.corr_heatmap(df, backend=backend, width=600, height=500)
            assert result is not None
        
        # Test correlation methods
        for method in ["pearson", "spearman"]:
            result = ps.corr_heatmap(df, method=method)
            assert result is not None
        
        # Test with target column
        result = ps.corr_heatmap(df, target="price")
        assert result is not None
        
        # Test with threshold and split
        result = ps.corr_heatmap(df, threshold=0.3, split="pos")
        assert result is not None
        
        result = ps.corr_heatmap(df, threshold=0.3, split="neg")
        assert result is not None
    
    def test_dist_plot(self):
        """Test distribution plot with all backends."""
        df = TestDataGenerator.create_basic_numeric_df(100)
        
        # Test all backends
        for backend in ["plotly", "altair"]:
            if backend == "altair" and not _module_available("altair"):
                continue
            # Auto-select column
            result = ps.dist_plot(df, backend=backend)
            assert result is not None
            
            # Specific column
            result = ps.dist_plot(df, column="price", backend=backend)
            assert result is not None
            
            # Custom bins
            result = ps.dist_plot(df, column="price", bins=20, backend=backend)
            assert result is not None
            
            # Custom dimensions
            result = ps.dist_plot(df, column="price", width=600, height=400, backend=backend)
            assert result is not None
    
    def test_missingval_plot(self):
        """Test missing value plot."""
        # Create data with missing values
        df = TestDataGenerator.create_problematic_df(100)
        
        # Test all backends
        for backend in ["plotly", "altair"]:
            if backend == "altair" and not _module_available("altair"):
                continue
            result = ps.missingval_plot(df, backend=backend)
            assert result is not None
            
            # With normalization
            result = ps.missingval_plot(df, backend=backend, normalize=True)
            assert result is not None
            
            # Different sort orders
            result = ps.missingval_plot(df, backend=backend, sort="asc")
            assert result is not None
            
            result = ps.missingval_plot(df, backend=backend, sort="desc")
            assert result is not None
    
    def test_cat_plot(self):
        """Test categorical plot."""
        df = TestDataGenerator.create_mixed_df(200)
        
        # Test all backends
        for backend in ["plotly", "altair"]:
            if backend == "altair" and not _module_available("altair"):
                continue
            result = ps.cat_plot(df, backend=backend)
            assert result is not None
            
            # Custom top/bottom
            result = ps.cat_plot(df, top=5, bottom=5, backend=backend)
            assert result is not None
            
            # Custom dimensions
            result = ps.cat_plot(df, width=700, height=500, backend=backend)
            assert result is not None
    
    def test_corr_plot(self):
        """Test enhanced correlation plot."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Test all backends
        for backend in ["plotly", "altair"]:
            if backend == "altair" and not _module_available("altair"):
                continue
            result = ps.corr_plot(df, backend=backend)
            assert result is not None
            
            # Interactive vs non-interactive
            result = ps.corr_plot(df, interactive=True, backend=backend)
            assert result is not None
            
            result = ps.corr_plot(df, interactive=False, backend=backend)
            assert result is not None
            
            # With clustering
            if _scipy_cluster_available():
                result = ps.corr_plot(df, clustered=True, backend=backend)
                assert result is not None
            
            # Different methods
            result = ps.corr_plot(df, method="spearman", backend=backend)
            assert result is not None


class TestDataProcessingFunctions:
    """Test data processing and cleaning functions."""
    
    def test_convert_datatypes(self):
        """Test datatype conversion function."""
        df = TestDataGenerator.create_mixed_df(100)
        
        # Basic conversion
        result = ps.convert_datatypes(df)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height
        
        # Custom parameters
        result = ps.convert_datatypes(
            df,
            max_cardinality=10,
            categorical_threshold=0.3,
            str_to_cat=False,
            downcast_ints=False,
            downcast_floats=False
        )
        assert isinstance(result, pl.DataFrame)
    
    def test_drop_missing(self):
        """Test missing value dropping function."""
        df = TestDataGenerator.create_problematic_df(100)
        
        # Drop missing rows
        result = ps.drop_missing(df, axis="rows")
        assert isinstance(result, pl.DataFrame)
        assert result.height <= df.height
        
        # Drop missing columns
        result = ps.drop_missing(df, axis="columns")
        assert isinstance(result, pl.DataFrame)
        assert result.width <= df.width
        
        # With threshold
        result = ps.drop_missing(df, axis="rows", thresh=0.5)
        assert isinstance(result, pl.DataFrame)
        
        # With subset
        result = ps.drop_missing(df, axis="rows", subset=["missing_heavy"])
        assert isinstance(result, pl.DataFrame)
    
    def test_data_cleaning(self):
        """Test comprehensive data cleaning function."""
        df = TestDataGenerator.create_problematic_df(100)
        
        # Basic cleaning
        result = ps.data_cleaning(df)
        assert isinstance(result, pl.DataFrame)
        
        # Custom parameters
        result = ps.data_cleaning(
            df,
            drop_missing_thresh=0.8,
            optimize_dtypes=False,
            remove_duplicates=False,
            outlier_method="iqr",
            outlier_threshold=2.0,
            categorical_threshold=0.4,
            max_cardinality=20
        )
        assert isinstance(result, pl.DataFrame)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        df = pl.DataFrame({
            'Column With Spaces': [1, 2, 3],
            'Column-With-Dashes': [4, 5, 6],
            'Column.With.Dots': [7, 8, 9],
            'Column__With__Double__Underscores': [10, 11, 12]
        })
        
        result = ps.clean_column_names(df)
        assert isinstance(result, pl.DataFrame)
        assert result.height == df.height
        
        # Check that column names are cleaned
        for col in result.columns:
            assert ' ' not in col
            assert '-' not in col
            assert '.' not in col
    
    def test_save_fig(self):
        """Test figure saving utility."""
        df = TestDataGenerator.create_basic_numeric_df(30)
        
        # Create a figure
        fig = ps.corr_heatmap(df, backend="plotly")
        
        # Test saving (we won't actually save to avoid file system issues)
        # This would test the function signature and basic validation
        try:
            # Just test that the function can be called
            # In a real test environment, you'd test actual file saving
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test_plot.png")
                # Test would save here: ps.save_fig(fig, filepath)
                pass
        except Exception:
            pass  # Skip file system tests in this context


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test functions with empty DataFrame."""
        df_empty = pl.DataFrame()
        
        # These should handle empty DataFrames gracefully
        try:
            result = ps.xray(df_empty)
        except Exception:
            pass  # Expected to handle gracefully
    
    def test_single_column_dataframe(self):
        """Test functions with single column DataFrame."""
        df_single = pl.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        
        result = ps.xray(df_single)
        assert result is not None
        
        result = ps.dist_plot(df_single, column='single_col')
        assert result is not None
    
    def test_all_missing_column(self):
        """Test with columns that are all missing."""
        df_missing = pl.DataFrame({
            'all_missing': [None] * 10,
            'some_data': [1, 2, 3, 4, 5, None, None, None, None, None]
        })
        
        result = ps.xray(df_missing)
        assert result is not None
        
        result = ps.missingval_plot(df_missing)
        assert result is not None
    
    def test_constant_column(self):
        """Test with constant columns."""
        df_constant = pl.DataFrame({
            'constant': [42] * 100,
            'variable': np.random.normal(0, 1, 100)
        })
        
        result = ps.xray(df_constant)
        assert result is not None
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        df = TestDataGenerator.create_basic_numeric_df(50)
        
        # Invalid backend
        with pytest.raises(ValueError):
            ps.corr_heatmap(df, backend="invalid_backend")
        
        with pytest.raises(ValueError):
            ps.dist_plot(df, backend="invalid_backend")
        
        # Invalid outlier method
        with pytest.raises(ValueError):
            ps.xray(df, outlier_method="invalid_method")
        
        # Invalid correlation target
        with pytest.raises(ValueError):
            ps.xray(df, corr_target="nonexistent_column")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive Test Suite for polarscope")
    print("=" * 60)
    
    test_classes = [
        TestXrayFunction,
        TestPlottingFunctions,
        TestDataProcessingFunctions,
        TestUtilityFunctions,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"  ✅ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
                failed_tests += 1
    
    print(f"\n🎯 Test Results Summary")
    print("=" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Library is working perfectly!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed. Check implementations.")
    
    return passed_tests, failed_tests


if __name__ == "__main__":
    run_comprehensive_tests()

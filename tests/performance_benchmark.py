"""
Performance benchmark for Polarscope library.
Inspired by klib's performance testing approach but adapted for Polars and our specific functions.
"""

import time
import psutil
import numpy as np
import polars as pl
import polarscope as ps
from typing import Dict, List, Tuple, Callable, Any
import warnings
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class PerformanceBenchmark:
    """Performance benchmarking suite for polarscope functions."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def create_test_data(self, n_rows: int, n_cols: int, include_categorical: bool = True, 
                        missing_rate: float = 0.1) -> pl.DataFrame:
        """Create test dataset with specified characteristics."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Numeric columns
        for i in range(n_cols // 2):
            col_data = np.random.normal(100, 20, n_rows)
            # Add missing values
            if missing_rate > 0:
                missing_indices = np.random.choice(n_rows, int(n_rows * missing_rate), replace=False)
                col_data[missing_indices] = np.nan
            data[f'numeric_{i}'] = col_data
        
        # Integer columns
        for i in range(n_cols // 4):
            col_data = np.random.randint(1, 1000, n_rows)
            if missing_rate > 0:
                missing_indices = np.random.choice(n_rows, int(n_rows * missing_rate), replace=False)
                col_data = col_data.astype(float)
                col_data[missing_indices] = np.nan
            data[f'integer_{i}'] = col_data
        
        # Categorical columns
        if include_categorical:
            for i in range(n_cols // 4):
                categories = [f'cat_{j}' for j in range(10)]
                col_data = np.random.choice(categories, n_rows)
                if missing_rate > 0:
                    missing_indices = np.random.choice(n_rows, int(n_rows * missing_rate), replace=False)
                    # Convert to list to handle None values properly
                    col_data = col_data.tolist()
                    for idx in missing_indices:
                        col_data[idx] = None
                data[f'categorical_{i}'] = col_data
        
        return pl.DataFrame(data)
    
    def measure_performance(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time and memory usage of a function."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        return {
            'execution_time': execution_time,
            'memory_usage_mb': max(0, memory_usage),  # Ensure non-negative
            'success': success,
            'error': error,
            'result_type': type(result).__name__ if result is not None else None
        }
    
    def benchmark_xray_function(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Benchmark xray function with different configurations."""
        print("🔬 Benchmarking xray() function...")
        
        xray_configs = [
            ("basic", {}),
            ("expanded", {"expanded": True}),
            ("high_precision", {"decimals": 4, "compact": True}),
            ("with_correlation", {"corr_target": None}),  # Will be set dynamically
            ("custom_percentiles", {"percentiles": [0.1, 0.25, 0.5, 0.75, 0.9]}),
            ("strict_quality", {
                "missing_threshold": 0.1, 
                "skew_threshold": 1.0, 
                "shakiness_threshold": 1
            }),
            ("dataframe_output", {"great_tables": False})
        ]
        
        for dataset_name, df in datasets.items():
            self.results[f'xray_{dataset_name}'] = {}
            
            # Get a numeric column for correlation if available
            numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
            corr_target = numeric_cols[0] if numeric_cols else None
            
            for config_name, config in xray_configs:
                if config_name == "with_correlation" and corr_target:
                    config = {"corr_target": corr_target}
                elif config_name == "with_correlation" and not corr_target:
                    continue  # Skip if no numeric columns
                
                print(f"  📊 {dataset_name} - {config_name}")
                result = self.measure_performance(ps.xray, df, **config)
                self.results[f'xray_{dataset_name}'][config_name] = result
    
    def benchmark_plotting_functions(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Benchmark plotting functions."""
        print("\n📈 Benchmarking plotting functions...")
        
        plotting_functions = [
            ("corr_heatmap", ps.corr_heatmap, [
                ({}, "default"),
                ({"backend": "plotly"}, "plotly"),
                ({"backend": "seaborn"}, "seaborn"),
                ({"backend": "altair"}, "altair"),
                ({"annotate": False}, "no_annotations")
            ]),
            ("dist_plot", ps.dist_plot, [
                ({}, "default"),
                ({"backend": "plotly"}, "plotly"),
                ({"backend": "seaborn"}, "seaborn"),
                ({"backend": "altair"}, "altair"),
                ({"bins": 50}, "many_bins")
            ]),
            ("missingval_plot", ps.missingval_plot, [
                ({}, "default"),
                ({"backend": "plotly"}, "plotly"),
                ({"backend": "seaborn"}, "seaborn"),
                ({"backend": "altair"}, "altair"),
                ({"normalize": True}, "normalized")
            ])
        ]
        
        for dataset_name, df in datasets.items():
            for func_name, func, configs in plotting_functions:
                self.results[f'{func_name}_{dataset_name}'] = {}
                
                for config, config_name in configs:
                    print(f"  📊 {func_name} - {dataset_name} - {config_name}")
                    
                    # Add column parameter for dist_plot if needed
                    if func_name == "dist_plot":
                        numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
                        if numeric_cols and "column" not in config:
                            config = {**config, "column": numeric_cols[0]}
                    
                    result = self.measure_performance(func, df, **config)
                    self.results[f'{func_name}_{dataset_name}'][config_name] = result
    
    def benchmark_data_processing(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Benchmark data processing functions."""
        print("\n🔧 Benchmarking data processing functions...")
        
        processing_functions = [
            ("convert_datatypes", ps.convert_datatypes, [
                ({}, "default"),
                ({"str_to_cat": False}, "no_categoricals"),
                ({"downcast_ints": False, "downcast_floats": False}, "no_downcasting")
            ]),
            ("drop_missing", ps.drop_missing, [
                ({"axis": "rows"}, "drop_rows"),
                ({"axis": "columns"}, "drop_columns"),
                ({"axis": "rows", "thresh": 0.5}, "thresh_50pct")
            ]),
            ("data_cleaning", ps.data_cleaning, [
                ({}, "default"),
                ({"optimize_dtypes": False}, "no_optimization"),
                ({"remove_duplicates": False}, "keep_duplicates"),
                ({"outlier_method": None}, "no_outlier_removal")
            ])
        ]
        
        for dataset_name, df in datasets.items():
            for func_name, func, configs in processing_functions:
                self.results[f'{func_name}_{dataset_name}'] = {}
                
                for config, config_name in configs:
                    print(f"  🔧 {func_name} - {dataset_name} - {config_name}")
                    result = self.measure_performance(func, df, **config)
                    self.results[f'{func_name}_{dataset_name}'][config_name] = result
    
    def run_full_benchmark(self) -> None:
        """Run complete performance benchmark suite."""
        print("🚀 Starting polarscope Performance Benchmark")
        print("=" * 60)
        
        # Create test datasets of varying sizes
        datasets = {
            "small": self.create_test_data(1000, 10),
            "medium": self.create_test_data(10000, 20),
            "large": self.create_test_data(50000, 30),
            "wide": self.create_test_data(5000, 100),  # Many columns
            "problematic": self.create_test_data(10000, 15, missing_rate=0.3)  # High missing rate
        }
        
        print(f"📊 Created {len(datasets)} test datasets:")
        for name, df in datasets.items():
            print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Run benchmarks
        start_total = time.perf_counter()
        
        self.benchmark_xray_function(datasets)
        self.benchmark_plotting_functions(datasets)
        self.benchmark_data_processing(datasets)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        print(f"\n⏱️  Total benchmark time: {total_time:.2f} seconds")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self) -> None:
        """Generate and display performance summary report."""
        print("\n📋 Performance Summary Report")
        print("=" * 50)
        
        # Collect all successful runs
        successful_runs = []
        failed_runs = []
        
        for test_name, test_configs in self.results.items():
            for config_name, result in test_configs.items():
                if result['success']:
                    successful_runs.append({
                        'test': test_name,
                        'config': config_name,
                        'time': result['execution_time'],
                        'memory': result['memory_usage_mb']
                    })
                else:
                    failed_runs.append({
                        'test': test_name,
                        'config': config_name,
                        'error': result['error']
                    })
        
        # Success rate
        total_tests = len(successful_runs) + len(failed_runs)
        success_rate = len(successful_runs) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"✅ Success Rate: {success_rate:.1f}% ({len(successful_runs)}/{total_tests} tests)")
        
        if failed_runs:
            print(f"❌ Failed Tests: {len(failed_runs)}")
            for run in failed_runs[:5]:  # Show first 5 failures
                print(f"  • {run['test']} - {run['config']}: {run['error'][:50]}...")
        
        # Performance statistics
        if successful_runs:
            times = [run['time'] for run in successful_runs]
            memories = [run['memory'] for run in successful_runs if run['memory'] > 0]
            
            print(f"\n⏱️  Execution Time Statistics:")
            print(f"  • Fastest: {min(times):.4f}s")
            print(f"  • Slowest: {max(times):.4f}s") 
            print(f"  • Average: {np.mean(times):.4f}s")
            print(f"  • Median: {np.median(times):.4f}s")
            
            if memories:
                print(f"\n💾 Memory Usage Statistics:")
                print(f"  • Lowest: {min(memories):.2f} MB")
                print(f"  • Highest: {max(memories):.2f} MB")
                print(f"  • Average: {np.mean(memories):.2f} MB")
            
            # Top performers and resource intensive tests
            print(f"\n🏆 Top 5 Fastest Tests:")
            fastest = sorted(successful_runs, key=lambda x: x['time'])[:5]
            for i, run in enumerate(fastest, 1):
                print(f"  {i}. {run['test']} - {run['config']}: {run['time']:.4f}s")
            
            print(f"\n🐌 Top 5 Slowest Tests:")
            slowest = sorted(successful_runs, key=lambda x: x['time'], reverse=True)[:5]
            for i, run in enumerate(slowest, 1):
                print(f"  {i}. {run['test']} - {run['config']}: {run['time']:.4f}s")
        
        # Resource usage assessment
        print(f"\n📊 Overall Assessment:")
        if success_rate >= 95:
            print("  🎉 Excellent: Very high success rate")
        elif success_rate >= 85:
            print("  ✅ Good: High success rate")
        else:
            print("  ⚠️  Needs attention: Lower success rate")
        
        avg_time = np.mean([run['time'] for run in successful_runs]) if successful_runs else 0
        if avg_time < 0.1:
            print("  ⚡ Lightning fast: Excellent performance")
        elif avg_time < 1.0:
            print("  🚀 Fast: Good performance")
        elif avg_time < 5.0:
            print("  ⏱️  Moderate: Acceptable performance")
        else:
            print("  🐌 Slow: Consider optimization")
    
    def save_detailed_results(self, filename: str = "performance_results.json") -> None:
        """Save detailed results to JSON file."""
        import json
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")


def run_performance_benchmark():
    """Main function to run the performance benchmark."""
    # Check system resources
    print("🖥️  System Information:")
    print(f"  CPU Count: {psutil.cpu_count()}")
    print(f"  Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print()
    
    # Run benchmark
    benchmark = PerformanceBenchmark()
    benchmark.run_full_benchmark()
    
    # Optionally save results
    try:
        benchmark.save_detailed_results()
    except Exception:
        pass  # Skip if file operations not available
    
    return benchmark


if __name__ == "__main__":
    run_performance_benchmark()

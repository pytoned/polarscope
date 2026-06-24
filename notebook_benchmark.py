"""
Simple notebook-friendly performance benchmark for Polarscope.
"""

import time
import psutil
import numpy as np
import polars as pl
import polarscope as ps
from polarscope.datasets import titanic, diabetes

def measure_performance(func, *args, **kwargs):
    """Measure execution time and memory usage of a function."""
    # Get initial memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Measure execution time
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    # Get final memory
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'execution_time': (end_time - start_time) * 1000,  # ms
        'memory_used': mem_after - mem_before,  # MB
        'result_type': type(result).__name__
    }

def create_test_data(n_rows=1000, n_cols=10):
    """Create test dataset."""
    np.random.seed(42)
    data = {}
    
    # Numeric columns
    for i in range(n_cols // 2):
        data[f'numeric_{i}'] = np.random.normal(100, 20, n_rows)
    
    # Categorical columns  
    for i in range(n_cols // 2):
        categories = ['A', 'B', 'C', 'D', 'E']
        data[f'category_{i}'] = np.random.choice(categories, n_rows)
    
    return pl.DataFrame(data)

def quick_benchmark():
    """Run a quick performance benchmark."""
    print("🔬 Polarscope Quick Performance Benchmark")
    print("=" * 50)
    
    # Test datasets
    datasets = {
        'titanic': titanic(),
        'diabetes': diabetes(), 
        'small_synthetic': create_test_data(1000, 8),
        'medium_synthetic': create_test_data(5000, 12)
    }
    
    results = {}
    
    for name, df in datasets.items():
        print(f"\n📊 Testing {name} dataset ({df.height} rows × {df.width} cols)")
        
        # Test basic xray
        result = measure_performance(ps.xray, df, great_tables=False)
        print(f"  xray (basic): {result['execution_time']:.1f}ms")
        
        # Test expanded xray
        result = measure_performance(ps.xray, df, expanded=True, great_tables=False)
        print(f"  xray (expanded): {result['execution_time']:.1f}ms")
        
        # Test with Great Tables (if small enough)
        if df.height < 2000:
            result = measure_performance(ps.xray, df, great_tables=True)
            print(f"  xray (great_tables): {result['execution_time']:.1f}ms")
        
        results[name] = result
    
    print(f"\n🎉 Benchmark complete!")
    return results

if __name__ == "__main__":
    quick_benchmark()

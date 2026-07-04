from .clean import clean_column_names, convert_datatypes, drop_missing, data_cleaning, fix
from .plots import corr_heatmap, dist_plot, missingval_plot, cat_plot, corr_plot
from .utils import save_fig
from .xray import xray
from . import datasets
from .datasets import titanic, diabetes, cardio
from . import clean
from . import plots
from . import utils

__all__ = [
    "cardio",               # Built-in Cardiovascular Disease dataset loader
    "cat_plot",             # Categorical data plotting
    "clean",                # Data cleaning module
    "clean_column_names",   # Column name standardization
    "convert_datatypes",    # Intelligent dtype optimization
    "corr_heatmap",         # Correlation heatmap visualization
    "corr_plot",            # Correlation scatter plots
    "data_cleaning",        # Deprecated cleaning pipeline (use fix)
    "datasets",             # Built-in datasets for testing
    "diabetes",             # Built-in Diabetes dataset loader
    "dist_plot",            # Distribution plotting
    "drop_missing",         # Missing value removal
    "fix",                  # One-call clean & optimize pipeline
    "missingval_plot",      # Missing value pattern visualization
    "plots",                # Plotting module
    "save_fig",             # Universal figure saving utility
    "titanic",              # Built-in Titanic dataset loader
    "utils",                # Utility functions module
    "xray",                 # Main data inspection function
]
__version__ = "1.9.0"

# Package metadata
__title__ = "polarscope"
__description__ = "🔬 Simple data inspection tools for Polars"
__author__ = "Anders & Co."

# Add module-level docstring for better IDE support
__doc__ = """
Polarscope: Simple data inspection tools for Polars DataFrames

Main functions:
    xray(df) - Comprehensive data inspection and quality assessment
    datasets.titanic() - Load built-in Titanic dataset
    datasets.diabetes() - Load built-in Diabetes dataset
    
Example:
    import polarscope as ps
    from polarscope.datasets import titanic
    
    df = titanic()
    ps.xray(df)
"""

# 🔬 Polarscope

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polars.svg)](https://pypi.org/project/polars/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Simple data inspection tools for Polars** 🐻‍❄️

Polarscope is a basic data analysis library for Polars DataFrames. It provides an `xray()` function for data inspection and some plotting utilities. Still early in development with more features planned.

---

## ✨ Current Features

### 🔬 Data Inspection (`xray`)
- Summary statistics for numeric columns by default.
- Optional `include='all'` support for mixed-type data.
- Optional expanded output (`expanded=True`) with normality/uniformity tests, outlier metrics, usability flags, and correlation details.
- Great Tables output (`great_tables=True`) or plain Polars DataFrame output (`great_tables=False`).

### 📊 Built-in Datasets
- `from polarscope.datasets import titanic, diabetes`
- Includes loader helpers and dataset metadata.

### 🧹 Utilities
- Column name cleaning (`clean_column_names`)
- Datatype optimization (`convert_datatypes`)
- Missing value dropping (`drop_missing`)
- Correlation/missing/distribution/categorical plots

### ✅ Polars-Only Data Handling
- Data handling is implemented with Polars.
- No Pandas is required for core data processing.

---

## 🚀 Quick Start

### Installation

```bash
pip install polarscope
```

Optional extras:

```bash
pip install "polarscope[all]"      # plotly + altair + scipy helpers
pip install "polarscope[dev]"      # pytest + coverage tools
```

### Basic Usage

```python
import polars as pl
import polarscope as ps
from polarscope.datasets import titanic

# Use a built-in dataset or load your own
df = titanic()
# df = pl.read_csv("your_data.csv")

# Get basic data summary
ps.xray(df)

# More detailed analysis
ps.xray(df, expanded=True)

# Custom title and correlation analysis
ps.xray(df, title="My Data Analysis", corr_target="Survived")
```

### Common `xray()` Options

```python
ps.xray(
    df,
    include="all",                # include all columns (not only numeric)
    expanded=True,                # additional stats/tests/quality fields
    corr_target="column_name",    # correlation against target column
    outlier_method="iqr",         # or "percentile"/"zscore"
    percentiles=[0.1, 0.5, 0.9],  # custom percentiles
    decimals=2,
    great_tables=False            # return Polars DataFrame
)
```

---

## 🩺 Troubleshooting

### Notebook still uses old code
If you changed source code locally but notebooks still show old behavior, reinstall in the same interpreter and restart kernel:

```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "/path/to/polarscope"])
```

### `ValueError: Only the x-axis of a nanoplot allows strings`
This comes from Great Tables nanoplot rendering when unsupported payloads reach the renderer. Update to latest local source and restart kernel.

---

## 🤝 Contributing

This is a small project, but contributions are welcome! Feel free to report bugs or suggest improvements.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by [`klib`](https://github.com/akanz1/klib) and built with [`great_tables`](https://github.com/posit-dev/great-tables).

---

**🔬 A simple tool for basic Polars data inspection.**

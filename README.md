![Polarscope — x-ray your Polars DataFrames: summary stats, distribution nanoplots, outliers and correlations in one table](docs/img/polarscope-cover.svg)

# 🔬 Polarscope - describe() on steroids

**One call, one table: a full profile of any Polars DataFrame. No pandas, anywhere, ever.**

```python
import polars as pl
import polarscope as ps

df = pl.read_csv('your-data.csv')
ps.xray(df)
```

![polarscope xray() profiling 70,000 rows in 21 ms](docs/img/xray-hero.png)

<sub>70,000 rows × 13 columns profiled in 21 ms — every column's dtype, nulls, quartiles, outliers, skew and distribution in a single call.</sub>

[![PyPI](https://img.shields.io/pypi/v/polarscope.svg)](https://pypi.org/project/polarscope/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polarscope.svg)](https://pypi.org/project/polarscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/polarscope?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/polarscope)

**Simple data inspection tools for Polars** 🐻‍❄️

Polarscope is a basic data analysis library for Polars DataFrames. It provides an `xray()` function for data inspection and some plotting utilities. Still early in development with more features planned.

---

## ✨ Current Features

### 🔬 Dataframe Summary Statistics (`xray`)
- Summary statistics for numeric columns by default; `include='all'` covers every dtype.
- Statistics for String/Categorical/Enum columns: top value and frequency, min/median/avg/max length, mode share, top-3 values, and value samples.
- Boolean columns analyzed as 0/1 (Mean = share of True); temporal columns report earliest/latest timestamps.
- Distribution nanoplot per column.
- Optional title, footnote, source note, and Great Tables theme (`title=…`, `footnote=…`, `source_note=…`, `theme="gray"`).
- Optional correlation (`corr_target=column_name`) against target for every column with nanoplot visualization.
- Optional expanded output (`expanded=True`) with normality/uniformity tests, outlier metrics and modelling usability flags.
- Optional compact mode (`compact=True`) that shortens big numbers to K (thousands) and M (millions) and decimal precision (`decimals=int`)
- Great Tables output by default or plain Polars DataFrame output (`great_tables=False`).

Quickly compute the correlation (Spearman or Pearson) to a target column:

![xray() on a DataFrame with correlation to target column enabled](docs/img/xray_corr.png)

### 🧹 Cleaning & Optimization (`fix`)
- `ps.fix(df)` cleans and optimizes in one call: column renaming (`case="snake"/"camel"/"pascal"/"kebab"/"upper"/"lower"`), whitespace stripping (empty strings → null), dtype shrinking, and empty-column removal — with a compact report of what changed.
- Opt-in extras: `drop_duplicate_rows`, `missing_threshold`, `drop_constant_columns`, `outliers="iqr"/"zscore"`.
- Lossless by default: anything that removes or alters data must be switched on explicitly.
- Building blocks also available standalone: `clean_column_names`, `convert_datatypes`, `drop_missing`.

### 📈 Plots
- Correlation/missing/distribution/categorical plots (plotly or altair backends)

### 📊 Built-in Datasets
- Three datasets ship with the package: `ps.titanic()`, `ps.diabetes()`, and `ps.cardio()`.
- Also available via `from polarscope.datasets import titanic, diabetes, cardio`, with loader functions.

### ✅ Polars-Only Data Handling
- Data handling is implemented with Polars.
- No Pandas, anywhere, ever. 

---

### 🔗 Customizing the Polarscope output

Polarscope is build on top of amazing libraries like Great Tables, Plotly and Altair. Polarscope functions like ps.xray() return native objects from these libraries 
(such as a GT table, a Plotly Figure, or an Altair Chart), so you retain complete access to their underlying APIs. This allows you to seamlessly chain native methods to further customize styling, annotations, 
and layout configurations directly on the Polarscope output. Below are three examples demonstrating how to extend these objects.

Great Tables customization:

```python
from great_tables import md, loc, style   # imported to enable extra functionality

(
    ps.xray(df[:, :6])
    .tab_header(
        title = "Example: GT method chaining onto ps.xray() with custom table styling, a source note and a footnote in markdown.",
        subtitle = ""
    )
    .tab_footnote(footnote=md("Some ***very*** insightful footnote about the data."))
    .tab_source_note("Source: data gathered from the Polarscope Times")
    .tab_style(
        style=style.text(color="purple", weight="bold"),
        locations=loc.footer()
    )
    .tab_options(
        table_width='60%',
        table_background_color="#F1E1FC",
        heading_background_color="#F1E1FC",
        column_labels_background_color="#F1E1FC",
        table_font_names="DejaVu Sans",
        column_labels_font_weight="bold"
    )
)
```

Customizing a GT table using method chaining:

![Customizing a GT table using method chaining](docs/img/gt_styling.png)


Plotly customization:

```python
(
    # base plot using Plotly backend
    ps.cat_plot(df.select(pl.col('sex', 'embarked')))
    
    # Customize the Plotly go object
    .update_traces(
        cliponaxis=False,
        marker_color="#25023b"  # Changes the fill color of the bars
    )
    .update_layout(
        template="plotly_white", 
        title="Example of Plotly customization",
        margin={"t": 100, "b": 100},
        plot_bgcolor="#F1E1FC",
        paper_bgcolor="#F1E1FC",
    )
)
```

Customizing a Plotly go object using method chaining:

![Customizing a Plotly chart](docs/img/plotly_styling.png)


Altair customization:

```python
import altair as alt

# base correlation plot using Polarscope corr_plot() with Altair backend
base_chart = ps.corr_plot(df[:, :9], width=600, height=300, backend="altair")

# Fetch the correlation values directly from the base chart
text_labels = base_chart.mark_text(baseline='middle').encode(
    text=alt.Text('correlation:Q', format='.0%'),
    
    # Dynamically change text color based on the correlation value and use white text if cell background is dark
    color=alt.condition(
        abs(alt.datum.correlation) > 0.7,
        alt.value('white'),
        alt.value('black')
    )
)

# Layer the charts (+) and chain final configurations onto the result
final_chart = (
    (base_chart + text_labels)
    .properties(
        title="Example of Altair backend chart customization",
        padding={"left": 40, "right": 40, "top": 20, "bottom": 20}
    )
    .configure_title(
        font="DejaVu Sans",
        color="darkblue",
        anchor="start",
        fontSize=20,
    )
)
final_chart.show()
```

Customizing an Altair chart object using method chaining:

![Customizing an Altair chart object using method chaining](docs/img/altair_styling.png)

---

## 🔍 How Polarscope compares

| | **polarscope** 1.9.6 | skimpy 0.0.21 | ydata-profiling 4.18.4 | klib 1.4.1 |
|---|---|---|---|---|
| Takes a Polars DataFrame directly | ✅ | ✅ | ❌ convert to pandas first | ❌ convert to pandas first |
| Requires/converts to pandas | **✅ never** | ❌ hard dependency | ❌ | ❌ |
| Core dependencies | **3** | 12 | 21 | 8 |
| Minimum Python | **3.9** | 3.11 | 3.10 | 3.10 |
| Primary output | Great Tables HTML, or Polars DataFrame | terminal (rich) | standalone HTML report | matplotlib/seaborn figures |
| String / categorical stats | ✅ | ✅ | ✅ | partial |
| One-call cleaning and dtype optimization | ✅ `ps.fix()` | ❌ | ❌ | ✅ |

**Where the others are the better choice:**

- **ydata-profiling** produces a far deeper report in one call — variable interactions, correlation matrices, alerts and warnings. If you want an exhaustive standalone HTML artifact and don't mind the 21 dependencies or the pandas conversion, it does more than polarscope does.
- **skimpy** also reads Polars natively and prints straight to the terminal, which is the nicer fit for CLI and script workflows. Polarscope targets notebooks with presentation ready outputs.
- **klib** has a mature pandas cleaning and plotting suite. Polarscope's plotting API is openly modelled on it.

**Where polarscope wins:** it is the only one of the four with **no pandas anywhere in its dependency tree**, and the only one installable on Python 3.9 — which matters if your environment is locked down and you cannot pull in pandas, seaborn and numba just to look at a table.

<sub>Verified 2026-08-10 against each project's published PyPI metadata (`requires_dist`, `requires_python`) at the versions listed. Dependency counts exclude optional extras. Corrections welcome — please open an issue.</sub>

---

## 🚀 Quick Start

### Installation

```bash
pip install polarscope
```

Optional extras:

```bash
pip install "polarscope[altair]"   # optional Altair plotting backend
pip install "polarscope[plotly]"   # Kaleido support for static Plotly images
pip install "polarscope[scipy]"    # statistical tests
pip install "polarscope[all]"      # all optional features
pip install "polarscope[dev]"      # pytest + coverage tools
```

Plotly is included in the base installation and is the default plotting backend.

### Basic Usage

```python
import polars as pl
import polarscope as ps

# Use a built-in dataset or load your own
df = ps.titanic()          # also: ps.diabetes(), ps.cardio()
# df = pl.read_csv("your_data.csv")

# Get basic data summary (all column types)
ps.xray(df, include="all")

# More detailed analysis
ps.xray(df, include="all", expanded=True)

# Custom title and correlation analysis
ps.xray(df, title="My Data Analysis", corr_target="Survived")

# Clean & optimize in one call (snake_case names, trimmed strings,
# shrunk dtypes, empty columns dropped - with a report)
df = ps.fix(df)

# Or opt in to deeper cleaning
df = ps.fix(df, case="camel", drop_duplicate_rows=True, missing_threshold=0.9)
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
    footnote="Excludes nulls.",   # Great Tables footnote (requires great_tables>=0.22)
    source_note="Source: …",      # Great Tables source note
    theme="gray",                 # or 1-6, or {"style": 2, "color": "blue"}
    great_tables=False            # return Polars DataFrame
)
```

`ps.xray()`, `ps.fix()`, and the plotting helpers also accept a Polars `LazyFrame` (collected at the call boundary). Other frame types raise a clear `TypeError` — pandas is not converted automatically.

### Customizing Beyond the Built-in Options

polarscope returns the underlying library's own object rather than a wrapper, so
anything `great_tables` or your plotting backend can do is available by chaining
onto the result — no polarscope parameter needed.

`ps.xray()` returns a `GT` object. Chained calls run after polarscope's own
styling, so they win where the two overlap:

```python
from great_tables import loc, md

(
    ps.xray(df)
    .tab_options(table_background_color="#fdf6e3")
    .tab_footnote(
        footnote=md("Excludes nulls."),
        locations=loc.body(columns="mean", rows=[0]),
    )
)
```

Column names are not relabelled, so the names from `great_tables=False` output
are the identifiers to use in chained calls.

The plotting functions return a Plotly `Figure` or Altair `Chart`:

```python
# Per figure
ps.missingval_plot(df).update_layout(template="plotly_dark")

# Or globally, using the backend's own theme setting
import plotly.io as pio
pio.templates.default = "plotly_dark"
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to report bugs or suggest improvements.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Inspired by [`klib`](https://github.com/akanz1/klib) and built with [`Polars`](https://github.com/pola-rs/polars) and [`great_tables`](https://github.com/posit-dev/great-tables).

---

**🔬 A quick and lightweight tool for Polars dataframe stats and visualizations.**

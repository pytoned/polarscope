"""
Dataset loading utilities for Polarscope.

This module provides easy access to built-in datasets for testing and demonstration
of Polarscope functionality.
"""

import polars as pl
from pathlib import Path


def _normalize_dataset_relative_path(filename: str) -> Path:
    """
    Normalize a relative dataset path and block traversal outside dataset roots.

    This allows safe subpaths such as ``subdir/file.csv`` while rejecting
    absolute paths and ``..`` escapes.
    """
    raw = filename.replace("\\", "/")
    path = Path(raw)

    if path.is_absolute() or (":" in raw.split("/")[0]):
        raise ValueError("Dataset path must be relative")

    safe_parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not safe_parts:
                raise ValueError("Dataset path cannot escape dataset directory")
            safe_parts.pop()
            continue
        safe_parts.append(part)

    if not safe_parts:
        raise ValueError("Dataset path is empty after normalization")

    normalized = Path(*safe_parts)
    if normalized.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Dataset file must have .csv or .parquet extension")

    return normalized


def _get_data_path(filename: str) -> Path:
    """Get the path to a dataset file, handling both development and installed package scenarios."""
    normalized = _normalize_dataset_relative_path(filename)

    # Try package data first (installed package scenario)
    try:
        import importlib.resources as pkg_resources

        try:
            resource_path = pkg_resources.files("polarscope").joinpath("data", *normalized.parts)
            if resource_path.is_file():
                return Path(resource_path)
        except (AttributeError, ModuleNotFoundError):
            pass
    except (ImportError, FileNotFoundError):
        pass

    # Development mode fallback (source tree): datasets live in polarscope/data
    current_file = Path(__file__).parent
    candidate_paths = [
        current_file / "data" / normalized,
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not find dataset file: {normalized}")


def load_titanic(return_polars: bool = True) -> pl.DataFrame:
    """
    Load the Titanic dataset.
    
    The Titanic dataset contains passenger information from the RMS Titanic,
    including survival status, passenger class, demographics, and fare information.
    Perfect for demonstrating data analysis, missing value handling, and 
    classification tasks.
    
    Parameters
    ----------
    return_polars : bool, default True
        If True, returns a Polars DataFrame. If False, returns the file path
        as a string for custom loading.
    
    Returns
    -------
    pl.DataFrame or str
        Either a Polars DataFrame containing the Titanic data, or the path
        to the parquet file.
        
    Examples
    --------
    Load the Titanic dataset:
    
    >>> from polarscope.datasets import load_titanic
    >>> df = load_titanic()
    >>> print(df.shape)
    (156, 12)
    
    Get the file path instead:
    
    >>> file_path = load_titanic(return_polars=False)
    >>> df = pl.read_parquet(file_path)
    
    Dataset Information
    -------------------
    - **Rows**: 156 passengers (subset of original dataset)
    - **Columns**: 12 features
    - **Missing values**: Yes (Age, Cabin, Embarked)
    - **Data types**: Mixed (numeric, string, categorical)
    
    Columns:
    - PassengerId: Unique passenger identifier
    - Survived: Survival status (0 = No, 1 = Yes)
    - Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
    - Name: Passenger name
    - Sex: Gender (male/female)
    - Age: Age in years
    - SibSp: Number of siblings/spouses aboard
    - Parch: Number of parents/children aboard
    - Ticket: Ticket number
    - Fare: Passenger fare
    - Cabin: Cabin number
    - Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
    """
    data_path = _get_data_path("titanic.parquet")
    
    if not return_polars:
        return str(data_path)
    
    return pl.read_parquet(data_path)


def load_diabetes(return_polars: bool = True) -> pl.DataFrame:
    """
    Load the Diabetes dataset.
    
    The Pima Indians Diabetes dataset contains medical diagnostic information
    for predicting diabetes onset. All features are numeric, making it excellent
    for statistical analysis and demonstrating numeric data processing.
    
    Parameters
    ----------
    return_polars : bool, default True
        If True, returns a Polars DataFrame. If False, returns the file path
        as a string for custom loading.
    
    Returns
    -------
    pl.DataFrame or str
        Either a Polars DataFrame containing the diabetes data, or the path
        to the parquet file.
        
    Examples
    --------
    Load the diabetes dataset:
    
    >>> from polarscope.datasets import load_diabetes
    >>> df = load_diabetes()
    >>> print(df.shape)
    (768, 9)
    
    Get the file path instead:
    
    >>> file_path = load_diabetes(return_polars=False)
    >>> df = pl.read_parquet(file_path)
    
    Dataset Information
    -------------------
    - **Rows**: 768 patients
    - **Columns**: 9 features (8 predictors + 1 target)
    - **Missing values**: None (but contains zeros that may represent missing)
    - **Data types**: All numeric
    
    Columns:
    - Pregnancies: Number of times pregnant
    - Glucose: Plasma glucose concentration
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index (weight in kg/(height in m)^2)
    - DiabetesPedigreeFunction: Diabetes pedigree function
    - Age: Age in years
    - Outcome: Class variable (0 or 1) - diabetes diagnosis
    """
    data_path = _get_data_path("diabetes.parquet")

    if not return_polars:
        return str(data_path)

    return pl.read_parquet(data_path)


def load_cardio(return_polars: bool = True) -> pl.DataFrame:
    """
    Load the Cardiovascular Disease dataset.

    The Cardio dataset contains health examination records used to predict the
    presence of cardiovascular disease. It is a larger, all-numeric dataset,
    making it useful for demonstrating performance on bigger frames and for
    classification tasks.

    Parameters
    ----------
    return_polars : bool, default True
        If True, returns a Polars DataFrame. If False, returns the file path
        as a string for custom loading.

    Returns
    -------
    pl.DataFrame or str
        Either a Polars DataFrame containing the cardio data, or the path
        to the parquet file.

    Examples
    --------
    Load the cardio dataset:

    >>> from polarscope.datasets import load_cardio
    >>> df = load_cardio()
    >>> print(df.shape)
    (70000, 13)

    Get the file path instead:

    >>> file_path = load_cardio(return_polars=False)
    >>> df = pl.read_parquet(file_path)

    Dataset Information
    -------------------
    - **Rows**: 70,000 examination records
    - **Columns**: 13 features (incl. the ``cardio`` target)
    - **Missing values**: None
    - **Data types**: All numeric

    Columns:
    - id: Record identifier
    - age: Age in days
    - gender: Gender code (1, 2)
    - height: Height in cm
    - weight: Weight in kg
    - ap_hi: Systolic blood pressure
    - ap_lo: Diastolic blood pressure
    - cholesterol: Cholesterol level (1 = normal, 2 = above normal, 3 = well above normal)
    - gluc: Glucose level (1 = normal, 2 = above normal, 3 = well above normal)
    - smoke: Whether the patient smokes (0/1)
    - alco: Whether the patient drinks alcohol (0/1)
    - active: Whether the patient is physically active (0/1)
    - cardio: Presence of cardiovascular disease (0 = No, 1 = Yes) - target
    """
    data_path = _get_data_path("cardio.parquet")

    if not return_polars:
        return str(data_path)

    return pl.read_parquet(data_path)


# Convenience aliases for easier access
titanic = load_titanic
diabetes = load_diabetes
cardio = load_cardio


def list_datasets() -> list[str]:
    """
    List all available datasets.

    Returns
    -------
    list[str]
        List of available dataset names.

    Examples
    --------
    >>> from polarscope.datasets import list_datasets
    >>> datasets = list_datasets()
    >>> print(datasets)
    ['titanic', 'diabetes', 'cardio']
    """
    return ['titanic', 'diabetes', 'cardio']


def dataset_info(name: str) -> str:
    """
    Get information about a specific dataset.
    
    Parameters
    ----------
    name : str
        Name of the dataset ('titanic' or 'diabetes').
        
    Returns
    -------
    str
        Detailed information about the dataset.
        
    Examples
    --------
    >>> from polarscope.datasets import dataset_info
    >>> info = dataset_info('titanic')
    >>> print(info)
    """
    if name.lower() == 'titanic':
        return """
Titanic Dataset
===============
Famous passenger manifest from RMS Titanic with survival outcomes.

• Rows: 156 passengers
• Columns: 12 features  
• Missing values: Yes (Age, Cabin, Embarked)
• Use case: Classification, missing value analysis, categorical data
• Perfect for: Demonstrating xray(), missing value plots, correlation analysis
        """.strip()
    
    elif name.lower() == 'diabetes':
        return """
Diabetes Dataset  
================
Pima Indians Diabetes medical diagnostic data.

• Rows: 768 patients
• Columns: 9 features
• Missing values: None (but zeros may represent missing)
• Use case: Medical prediction, statistical analysis
• Perfect for: Demonstrating statistical tests, distribution analysis, correlation
        """.strip()

    elif name.lower() == 'cardio':
        return """
Cardiovascular Disease Dataset
==============================
Health examination records for cardiovascular disease prediction.

• Rows: 70,000 records
• Columns: 13 features (incl. the `cardio` target)
• Missing values: None
• Use case: Classification, performance on larger frames
• Perfect for: Demonstrating xray() on big data, distribution/outlier analysis
        """.strip()

    else:
        available = ', '.join(list_datasets())
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")


# For backward compatibility and convenience
__all__ = [
    'load_titanic', 'load_diabetes', 'load_cardio',
    'titanic', 'diabetes', 'cardio',
    'list_datasets', 'dataset_info'
]

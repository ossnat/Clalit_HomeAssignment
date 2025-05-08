import time
from functools import wraps
from typing import Optional, Dict, List

import pandas as pd


def timing_decorator(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper


@timing_decorator
def load_data(file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from file efficiently with optional sampling for large datasets.

    Args:
        file_path: Path to the data file
        sample_size: Number of samples to randomly select (None for all data)

    Returns:
        DataFrame with loaded data
    """
    # Check file extension and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delimiter=' ')
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # If sample size provided and smaller than actual size, sample the data
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify and categorize column types in the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with columns categorized by type
    """
    column_types = {
        "id_columns": [],
        "target_columns": ["Disease"],
        "numerical_features": [],
        "categorical_features": []
    }

    # Identify ID columns
    if "ID" in df.columns:
        column_types["id_columns"].append("ID")

    # Identify numerical and categorical features
    for col in df.columns:
        # Skip ID and target columns
        if col in column_types["id_columns"] or col in column_types["target_columns"]:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            column_types["numerical_features"].append(col)
        else:
            column_types["categorical_features"].append(col)

    return column_types
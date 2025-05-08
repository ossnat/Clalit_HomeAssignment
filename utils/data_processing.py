from typing import Dict, Any

import pandas as pd
from scipy import stats
from general_utils import timing_decorator


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic information about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with basic dataset information
    """
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        "missing_values": df.isna().sum().to_dict()
    }
    return info


@timing_decorator
def analyze_categorical_columns(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Analyze categorical columns and count their unique values.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with categorical column value counts
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    result = {}

    for col in categorical_cols:
        # Use value_counts for efficiency
        result[col] = df[col].value_counts().to_dict()

    return result


@timing_decorator
def analyze_numerical_columns(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze numerical columns with descriptive statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with numerical column statistics
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    result = {}

    # Calculate statistics in one go for efficiency
    stats_df = df[numerical_cols].describe().T

    # Add additional statistics
    for col in numerical_cols:
        col_data = df[col].dropna()

        result[col] = {
            "mean": stats_df.loc[col, "mean"],
            "std": stats_df.loc[col, "std"],
            "min": stats_df.loc[col, "min"],
            "25%": stats_df.loc[col, "25%"],
            "median": stats_df.loc[col, "50%"],
            "75%": stats_df.loc[col, "75%"],
            "max": stats_df.loc[col, "max"],
            "skewness": stats.skew(col_data),
            "kurtosis": stats.kurtosis(col_data)
        }

    return result

from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from general_utils import timing_decorator

@timing_decorator
def plot_disease_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of disease labels.

    Args:
        df: Input DataFrame
        save_path: Path to save the figure (None for display only)
    """
    plt.figure(figsize=(12, 6))
    disease_counts = df['Disease'].value_counts()

    # Plot the distribution
    sns.barplot(x=disease_counts.index, y=disease_counts.values)
    plt.title('Distribution of Disease Types', fontsize=15)
    plt.xlabel('Disease', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


@timing_decorator
def plot_numerical_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None,
                               save_dir: Optional[str] = None) -> None:
    """
    Plot histograms for numerical features.

    Args:
        df: Input DataFrame
        columns: List of columns to plot (None for all numerical)
        save_dir: Directory to save figures (None for display only)
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Plot 2 columns per row
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=15)
            axes[i].set_xlabel(col, fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/numerical_distributions.png")
    plt.show()


@timing_decorator
def plot_categorical_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                 save_dir: Optional[str] = None) -> None:
    """
    Plot count plots for categorical features.

    Args:
        df: Input DataFrame
        columns: List of columns to plot (None for all categorical)
        save_dir: Directory to save figures (None for display only)
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns

    # Plot 2 columns per row
    n_cols = 2
    n_rows = (len(columns) + n_cols - 1) // n_cols

    if len(columns) > 0:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(columns):
            if i < len(axes):
                sns.countplot(y=col, data=df, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}', fontsize=15)
                axes[i].set_xlabel('Count', fontsize=12)
                axes[i].set_ylabel(col, fontsize=12)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/categorical_distributions.png")
        plt.show()


@timing_decorator
def plot_feature_correlations(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot correlation heatmap for numerical features.

    Args:
        df: Input DataFrame
        save_path: Path to save the figure (None for display only)
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Calculate correlations
    corr_matrix = df[numerical_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")

    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


@timing_decorator
def plot_numerical_by_disease(df: pd.DataFrame, columns: Optional[List[str]] = None,
                            save_dir: Optional[str] = None) -> None:
    """
    Plot boxplots of numerical features grouped by disease.

    Args:
        df: Input DataFrame
        columns: List of columns to plot (None for all numerical)
        save_dir: Directory to save figures (None for display only)
    """
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Disease', y=col, data=df)
        plt.title(f'{col} by Disease Type', fontsize=15)
        plt.xlabel('Disease', fontsize=15)
        plt.ylabel(col, fontsize=15)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/{col}_by_disease.png")
        plt.show()


@timing_decorator
def plot_categorical_by_disease(df: pd.DataFrame, columns: Optional[List[str]] = None,
                              save_dir: Optional[str] = None) -> None:
    """
    Plot count plots of categorical features grouped by disease.

    Args:
        df: Input DataFrame
        columns: List of columns to plot (None for all categorical)
        save_dir: Directory to save figures (None for display only)
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove Disease from columns to analyze
        if 'Disease' in columns:
            columns.remove('Disease')

    for col in columns:
        plt.figure(figsize=(12, 6))
        # Create grouped counts for plotting
        grouped_data = df.groupby(['Disease', col]).size().unstack(fill_value=0)
        grouped_data.plot(kind='bar', stacked=True)

        plt.title(f'Distribution of {col} by Disease Type', fontsize=15)
        plt.xlabel('Disease', fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.xticks(rotation=45)
        plt.legend(title=col)
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/{col}_by_disease.png")
        plt.show()

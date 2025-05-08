import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import time
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from joblib import Parallel, delayed
import dask.dataframe as dd
from utils.general_utils import timing_decorator

class DataProcessor:
    """
    Class for efficiently processing large datasets.
    Implements strategies for handling large data like chunking and parallel processing.
    """

    def __init__(self, file_path: str = None, df: pd.DataFrame = None,
                 chunk_size: int = 100000, sampling_frac:int = 10**-3,
                 n_jobs: int = -1):
        """
        Initialize the data processor.

        Args:
            file_path: Path to the data file (None if df is provided)
            df: DataFrame to use directly (None if file_path is provided)
            chunk_size: Size of chunks for processing large data
            n_jobs: Number of parallel jobs (-1 for all available cores)
        """
        self.file_path = file_path
        self.df = df
        self.chunk_size = chunk_size
        self.sampling_frac = sampling_frac
        self.n_jobs = n_jobs
        self.dask_df = None

        # Load data if file_path is provided
        if file_path and not df:
            self.load_data()

    @timing_decorator
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data with efficient strategies for large files.

        Args:
            sample_size: Number of samples for testing (None for all data)

        Returns:
            DataFrame with loaded data
        """
        if not self.file_path:
            raise ValueError("File path not provided")

        file_ext = os.path.splitext(self.file_path)[1].lower()

        # For large CSV files, use dask for lazy loading
        if file_ext == '.csv':
            print(f"Loading data with dask from {self.file_path}")
            self.dask_df = dd.read_csv(self.file_path)

            # For sampling, we compute a subset
            if sample_size:
                # Get random sample
                self.df = self.dask_df.sample(frac= self.sampling_frac).compute()
            else:
                # For full data, keep using dask for computations
                # and only convert to pandas when needed
                self.df = self.dask_df.compute()

        elif file_ext in ['.xlsx', '.xls']:
            # For Excel files, use chunking
            print(f"Loading Excel data from {self.file_path}")
            if sample_size:
                self.df = pd.read_excel(self.file_path, nrows=sample_size)
            else:
                # Use chunking for large Excel files
                chunks = []
                for chunk in pd.read_excel(self.file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)
                self.df = pd.concat(chunks)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        print(f"Data loaded with shape: {self.df.shape}")
        return self.df

    @timing_decorator
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset efficiently.

        Returns:
            Cleaned DataFrame
        """
        if self.dask_df is not None:
            # Handle with dask for large datasets
            # Convert column types for efficiency
            for col in self.dask_df.columns:
                # Try to convert object columns to categorical
                if self.dask_df[col].dtype == 'object':
                    self.dask_df[col] = self.dask_df[col].astype('category')

            # Handle missing values
            # For demonstration, fill numeric with mean and categorical with mode
            # In practice, more sophisticated strategies may be needed
            self.df = self.dask_df.compute()

        # Type conversion for memory efficiency
        cat_columns = self.df.select_dtypes(include=['object']).columns
        for col in cat_columns:
            self.df[col] = self.df[col].astype('category')

        # Handle missing values
        num_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        for col in cat_columns:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        print(f"Data cleaned with shape: {self.df.shape}")
        return self.df

    @timing_decorator
    def get_distribution_stats(self, column: str) -> Dict[str, Any]:
        """
        Get distribution statistics for a column.

        Args:
            column: Column name to analyze

        Returns:
            Dictionary with distribution statistics
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")

        col_data = self.df[column]

        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric columns
            result = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "skewness": stats.skew(col_data.dropna()),
                "kurtosis": stats.kurtosis(col_data.dropna()),
                "quantiles": {
                    "25%": col_data.quantile(0.25),
                    "50%": col_data.quantile(0.50),
                    "75%": col_data.quantile(0.75),
                    "90%": col_data.quantile(0.90),
                    "95%": col_data.quantile(0.95),
                    "99%": col_data.quantile(0.99)
                }
            }
        else:
            # For categorical columns
            value_counts = col_data.value_counts()
            result = {
                "unique_values": col_data.nunique(),
                "most_common": value_counts.index[0] if not value_counts.empty else None,
                "most_common_count": value_counts.iloc[0] if not value_counts.empty else 0,
                "distribution": value_counts.to_dict()
            }

        return result

    @timing_decorator
    def analyze_all_distributions(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze distributions of all columns.

        Returns:
            Dictionary with distribution statistics for all columns
        """
        results = {}

        # Process columns in parallel for efficiency
        def process_column(column):
            return column, self.get_distribution_stats(column)

        # Use parallel processing
        processed_results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_column)(col) for col in self.df.columns
        )

        # Combine results
        for col, stats in processed_results:
            results[col] = stats

        return results

    @timing_decorator
    def chunked_apply(self, func, *args, **kwargs) -> Any:
        """
        Apply a function to the DataFrame in chunks for memory efficiency.

        Args:
            func: Function to apply to each chunk
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Combined result from all chunks
        """
        if self.dask_df is not None:
            # Use dask's built-in parallel processing
            result = func(self.dask_df, *args, **kwargs)
            return result.compute()

        # Process in chunks
        chunks = [self.df[i:i + self.chunk_size] for i in range(0, len(self.df), self.chunk_size)]

        # Process each chunk
        results = []
        for chunk in chunks:
            chunk_result = func(chunk, *args, **kwargs)
            results.append(chunk_result)

        # Combine results (the combination method depends on the function)
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], dict):
            # Merge dictionaries
            combined = {}
            for res in results:
                combined.update(res)
            return combined
        elif isinstance(results[0], (int, float)):
            # For numeric results, often we want the mean
            return np.mean(results)
        else:
            # Default: return list of results
            return results

    @timing_decorator
    def plot_feature_distribution(self, feature: str,
                                  by_disease: bool = False,
                                  bins: int = 30,
                                  figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot distribution of a feature with optimized performance for large datasets.

        Args:
            feature: Feature name to plot
            by_disease: Whether to plot distribution by disease category
            bins: Number of bins for histogram
            figsize: Figure size
        """
        if feature not in self.df.columns:
            raise ValueError(f"Feature {feature} not found in DataFrame")

        plt.figure(figsize=figsize)

        # Sample data if dataset is very large
        plot_df = self.df
        if len(self.df) > 100000:
            plot_df = self.df.sample(n=100000, random_state=42)
            print(f"Using a sample of 100,000 records for plotting {feature}")

        if pd.api.types.is_numeric_dtype(plot_df[feature]):
            if by_disease:
                # Plot numerical feature by disease
                for disease in plot_df['Disease'].unique():
                    disease_data = plot_df[plot_df['Disease'] == disease][feature].dropna()
                    sns.kdeplot(disease_data, label=disease)

                plt.title(f'Distribution of {feature} by Disease Type', fontsize=15)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.legend()
            else:
                # Simple histogram with KDE
                sns.histplot(plot_df[feature].dropna(), kde=True, bins=bins)
                plt.title(f'Distribution of {feature}', fontsize=15)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
        else:
            # For categorical features
            if by_disease:
                # Create a grouped count plot
                disease_counts = {}
                for disease in plot_df['Disease'].unique():
                    disease_data = plot_df[plot_df['Disease'] == disease][feature].value_counts(normalize=True)
                    disease_counts[disease] = disease_data

                disease_df = pd.DataFrame(disease_counts)
                disease_df.plot(kind='bar')
                plt.title(f'Distribution of {feature} by Disease Type', fontsize=15)
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('Proportion', fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(title='Disease')
            else:
                # Simple count plot
                sns.countplot(y=feature, data=plot_df)
                plt.title(f'Distribution of {feature}', fontsize=15)
                plt.xlabel('Count', fontsize=12)
                plt.ylabel(feature, fontsize=12)

        plt.tight_layout()
        plt.show()

    @timing_decorator
    def plot_all_distributions(self, by_disease: bool = False,
                               save_dir: Optional[str] = None) -> None:
        """
        Plot distributions for all features efficiently.

        Args:
            by_disease: Whether to plot distributions by disease
            save_dir: Directory to save figures (None for display only)
        """
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Plot disease distribution first
        if 'Disease' in self.df.columns:
            plt.figure(figsize=(12, 6))
            disease_counts = self.df['Disease'].value_counts()
            sns.barplot(x=disease_counts.index, y=disease_counts.values)
            plt.title('Distribution of Disease Types', fontsize=15)
            plt.xlabel('Disease', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()

            if save_dir:
                plt.savefig(f"{save_dir}/disease_distribution.png")
                plt.close()
            else:
                plt.show()

        # Plot all other features
        for col in self.df.columns:
            if col == 'Disease':
                continue

            try:
                print(f"Plotting distribution for {col}...")
                plt.figure(figsize=(12, 6))
                self.plot_feature_distribution(col, by_disease=by_disease)

                if save_dir:
                    plt.savefig(f"{save_dir}/{col}_distribution.png")
                    plt.close()
            except Exception as e:
                print(f"Error plotting {col}: {str(e)}")
                continue

    @timing_decorator
    def analyze_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical features efficiently.

        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')

        Returns:
            Correlation matrix DataFrame
        """
        # Get numerical columns
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # For large datasets, sample for faster computation
        if len(self.df) > 100000:
            print("Using a sample of 100,000 records for correlation analysis")
            corr_df = self.df.sample(n=100000, random_state=42)
        else:
            corr_df = self.df

        # Calculate correlation matrix
        corr_matrix = corr_df[num_cols].corr(method=method)
        return corr_matrix

    @timing_decorator
    def plot_correlation_heatmap(self, method: str = 'pearson',
                                 figsize: Tuple[int, int] = (12, 10),
                                 save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap for numerical features.

        Args:
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            figsize: Figure size
            save_path: Path to save the figure (None for display only)
        """
        corr_matrix = self.analyze_correlation_matrix(method=method)

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f")

        plt.title(f'Feature Correlation Heatmap ({method})', fontsize=15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @timing_decorator
    def analyze_feature_importance_by_disease(self) -> pd.DataFrame:
        """
        Analyze feature importance for distinguishing disease types.
        Uses ANOVA for numerical features.

        Returns:
            DataFrame with feature importance metrics
        """
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        results = []

        # For large datasets, sample for faster computation
        if len(self.df) > 100000:
            print("Using a sample of 100,000 records for feature importance analysis")
            analysis_df = self.df.sample(n=100000, random_state=42)
        else:
            analysis_df = self.df

        # Perform ANOVA for each numerical feature
        for col in numerical_cols:
            # Skip if Disease or ID column
            if col in ['Disease', 'ID']:
                continue

            disease_groups = analysis_df.groupby('Disease')[col].apply(list)

            try:
                f_stat, p_value = stats.f_oneway(*disease_groups)
                results.append({
                    'Feature': col,
                    'F-statistic': f_stat,
                    'p-value': p_value,
                    'Significant': p_value < 0.05
                })
            except Exception as e:
                print(f"Error analyzing {col}: {str(e)}")
                continue

        result_df = pd.DataFrame(results).sort_values('F-statistic', ascending=False)
        return result_df

    @timing_decorator
    def analyze_categorical_significance(self) -> pd.DataFrame:
        """
        Analyze significance of categorical features using Chi-square tests.

        Returns:
            DataFrame with Chi-square test results
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove Disease from columns to analyze
        if 'Disease' in categorical_cols:
            categorical_cols.remove('Disease')

        results = []

        # For large datasets, sample for faster computation
        if len(self.df) > 100000:
            print("Using a sample of 100,000 records for categorical significance analysis")
            analysis_df = self.df.sample(n=100000, random_state=42)
        else:
            analysis_df = self.df

        for col in categorical_cols:
            # Create contingency table
            contingency = pd.crosstab(analysis_df['Disease'], analysis_df[col])

            try:
                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                results.append({
                    'Feature': col,
                    'Chi-square': chi2,
                    'p-value': p,
                    'Significant': p < 0.05
                })
            except Exception as e:
                print(f"Error analyzing {col}: {str(e)}")
                continue

        result_df = pd.DataFrame(results).sort_values('Chi-square', ascending=False)
        return result_df

    def run_complete_eda(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete exploratory data analysis on the dataset efficiently.

        Args:
            save_dir: Directory to save figures (None for display only)

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Basic information
        print("Analyzing distributions...")
        results['distributions'] = self.analyze_all_distributions()

        # Plot distributions
        print("\nPlotting distributions...")
        self.plot_all_distributions(by_disease=False, save_dir=save_dir)

        # Correlation analysis
        print("\nAnalyzing feature correlations...")
        results['correlation_matrix'] = self.analyze_correlation_matrix()

        # Plot correlation heatmap
        print("\nPlotting correlation heatmap...")
        save_path = f"{save_dir}/correlation_heatmap.png" if save_dir else None
        self.plot_correlation_heatmap(save_path=save_path)

        # Feature importance
        print("\nAnalyzing feature importance...")
        results['feature_importance'] = self.analyze_feature_importance_by_disease()

        # Categorical significance
        print("\nAnalyzing categorical significance...")
        results['categorical_significance'] = self.analyze_categorical_significance()

        return results

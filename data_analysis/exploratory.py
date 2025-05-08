import pandas as pd
from scipy import stats


def analyze_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple analysis of feature importance using ANOVA for numerical features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with feature importance scores
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    results = []

    # Perform ANOVA for each numerical feature
    for col in numerical_cols:
        disease_groups = df.groupby('Disease')[col].apply(list)

        try:
            f_stat, p_value = stats.f_oneway(*disease_groups)
            results.append({
                'Feature': col,
                'F-statistic': f_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
        except:
            # Skip features that cause errors in ANOVA
            continue

    result_df = pd.DataFrame(results).sort_values('F-statistic', ascending=False)
    return result_df


def analyze_categorical_significance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze significance of categorical features using Chi-square tests.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Chi-square test results
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove Disease from columns to analyze
    if 'Disease' in categorical_cols:
        categorical_cols.remove('Disease')

    results = []

    for col in categorical_cols:
        # Create contingency table
        contingency = pd.crosstab(df['Disease'], df[col])

        try:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            results.append({
                'Feature': col,
                'Chi-square': chi2,
                'p-value': p,
                'Significant': p < 0.05
            })
        except:
            # Skip features that cause errors
            continue

    result_df = pd.DataFrame(results).sort_values('Chi-square', ascending=False)
    return result_df

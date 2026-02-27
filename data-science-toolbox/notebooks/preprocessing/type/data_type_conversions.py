"""
Data Type Conversion Helpers for Outlier Pipeline

These functions convert common string formats to numeric types
before outlier analysis. Use these BEFORE calling analyze_outliers_all().
"""

import pandas as pd
import numpy as np
import re


def currency_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert currency strings (e.g., '€45000', '$1,234.56') to numeric.
    
    Parameters:
    -----------
    series : pd.Series
        Series with currency strings
    
    Returns:
    --------
    pd.Series (numeric)
        Converted numeric series
    
    Examples:
    ---------
    >>> s = pd.Series(['€45000', '€62000', '€78000'])
    >>> currency_to_numeric(s)
    0    45000.0
    1    62000.0
    2    78000.0
    dtype: float64
    """
    if pd.api.types.is_numeric_dtype(series):
        return series  # Already numeric
    
    # Remove currency symbols and whitespace
    cleaned = series.astype(str).str.replace(r'[€$£¥₹\s]', '', regex=True)
    
    # Remove commas (thousands separator)
    cleaned = cleaned.str.replace(',', '', regex=False)
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(cleaned, errors='coerce')


def percentage_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert percentage strings (e.g., '8.5%', '10.2%') to numeric.
    
    Parameters:
    -----------
    series : pd.Series
        Series with percentage strings
    
    Returns:
    --------
    pd.Series (numeric)
        Converted numeric series (percentage value, not decimal)
    
    Examples:
    ---------
    >>> s = pd.Series(['8.5%', '10.2%', '12.0%'])
    >>> percentage_to_numeric(s)
    0     8.5
    1    10.2
    2    12.0
    dtype: float64
    """
    if pd.api.types.is_numeric_dtype(series):
        return series  # Already numeric
    
    # Remove % sign and whitespace
    cleaned = series.astype(str).str.replace(r'[%\s]', '', regex=True)
    
    # Convert to numeric
    return pd.to_numeric(cleaned, errors='coerce')


def prepare_data_for_outlier_analysis(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Convert string columns to numeric before outlier analysis.
    
    This function should be called BEFORE analyze_outliers_all().
    It converts currency and percentage strings to numeric types.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with potentially string columns
    config : dict
        Outlier config dict. This function uses it to identify
        which columns need conversion (based on detection_method).
    
    Returns:
    --------
    pd.DataFrame
        Copy with numeric conversions applied
    
    Example:
    --------
    >>> data = pd.read_csv('loan_data.csv')
    >>> # At this point, customer_income and loan_int_rate are strings
    >>> 
    >>> data_clean = prepare_data_for_outlier_analysis(data, OUTLIER_CONFIG)
    >>> # Now they're numeric
    >>> 
    >>> outlier_analysis = analyze_outliers_all(data_clean, OUTLIER_CONFIG)
    """
    df_clean = df.copy()
    
    # Define which columns need which conversion
    # You can customize this based on your data
    currency_columns = [
        'customer_income',
        'loan_amnt',
    ]
    
    percentage_columns = [
        'loan_int_rate',
    ]
    
    # Convert currency columns
    for col in currency_columns:
        if col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = currency_to_numeric(df_clean[col])
                print(f"✓ Converted {col} from currency string to numeric")
    
    # Convert percentage columns
    for col in percentage_columns:
        if col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = percentage_to_numeric(df_clean[col])
                print(f"✓ Converted {col} from percentage string to numeric")
    
    return df_clean


# Example: How to use in your notebook
if __name__ == "__main__":
    # Before outlier analysis
    data = pd.read_csv('your_data.csv')
    
    # Step 0: Convert string columns to numeric
    data = prepare_data_for_outlier_analysis(data, OUTLIER_CONFIG)
    
    # Step 1: Analyze outliers
    outlier_analysis = analyze_outliers_all(data, OUTLIER_CONFIG)
    
    # Step 2: Treat outliers
    data_clean, audit = handle_outliers(data, OUTLIER_CONFIG, outlier_analysis)

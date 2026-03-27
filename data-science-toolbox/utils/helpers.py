"""
ds-toolkit/utils/helpers.py
Reusable helper functions for use across any project.
Import with: from utils.helpers import timer, print_section
"""
import time
import functools
import pandas as pd
import numpy as np


def timer(func):
    """Decorator: prints how long a function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_section(title):
    """Print a visible section divider."""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def df_summary(df):
    """One-stop DataFrame summary: shape, dtypes, nulls, uniques."""
    return pd.DataFrame({
        'dtype':      df.dtypes,
        'null_pct':   (df.isnull().mean() * 100).round(2),
        'n_unique':   df.nunique(),
        'sample_val': df.iloc[0] if len(df) > 0 else None
    })


def reduce_mem_usage(df):
    """Downcast numeric columns to reduce DataFrame memory footprint."""
    for col in df.select_dtypes(include=[np.number]).columns:
        c_min, c_max = df[col].min(), df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min <= c_min and c_max <= np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype); break
        else:
            for dtype in [np.float16, np.float32]:
                if np.finfo(dtype).min <= c_min and c_max <= np.finfo(dtype).max:
                    df[col] = df[col].astype(dtype); break
    return df


def cramers_v(x, y):
    """Cramér's V — association between two categorical variables (0=none, 1=perfect)."""
    from scipy.stats import chi2_contingency
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    return np.sqrt(phi2 / min(k-1, r-1))

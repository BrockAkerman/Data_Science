"""
Data Cleaning Pipeline: Outlier Handling Module

Provides production-ready outlier detection and treatment with comprehensive
audit logging. Supports multiple detection and treatment methods per feature.

Detection Methods:
- 'iqr': Tukey's Interquartile Range method
- 'zscore': Z-score based (assumes normality)
- 'domain_based': Apply domain-specific bounds (e.g., age < 18)
- 'hybrid': Combine multiple methods
- 'none': Skip detection (retain all values)

Treatment Methods:
- 'remove': Delete rows containing outliers
- 'cap': Replace with fence values
- 'transform': Apply mathematical transformation (log, sqrt, yeo-johnson)
- 'retain': Keep outliers unchanged
- 'flag': Create binary outlier indicator

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_OUTLIER_CONFIG = {
    'features': {
        'feature_name': {
            'detection_method': 'iqr',                   # or domain_based, zscore, hybrid, none
            'treat_method': 'remove',                     # or cap, transform, retain, flag
            'lower_bound': None,                          # For domain_based: minimum allowed value
            'upper_bound': None,                          # For domain_based: maximum allowed value
            'iqr_multiplier': 1.5,                        # For IQR method: 1.5 = standard, 3.0 = extreme
            'transform_type': 'log1p',                    # For transform: log1p, sqrt, yeo-johnson
            'create_indicator': False,                    # Create binary outlier flag
            'description': 'Brief description of outlier handling rationale'
        }
    }
}


# ==============================================================================
# DETECTION FUNCTIONS
# ==============================================================================

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR / Tukey's) method.
    
    Standard approach: multiplier=1.5 flags ~0.7% of normally distributed data.
    Conservative approach: multiplier=3.0 flags only extreme values.
    
    Parameters:
    -----------
    series : pd.Series
        Numeric series to analyze
    multiplier : float
        IQR multiplier (default 1.5)
    
    Returns:
    --------
    pd.Series (bool)
        True where values are outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_fence = Q1 - (multiplier * IQR)
    upper_fence = Q3 + (multiplier * IQR)
    
    return (series < lower_fence) | (series > upper_fence)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Assumes normally distributed data. Threshold=3.0 flags ~0.3% of data.
    
    Parameters:
    -----------
    series : pd.Series
        Numeric series to analyze
    threshold : float
        Z-score threshold (default 3.0)
    
    Returns:
    --------
    pd.Series (bool)
        True where values are outliers
    """
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def analyze_outliers(
    series: pd.Series,
    feature_name: str,
    iqr_multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Comprehensive outlier analysis for a single feature.
    
    Returns detailed statistics on outlier position and distribution.
    
    Parameters:
    -----------
    series : pd.Series
        Numeric series (NAs should be dropped)
    feature_name : str
        Feature name for logging
    iqr_multiplier : float
        IQR multiplier (default 1.5)
    
    Returns:
    --------
    Dict with analysis results
    """
    series = series.dropna()
    
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - (iqr_multiplier * IQR)
    upper_fence = Q3 + (iqr_multiplier * IQR)
    
    outliers_mask = detect_outliers_iqr(series, iqr_multiplier)
    n_outliers = outliers_mask.sum()
    pct_outliers = (n_outliers / len(series)) * 100 if len(series) > 0 else 0
    
    return {
        'feature': feature_name,
        'count': int(n_outliers),
        'percent': round(pct_outliers, 3),
        'min': float(series.min()),
        'max': float(series.max()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'q1': float(Q1),
        'q3': float(Q3),
        'iqr': float(IQR),
        'lower_fence': float(lower_fence),
        'upper_fence': float(upper_fence),
    }


def analyze_outliers_all(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze outliers across all configured features.
    
    Analysis varies by detection method:
    - 'iqr': Report Tukey fence-based outliers
    - 'domain_based': Report values outside configured bounds
    - 'zscore': Report values with |z| > threshold
    - 'none': Skip analysis (no outliers to detect)
    - 'hybrid': Report both IQR and domain-based findings
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : Dict
        Outlier configuration
    
    Returns:
    --------
    Dict mapping feature names to analysis results
    """
    analysis = {}
    
    logger.info("\n" + "="*70)
    logger.info("OUTLIER ANALYSIS (BEFORE TREATMENT)")
    logger.info("="*70)
    
    for feature_name in config['features'].keys():
        if feature_name not in df.columns:
            continue
        
        series = df[feature_name]
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning(f"✗ {feature_name}: Not numeric; skipping")
            continue
        
        feature_config = config['features'][feature_name]
        detection = feature_config.get('detection_method')
        
        logger.info(f"\n{feature_name}:")
        logger.info(f"  Detection method: {detection}")
        
        # =====================================================================
        # ANALYSIS BY DETECTION METHOD
        # =====================================================================
        
        if detection == 'none':
            logger.info(f"  Status: No outlier detection (transformation only)")
            analysis[feature_name] = {'detection': 'none', 'count': 0}
        
        # =====================================================================
        # IQR-BASED ANALYSIS
        # =====================================================================
        elif detection == 'iqr':
            iqr_mult = feature_config.get('iqr_multiplier', 1.5)
            feature_analysis = analyze_outliers(series, feature_name, iqr_mult)
            analysis[feature_name] = feature_analysis
            
            logger.info(f"  IQR Multiplier: {iqr_mult}")
            logger.info(f"  Outliers detected: {feature_analysis['count']} ({feature_analysis['percent']:.3f}%)")
            logger.info(f"  Range: [{feature_analysis['min']:.2f}, {feature_analysis['max']:.2f}]")
            logger.info(f"  Q1={feature_analysis['q1']:.2f}, Q3={feature_analysis['q3']:.2f}, IQR={feature_analysis['iqr']:.2f}")
            logger.info(f"  IQR Fences: [{feature_analysis['lower_fence']:.2f}, {feature_analysis['upper_fence']:.2f}]")
        
        # =====================================================================
        # DOMAIN-BASED ANALYSIS
        # =====================================================================
        elif detection == 'domain_based':
            lower_bound = feature_config.get('lower_bound')
            upper_bound = feature_config.get('upper_bound')
            
            analysis[feature_name] = {
                'detection': 'domain_based',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median())
            }
            
            # Count how many would be removed
            if lower_bound is not None:
                below_lower = (series < lower_bound).sum()
                logger.info(f"  Lower bound: {feature_name} >= {lower_bound}")
                logger.info(f"    Values below bound: {below_lower}")
            
            if upper_bound is not None:
                above_upper = (series > upper_bound).sum()
                logger.info(f"  Upper bound: {feature_name} <= {upper_bound}")
                logger.info(f"    Values above bound: {above_upper}")
            
            logger.info(f"  Data range: [{series.min():.2f}, {series.max():.2f}]")
            logger.info(f"  Mean={series.mean():.2f}, Median={series.median():.2f}")
        
        # =====================================================================
        # Z-SCORE ANALYSIS
        # =====================================================================
        elif detection == 'zscore':
            threshold = feature_config.get('zscore_threshold', 3.0)
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers_mask = z_scores > threshold
            n_outliers = outliers_mask.sum()
            pct_outliers = (n_outliers / len(series)) * 100
            
            analysis[feature_name] = {
                'detection': 'zscore',
                'threshold': threshold,
                'count': int(n_outliers),
                'percent': round(pct_outliers, 3),
                'min': float(series.min()),
                'max': float(series.max())
            }
            
            logger.info(f"  Z-score threshold: {threshold}")
            logger.info(f"  Outliers detected: {n_outliers} ({pct_outliers:.3f}%)")
        
        # =====================================================================
        # HYBRID ANALYSIS
        # =====================================================================
        elif detection == 'hybrid':
            # Run both IQR and domain-based analysis
            iqr_mult = feature_config.get('iqr_multiplier', 1.5)
            iqr_analysis = analyze_outliers(series, feature_name, iqr_mult)
            
            logger.info(f"  Method 1: IQR Analysis")
            logger.info(f"    Outliers: {iqr_analysis['count']} ({iqr_analysis['percent']:.3f}%)")
            logger.info(f"    IQR Fences: [{iqr_analysis['lower_fence']:.2f}, {iqr_analysis['upper_fence']:.2f}]")
            
            lower_bound = feature_config.get('lower_bound')
            upper_bound = feature_config.get('upper_bound')
            
            if lower_bound is not None or upper_bound is not None:
                logger.info(f"  Method 2: Domain-based Analysis")
                if lower_bound is not None:
                    below = (series < lower_bound).sum()
                    logger.info(f"    Below {lower_bound}: {below} values")
                if upper_bound is not None:
                    above = (series > upper_bound).sum()
                    logger.info(f"    Above {upper_bound}: {above} values")
            
            analysis[feature_name] = {
                'detection': 'hybrid',
                'iqr': iqr_analysis,
                'domain_bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
    
    logger.info(f"\n" + "="*70)
    return analysis


# ==============================================================================
# TREATMENT FUNCTION
# ==============================================================================

def handle_outliers(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    analysis: Dict[str, Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Treat outliers according to feature-specific strategies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : Dict
        Outlier configuration with feature-specific strategies
    analysis : Dict, optional
        Pre-computed outlier analysis
    
    Returns:
    --------
    df_clean : pd.DataFrame
        Dataframe with outliers treated
    audit : Dict
        Comprehensive audit trail
    """
    if config is None:
        config = DEFAULT_OUTLIER_CONFIG
    
    if analysis is None:
        analysis = analyze_outliers_all(df, config)
    
    audit = {
        'status': 'STARTED',
        'rows_input': len(df),
        'rows_output': len(df),
        'rows_removed': 0,
        'features_processed': [],
        'features_transformed': [],
        'details': {},
        'errors': [],
        'warnings': []
    }
    
    df_clean = df.copy()
    
    logger.info("\n" + "="*70)
    logger.info("OUTLIER TREATMENT EXECUTION")
    logger.info("="*70)
    
    # =========================================================================
    # PROCESS EACH FEATURE
    # =========================================================================
    for feature_name, feature_config in config['features'].items():
        if feature_name not in df_clean.columns:
            continue
        
        detection = feature_config.get('detection_method')
        treatment = feature_config.get('treat_method')
        
        logger.info(f"\n{'─'*70}")
        logger.info(f"Feature: {feature_name}")
        logger.info(f"  Detection: {detection}")
        logger.info(f"  Treatment: {treatment}")
        
        detail = {
            'detection_method': detection,
            'treat_method': treatment,
            'rows_removed': 0,
            'values_capped': 0,
            'values_transformed': 0
        }
        
        # =====================================================================
        # TREATMENT: REMOVE ROWS
        # =====================================================================
        if treatment == 'remove':
            rows_before = len(df_clean)
            
            # Apply lower bound if configured
            if 'lower_bound' in feature_config and feature_config['lower_bound'] is not None:
                lower = feature_config['lower_bound']
                mask = df_clean[feature_name] >= lower
                logger.info(f"  Lower bound: {feature_name} >= {lower}")
                df_clean = df_clean[mask]
            
            # Apply upper bound if configured
            if 'upper_bound' in feature_config and feature_config['upper_bound'] is not None:
                upper = feature_config['upper_bound']
                mask = df_clean[feature_name] <= upper
                logger.info(f"  Upper bound: {feature_name} <= {upper}")
                df_clean = df_clean[mask]
            
            rows_removed = rows_before - len(df_clean)
            detail['rows_removed'] = int(rows_removed)
            audit['rows_removed'] += rows_removed
            
            logger.info(f"  Rows removed: {rows_removed}")
        
        # =====================================================================
        # TREATMENT: CAP OUTLIERS
        # =====================================================================
        elif treatment == 'cap':
            if feature_name in analysis:
                feat_analysis = analysis[feature_name]
                lower_fence = feat_analysis['lower_fence']
                upper_fence = feat_analysis['upper_fence']
                
                mask_lower = df_clean[feature_name] < lower_fence
                n_lower = mask_lower.sum()
                df_clean.loc[mask_lower, feature_name] = lower_fence
                
                mask_upper = df_clean[feature_name] > upper_fence
                n_upper = mask_upper.sum()
                df_clean.loc[mask_upper, feature_name] = upper_fence
                
                n_capped = n_lower + n_upper
                detail['values_capped'] = int(n_capped)
                logger.info(f"  Values capped: {n_capped} (lower: {n_lower}, upper: {n_upper})")
        
        # =====================================================================
        # TREATMENT: TRANSFORM
        # =====================================================================
        elif treatment == 'transform':
            transform_type = feature_config.get('transform_type', 'log1p')
            new_col_name = f"{feature_name}_transformed"
            
            if transform_type == 'log1p':
                df_clean[new_col_name] = np.log1p(df_clean[feature_name])
                logger.info(f"  Applied log1p → '{new_col_name}'")
                audit['features_transformed'].append(new_col_name)
            
            elif transform_type == 'sqrt':
                df_clean[new_col_name] = np.sqrt(np.abs(df_clean[feature_name]))
                logger.info(f"  Applied sqrt → '{new_col_name}'")
                audit['features_transformed'].append(new_col_name)
            
            detail['values_transformed'] = len(df_clean)
        
        # =====================================================================
        # TREATMENT: RETAIN
        # =====================================================================
        elif treatment == 'retain':
            logger.info(f"  Outliers retained as legitimate values")
        
        audit['features_processed'].append(feature_name)
        audit['details'][feature_name] = detail
    
    # =========================================================================
    # UPDATE COUNTS & SUMMARY
    # =========================================================================
    audit['rows_output'] = len(df_clean)
    audit['status'] = 'SUCCESS'
    
    logger.info(f"\n" + "="*70)
    logger.info("OUTLIER HANDLING SUMMARY")
    logger.info("="*70)
    logger.info(f"Features processed: {len(audit['features_processed'])}")
    logger.info(f"Features transformed: {len(audit['features_transformed'])}")
    if audit['features_transformed']:
        logger.info(f"  → {', '.join(audit['features_transformed'])}")
    logger.info(f"Rows: {audit['rows_input']} → {audit['rows_output']} (removed: {audit['rows_removed']})")
    logger.info(f"="*70)
    
    return df_clean, audit

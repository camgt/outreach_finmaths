"""
Data loading utilities for historical market data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import streamlit as st


@st.cache_data
def load_config():
    """
    Load the data configuration file.
    
    Returns:
    --------
    dict : Configuration data
    """
    config_path = Path(__file__).parent.parent / "data" / "data_config.json"
    
    if not config_path.exists():
        st.error(f"Configuration file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_market_data(period_key):
    """
    Load historical market data for a specific period.
    
    Parameters:
    -----------
    period_key : str
        Key from data_config.json (e.g., 'crisis_2008')
    
    Returns:
    --------
    pandas.DataFrame : Market data with Date index
    dict : Period metadata
    """
    config = load_config()
    
    if period_key not in config['periods']:
        st.error(f"Period '{period_key}' not found in configuration")
        return None, None
    
    period_info = config['periods'][period_key]
    data_path = Path(__file__).parent.parent / period_info['file']
    
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Run: python scripts/download_data.py")
        return None, None
    
    # Load CSV
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    return df, period_info


@st.cache_data
def load_all_periods():
    """
    Load all available market periods.
    
    Returns:
    --------
    dict : Dictionary mapping period_key to (dataframe, metadata)
    """
    config = load_config()
    all_data = {}
    
    for period_key in config['periods'].keys():
        df, info = load_market_data(period_key)
        if df is not None:
            all_data[period_key] = (df, info)
    
    return all_data


def get_periods_by_regime(regime):
    """
    Get all periods matching a specific regime type.
    
    Parameters:
    -----------
    regime : str
        'crisis', 'bull', 'stability', or 'mixed'
    
    Returns:
    --------
    list : List of (period_key, dataframe, metadata) tuples
    """
    config = load_config()
    matching_periods = []
    
    for period_key, period_info in config['periods'].items():
        if period_info['regime'] == regime:
            df, info = load_market_data(period_key)
            if df is not None:
                matching_periods.append((period_key, df, info))
    
    return matching_periods


def extract_price_series(df, column='Close', max_points=None):
    """
    Extract price series from dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Market data
    column : str
        Column name to extract ('Close', 'Adj Close', etc.)
    max_points : int, optional
        Maximum number of points to return (for performance)
    
    Returns:
    --------
    numpy.array : Price series
    pandas.DatetimeIndex : Corresponding dates
    """
    if column not in df.columns:
        # Try alternative column names
        if 'Adj Close' in df.columns:
            column = 'Adj Close'
        elif 'Close' in df.columns:
            column = 'Close'
        else:
            raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Extract and ensure numeric values
    prices = pd.to_numeric(df[column].dropna(), errors='coerce').values
    dates = df[column].dropna().index
    
    # Remove any NaN values that resulted from coercion
    valid_mask = ~np.isnan(prices)
    prices = prices[valid_mask]
    dates = dates[valid_mask]
    
    # Subsample if needed
    if max_points and len(prices) > max_points:
        step = len(prices) // max_points
        prices = prices[::step]
        dates = dates[::step]
    
    return prices, dates


def get_period_summary():
    """
    Get a summary of all available periods for display.
    
    Returns:
    --------
    pandas.DataFrame : Summary table
    """
    config = load_config()
    
    summary_data = []
    for period_key, period_info in config['periods'].items():
        data_path = Path(__file__).parent.parent / period_info['file']
        exists = data_path.exists()
        
        summary_data.append({
            'Period': period_info['name'],
            'Regime': period_info['regime'].capitalize(),
            'Dates': f"{period_info['start']} to {period_info['end']}",
            'Ticker': period_info['ticker'],
            'Data Available': '✅' if exists else '❌'
        })
    
    return pd.DataFrame(summary_data)

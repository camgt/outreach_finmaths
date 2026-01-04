"""
Financial time series generation and modeling utilities.
Includes geometric Brownian motion and other stochastic processes.
"""

import numpy as np
import pandas as pd


def generate_gbm(
    S0=100,
    mu=0.05,
    sigma=0.2,
    T=1.0,
    steps=252,
    seed=None
):
    """
    Generate a price path using Geometric Brownian Motion (GBM).
    
    The stochastic differential equation is:
    dS = μ * S * dt + σ * S * dW
    
    Parameters:
    -----------
    S0 : float
        Initial price
    mu : float
        Drift (annualized expected return)
    sigma : float
        Volatility (annualized standard deviation)
    T : float
        Time horizon in years
    steps : int
        Number of time steps
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    numpy.array : Price path of length (steps + 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / steps
    
    # Generate random shocks
    dW = np.random.normal(0, np.sqrt(dt), steps)
    
    # Initialize price array
    prices = np.zeros(steps + 1)
    prices[0] = S0
    
    # Generate path
    for t in range(1, steps + 1):
        prices[t] = prices[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW[t-1]
        )
    
    return prices


def generate_gbm_with_drift(
    S0=100,
    drift_type='none',
    sigma=0.2,
    T=1.0,
    steps=252,
    seed=None
):
    """
    Generate GBM with different drift patterns.
    
    Parameters:
    -----------
    S0 : float
        Initial price
    drift_type : str
        'none', 'up', 'down', 'stable'
    sigma : float
        Volatility
    T : float
        Time horizon in years
    steps : int
        Number of time steps
    seed : int, optional
        Random seed
    
    Returns:
    --------
    numpy.array : Price path
    """
    drift_map = {
        'none': 0.0,
        'stable': 0.02,
        'up': 0.15,
        'down': -0.15,
        'bull': 0.20,
        'bear': -0.20,
    }
    
    mu = drift_map.get(drift_type, 0.0)
    return generate_gbm(S0, mu, sigma, T, steps, seed)


def generate_mean_reverting(
    S0=100,
    theta=0.5,
    mu=100,
    sigma=0.2,
    T=1.0,
    steps=252,
    seed=None
):
    """
    Generate mean-reverting price series (Ornstein-Uhlenbeck process).
    
    Parameters:
    -----------
    S0 : float
        Initial price
    theta : float
        Speed of mean reversion
    mu : float
        Long-term mean
    sigma : float
        Volatility
    T : float
        Time horizon in years
    steps : int
        Number of time steps
    seed : int, optional
        Random seed
    
    Returns:
    --------
    numpy.array : Price path
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / steps
    prices = np.zeros(steps + 1)
    prices[0] = S0
    
    for t in range(1, steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        prices[t] = prices[t-1] + theta * (mu - prices[t-1]) * dt + sigma * dW
    
    return prices


def add_trend(prices, trend_strength=0.001):
    """
    Add a linear trend to a price series.
    
    Parameters:
    -----------
    prices : numpy.array
        Original price series
    trend_strength : float
        Strength of trend per step
    
    Returns:
    --------
    numpy.array : Price series with trend
    """
    n = len(prices)
    trend = np.linspace(0, trend_strength * n, n)
    return prices + trend * prices[0]


def add_jumps(prices, jump_prob=0.02, jump_size_mean=0.05, jump_size_std=0.02, seed=None):
    """
    Add random jumps (discontinuities) to a price series.
    
    Parameters:
    -----------
    prices : numpy.array
        Original price series
    jump_prob : float
        Probability of a jump at each time step
    jump_size_mean : float
        Mean jump size (as fraction of price)
    jump_size_std : float
        Standard deviation of jump size
    seed : int, optional
        Random seed
    
    Returns:
    --------
    numpy.array : Price series with jumps
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(prices)
    result = prices.copy()
    
    for t in range(1, n):
        if np.random.random() < jump_prob:
            jump = np.random.normal(jump_size_mean, jump_size_std)
            result[t:] *= (1 + jump)
    
    return result


def match_moments(series1, series2):
    """
    Scale series2 to match the mean and standard deviation of series1.
    
    Parameters:
    -----------
    series1 : numpy.array
        Target series (to match)
    series2 : numpy.array
        Series to transform
    
    Returns:
    --------
    numpy.array : Transformed series2
    """
    mean1, std1 = np.mean(series1), np.std(series1)
    mean2, std2 = np.mean(series2), np.std(series2)
    
    if std2 == 0:
        return series2
    
    # Standardize series2 and rescale to match series1
    standardized = (series2 - mean2) / std2
    matched = standardized * std1 + mean1
    
    return matched


def calculate_returns(prices):
    """
    Calculate log returns from price series.
    
    Parameters:
    -----------
    prices : numpy.array or pandas.Series
        Price series
    
    Returns:
    --------
    numpy.array : Log returns
    """
    prices = np.array(prices)
    return np.diff(np.log(prices))


def calculate_volatility(prices, window=20):
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Parameters:
    -----------
    prices : numpy.array
        Price series
    window : int
        Rolling window size
    
    Returns:
    --------
    numpy.array : Rolling volatility
    """
    returns = calculate_returns(prices)
    
    if len(returns) < window:
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
    return rolling_vol.values

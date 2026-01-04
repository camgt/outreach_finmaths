"""
Option Mathematics Utilities
Payoff calculations and static replication tools.
"""
import numpy as np

def call_payoff(S, K):
    """Calculate the payoff of a European Call option at expiration."""
    return np.maximum(S - K, 0)

def put_payoff(S, K):
    """Calculate the payoff of a European Put option at expiration."""
    return np.maximum(K - S, 0)

def portfolio_payoff(S, positions):
    """
    Calculate the total payoff of a portfolio of instruments.
    
    Parameters:
    -----------
    S : array
        Stock prices at expiration
    positions : list of dicts
        Each dict has 'type' (call/put/cash/stock), 'K' (strike), 'q' (quantity)
    
    Returns:
    --------
    array : Total portfolio payoff at each stock price
    """
    total = np.zeros_like(S, dtype=float)
    for pos in positions:
        if pos['type'] == 'call':
            total += pos['q'] * call_payoff(S, pos['K'])
        elif pos['type'] == 'put':
            total += pos['q'] * put_payoff(S, pos['K'])
        elif pos['type'] == 'cash':
            total += pos['q'] * np.ones_like(S)
        elif pos['type'] == 'stock':
            total += pos['q'] * S
    return total

def get_named_payoff(name, S, params=None):
    """
    Get the payoff profile for a named exotic option.
    
    Parameters:
    -----------
    name : str
        Name of the option (digital, power, strangle, etc.)
    S : array
        Stock prices
    params : dict
        Parameters specific to the option type
        
    Returns:
    --------
    array : Payoff profile
    """
    if params is None:
        params = {}
    
    K = params.get('K', 100)
    K1 = params.get('K1', 90)
    K2 = params.get('K2', 110)
    
    if name == 'digital_call':
        # Pays $1 if S > K, else $0
        return np.where(S > K, 1.0, 0.0)
    
    elif name == 'digital_put':
        # Pays $1 if S < K, else $0
        return np.where(S < K, 1.0, 0.0)
    
    elif name == 'strangle':
        # Long put at K1, long call at K2
        return put_payoff(S, K1) + call_payoff(S, K2)
    
    elif name == 'butterfly':
        # 1 call at K1, -2 calls at K, 1 call at K2
        Km = params.get('Km', K)
        return call_payoff(S, K1) - 2*call_payoff(S, Km) + call_payoff(S, K2)
    
    elif name == 'power_call':
        # Pays (S-K)² if S > K
        return np.maximum(S - K, 0)**2 / 10  # Scaled for visibility
    
    elif name == 'collar':
        # Long stock + long put at K1 + short call at K2
        return S + put_payoff(S, K1) - call_payoff(S, K2)
    
    else:
        raise ValueError(f"Unknown payoff type: {name}")

def compute_static_replication(target_payoff, S, available_strikes, option_type='call'):
    """
    Compute the optimal static replication using available strikes.
    Uses finite differences to approximate the second derivative.
    
    Parameters:
    -----------
    target_payoff : array
        The payoff we want to replicate
    S : array
        Stock price grid
    available_strikes : list
        Available strike prices
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    dict : Replication weights for each strike
    """
    # This is a simplified version - in practice you'd use the 
    # Breeden-Litzenberger formula: weight(K) = f''(K)
    
    # For now, we'll use a simple approach: 
    # Solve the linear system to match the payoff at the available strikes
    n_strikes = len(available_strikes)
    
    # Build the design matrix: each column is a call/put payoff at a strike
    A = np.zeros((len(S), n_strikes))
    for i, K in enumerate(available_strikes):
        if option_type == 'call':
            A[:, i] = call_payoff(S, K)
        else:
            A[:, i] = put_payoff(S, K)
    
    # Solve least squares: min ||Ax - target||²
    weights, residual, rank, s = np.linalg.lstsq(A, target_payoff, rcond=None)
    
    return {K: w for K, w in zip(available_strikes, weights)}

def get_option_description(name):
    """Get a description of the named option."""
    descriptions = {
        'digital_call': {
            'name': 'Digital Call (Cash-or-Nothing)',
            'description': 'Pays a fixed amount ($1) if the stock ends above the strike, otherwise pays nothing.',
            'use_case': 'Used in binary bets on market direction. Common in structured products and FX markets.',
        },
        'strangle': {
            'name': 'Long Strangle',
            'description': 'Profits from large price moves in either direction. Combination of out-of-the-money call and put.',
            'use_case': 'Volatility play - used when expecting a big move but uncertain about direction (e.g., before earnings announcements).',
        },
        'butterfly': {
            'name': 'Butterfly Spread',
            'description': 'Profits when the stock stays near a target price. Limited risk on both sides.',
            'use_case': 'Range-bound strategy - used when expecting low volatility. Cheap way to bet on stability.',
        },
        'power_call': {
            'name': 'Power Call (Squared Payoff)',
            'description': 'Payoff grows quadratically with the stock price above strike.',
            'use_case': 'Provides leveraged exposure to upside. Used in structured notes for enhanced returns.',
        },
    }
    return descriptions.get(name, {
        'name': name,
        'description': 'Custom payoff profile',
        'use_case': 'Various hedging and speculation strategies',
    })

"""
Kelly Criterion calculations for optimal betting strategies.
"""

import numpy as np


def kelly_fraction(win_prob, win_multiplier, loss_multiplier=1.0):
    """
    Calculate the optimal Kelly fraction for betting with asymmetric payoffs.
    
    Parameters:
    -----------
    win_prob : float
        Probability of winning (0 to 1)
    win_multiplier : float
        How much you gain per unit bet when you win (e.g., 1.0 means you double your bet)
    loss_multiplier : float
        How much you lose per unit bet when you lose (default 1.0 means you lose your bet)
    
    Returns:
    --------
    float : Optimal fraction of capital to bet (0 to 1)
    
    Formula:
    --------
    f* = (p * w - q * l) / (w * l)
    where p = win probability, q = 1-p, w = win multiplier, l = loss multiplier
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0
    
    if win_multiplier <= 0 or loss_multiplier <= 0:
        return 0.0
    
    q = 1 - win_prob
    
    # Generalized Kelly formula for asymmetric payoffs
    kelly = (win_prob * win_multiplier - q * loss_multiplier) / (win_multiplier * loss_multiplier)
    
    # Kelly fraction should be between 0 and 1
    return max(0.0, min(1.0, kelly))


def simulate_betting_strategy(
    initial_capital,
    bet_fraction,
    n_bets,
    win_prob,
    win_multiplier,
    loss_multiplier=1.0,
    seed=None
):
    """
    Simulate a betting strategy over multiple bets.
    
    Parameters:
    -----------
    initial_capital : float
        Starting capital
    bet_fraction : float
        Fraction of current capital to bet each round (0 to 1)
    n_bets : int
        Number of bets to simulate
    win_prob : float
        Probability of winning each bet (0 to 1)
    win_multiplier : float
        Multiplier for winnings (e.g., 2.0 means you gain 2x your bet)
    loss_multiplier : float
        Multiplier for losses (default 1.0 means you lose your entire bet)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    list : Capital history over time (length = n_bets + 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    capital = initial_capital
    history = [capital]
    
    for _ in range(n_bets):
        if capital <= 0:
            history.append(0)
            continue
        
        # Calculate bet amount
        bet = capital * bet_fraction
        
        # Determine outcome
        if np.random.random() < win_prob:
            # Win
            capital += bet * win_multiplier
        else:
            # Lose
            capital -= bet * loss_multiplier
        
        # Ensure capital doesn't go negative
        capital = max(0, capital)
        history.append(capital)
    
    return history


def compare_strategies(
    initial_capital,
    n_bets,
    win_prob,
    win_multiplier,
    loss_multiplier=1.0,
    n_simulations=100,
    seed=None
):
    """
    Compare multiple betting strategies with repeated simulations.
    
    Parameters:
    -----------
    initial_capital : float
        Starting capital
    n_bets : int
        Number of bets per simulation
    win_prob : float
        Probability of winning each bet
    win_multiplier : float
        Multiplier for winnings
    loss_multiplier : float
        Multiplier for losses
    n_simulations : int
        Number of simulations to run for each strategy
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Results for different strategies
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate optimal Kelly fraction
    optimal_kelly = kelly_fraction(win_prob, win_multiplier, loss_multiplier)
    
    strategies = {
        "Kelly Optimal": optimal_kelly,
        "Half Kelly": optimal_kelly * 0.5,
        "Fixed 10%": 0.10,
        "Fixed 25%": 0.25,
        "Double Kelly": optimal_kelly * 2.0,
    }
    
    results = {}
    
    for strategy_name, bet_frac in strategies.items():
        all_simulations = []
        
        for _ in range(n_simulations):
            history = simulate_betting_strategy(
                initial_capital,
                bet_frac,
                n_bets,
                win_prob,
                win_multiplier,
                loss_multiplier,
                seed=None  # Different random outcomes for each sim
            )
            all_simulations.append(history)
        
        # Calculate statistics
        final_capitals = [sim[-1] for sim in all_simulations]
        
        results[strategy_name] = {
            "bet_fraction": bet_frac,
            "simulations": all_simulations,
            "mean_final": np.mean(final_capitals),
            "median_final": np.median(final_capitals),
            "std_final": np.std(final_capitals),
            "min_final": np.min(final_capitals),
            "max_final": np.max(final_capitals),
            "bankruptcy_rate": sum(1 for x in final_capitals if x < initial_capital * 0.01) / n_simulations,
        }
    
    return results, optimal_kelly

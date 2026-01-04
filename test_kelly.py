"""
Test Kelly criterion calculation
"""

import numpy as np
from utils.kelly_calculator import kelly_fraction, simulate_betting_strategy

# Test scenario: 55% win probability, even money bet
win_prob = 0.55
win_mult = 1.0
loss_mult = 1.0

kelly = kelly_fraction(win_prob, win_mult, loss_mult)
print(f"Kelly fraction: {kelly:.4f} ({kelly*100:.2f}%)")

# Test different fractions around Kelly
test_fractions = np.linspace(0.05, 0.20, 16)  # Test from 5% to 20%
n_sims = 2000
n_bets = 100

print(f"\nTesting {n_sims} simulations with {n_bets} bets each:")
print("-" * 70)

results = []

for frac in test_fractions:
    finals = []
    for _ in range(n_sims):
        history = simulate_betting_strategy(
            1000, frac, n_bets, win_prob, win_mult, loss_mult, seed=None
        )
        finals.append(history[-1])
    
    mean_final = np.mean(finals)
    median_final = np.median(finals)
    
    # Calculate geometric mean growth rate
    log_returns = [np.log(f/1000) if f > 0 else -np.inf for f in finals]
    valid_log_returns = [r for r in log_returns if r != -np.inf]
    geo_mean_growth = np.mean(valid_log_returns) / n_bets if valid_log_returns else -np.inf
    
    results.append((frac, geo_mean_growth))
    
    print(f"Fraction: {frac:.4f} ({frac*100:.2f}%)  Geo mean growth: {geo_mean_growth:.7f}")

print("\n" + "=" * 70)
best_frac, best_growth = max(results, key=lambda x: x[1])
print(f"BEST: {best_frac:.4f} ({best_frac*100:.2f}%) with growth rate {best_growth:.7f}")
print(f"Kelly calculated: {kelly:.4f} ({kelly*100:.2f}%)")
print(f"Ratio: Best/Kelly = {best_frac/kelly:.3f}x")

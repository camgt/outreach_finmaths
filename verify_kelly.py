"""
Verify Kelly criterion for the specific case:
win_prob = 0.55, win_mult = 1.0, loss_mult = 1.0
"""

import numpy as np
from utils.kelly_calculator import kelly_fraction, simulate_betting_strategy

# The exact scenario
win_prob = 0.55
win_mult = 1.0
loss_mult = 1.0

# Calculate Kelly
kelly = kelly_fraction(win_prob, win_mult, loss_mult)
print(f"Kelly formula result: {kelly:.4f} ({kelly*100:.2f}%)")
print()

# Theoretical verification
print("Theoretical Kelly (manual calculation):")
q = 1 - win_prob
kelly_manual = (win_prob * win_mult - q * loss_mult) / (win_mult * loss_mult)
print(f"  (p*w - q*l) / (w*l) = ({win_prob}*{win_mult} - {q}*{loss_mult}) / ({win_mult}*{loss_mult})")
print(f"  = ({win_prob * win_mult} - {q * loss_mult}) / {win_mult * loss_mult}")
print(f"  = {kelly_manual:.4f} ({kelly_manual*100:.2f}%)")
print()

# Expected log growth for different fractions
print("Expected log growth E[log(1+g)] for different fractions:")
print("-" * 70)

fractions = np.linspace(0.05, 0.20, 31)

for f in fractions:
    # Expected log growth
    log_growth_win = np.log(1 + f * win_mult)
    log_growth_lose = np.log(1 - f * loss_mult)
    expected_log_growth = win_prob * log_growth_win + (1-win_prob) * log_growth_lose
    
    print(f"f = {f:.4f} ({f*100:5.2f}%):  E[log(1+g)] = {expected_log_growth:.8f}")

print()
print("=" * 70)

# Find maximum
best_f = None
best_growth = -np.inf

for f in np.linspace(0.0, 0.3, 1000):
    if f >= 1.0:
        continue
    log_growth_win = np.log(1 + f * win_mult)
    log_growth_lose = np.log(1 - f * loss_mult) if f * loss_mult < 1 else -np.inf
    expected_log_growth = win_prob * log_growth_win + (1-win_prob) * log_growth_lose
    
    if expected_log_growth > best_growth:
        best_growth = expected_log_growth
        best_f = f

print(f"THEORETICAL OPTIMUM: f = {best_f:.4f} ({best_f*100:.2f}%) with E[log(1+g)] = {best_growth:.8f}")
print()

# Now test with simulations
print("SIMULATION VERIFICATION (5000 runs, 1000 bets each):")
print("-" * 70)

test_fractions = [0.10, 0.12, 0.14, 0.16]
n_sims = 5000
n_bets = 1000

for frac in test_fractions:
    finals = []
    for _ in range(n_sims):
        history = simulate_betting_strategy(
            1000, frac, n_bets, win_prob, win_mult, loss_mult, seed=None
        )
        finals.append(history[-1])
    
    # Geometric mean
    log_finals = [np.log(f/1000) for f in finals if f > 0]
    geo_mean_growth = np.mean(log_finals) / n_bets if log_finals else -np.inf
    
    # Arithmetic mean
    arith_mean = np.mean(finals)
    median = np.median(finals)
    
    print(f"f = {frac:.2f} ({frac*100:.1f}%):")
    print(f"  Geo mean growth rate: {geo_mean_growth:.8f}")
    print(f"  Arithmetic mean final: ${arith_mean:,.0f}")
    print(f"  Median final: ${median:,.0f}")
    print()

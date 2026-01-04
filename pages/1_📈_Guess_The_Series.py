"""
Guess the Real Series Mini-Game
Can you distinguish real market data from synthetic stochastic processes?
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import random

from utils.financial_models import (
    generate_gbm,
    generate_gbm_with_drift,
    match_moments,
    calculate_returns
)
from utils.data_loader import (
    load_all_periods,
    extract_price_series,
    get_periods_by_regime
)

# Color scheme
COLOR_REAL = "#4ECDC4"  # Turquoise
COLOR_SYNTHETIC = "#FF6B6B"  # Coral red
COLOR_NEUTRAL = "#95A5A6"  # Gray

# Ticker to common name mapping
TICKER_NAMES = {
    "SPY": "S&P 500 ETF",
    "^GSPC": "S&P 500 Index",
    "^IXIC": "Nasdaq Composite",
    "QQQ": "Nasdaq-100 ETF",
    "DIA": "Dow Jones ETF",
    "^DJI": "Dow Jones Index",
    "IWM": "Russell 2000 ETF",
    "^RUT": "Russell 2000 Index"
}

# Page configuration
st.set_page_config(
    page_title="Guess the Real Series",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Guess the Real Financial Series")

st.markdown("""
Financial markets are often modeled as **stochastic processes** - random walks with drift. 
But can you tell the difference between real market data and computer-generated synthetic series?

Test your intuition about market randomness!
""")

st.markdown("---")

# Initialize session state
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'game_data' not in st.session_state:
    st.session_state.game_data = None
if 'user_guesses' not in st.session_state:
    st.session_state.user_guesses = {}
if 'revealed' not in st.session_state:
    st.session_state.revealed = False
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = "Easy"

# Sidebar - Game Setup
with st.sidebar:
    st.header("🎮 Game Settings")
    
    difficulty = st.radio(
        "Difficulty Level",
        ["Easy", "Hard"],
        help="""
        **Easy**: Obvious differences between real and synthetic data
        **Hard**: Matched volatility makes them harder to distinguish
        """
    )
    
    st.session_state.difficulty = difficulty
    
    num_charts = st.slider(
        "Number of Charts",
        min_value=3,
        max_value=4,
        value=3,
        help="How many time series to compare"
    )
    
    st.markdown("---")
    
    if not st.session_state.game_started:
        if st.button("🎲 Start New Game", type="primary", use_container_width=True):
            st.session_state.game_started = True
            st.session_state.revealed = False
            st.session_state.user_guesses = {}
            st.rerun()
    else:
        if st.button("🔄 New Game", use_container_width=True):
            st.session_state.game_started = False
            st.session_state.game_data = None
            st.session_state.user_guesses = {}
            st.session_state.revealed = False
            st.rerun()

def generate_game_data(num_charts, difficulty):
    """Generate a mix of real and synthetic time series."""
    
    # Load available real data
    all_periods = load_all_periods()
    
    if not all_periods:
        st.error("No market data available. Please run: python scripts/download_data.py")
        return None
    
    # Determine how many real vs synthetic (at least 1 real, at least 1 synthetic)
    num_real = random.randint(1, num_charts - 1)
    num_synthetic = num_charts - num_real
    
    series_data = []
    
    # Select random real periods with preference for diversity
    available_periods = list(all_periods.keys())
    random.shuffle(available_periods)
    
    # For easy mode, prefer crisis periods for real data (more obvious patterns)
    if difficulty == "Easy":
        crisis_periods = [k for k in available_periods 
                         if all_periods[k][1]['regime'] == 'crisis']
        if crisis_periods:
            available_periods = crisis_periods + [p for p in available_periods 
                                                  if p not in crisis_periods]
    
    # Add real data
    for i in range(num_real):
        if i < len(available_periods):
            period_key = available_periods[i]
            df, info = all_periods[period_key]
            
            prices, dates = extract_price_series(df, max_points=500)
            
            # Normalize to start at 100
            prices = prices / prices[0] * 100
            
            series_data.append({
                'type': 'real',
                'prices': prices,
                'dates': dates,
                'name': info['name'],
                'ticker': info.get('ticker', 'N/A'),
                'description': info['description']
            })
    
    # Add synthetic data
    for i in range(num_synthetic):
        if difficulty == "Easy":
            # Easy mode: random walk with obvious characteristics
            drift = random.choice(['stable', 'up', 'down'])
            sigma = random.uniform(0.15, 0.25)
        else:
            # Hard mode: match characteristics of real data
            if series_data:  # Match to a real series
                real_prices = series_data[0]['prices']
                returns = calculate_returns(real_prices)
                mu = np.mean(returns) * 252  # Annualized
                sigma = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                mu = random.uniform(-0.05, 0.15)
                sigma = random.uniform(0.15, 0.30)
        
        # Generate synthetic series
        n_points = random.randint(250, 500)
        
        if difficulty == "Easy":
            synthetic_prices = generate_gbm_with_drift(
                S0=100,
                drift_type=drift,
                sigma=sigma,
                T=n_points/252,
                steps=n_points,
                seed=random.randint(1, 10000)
            )
        else:
            synthetic_prices = generate_gbm(
                S0=100,
                mu=mu,
                sigma=sigma,
                T=n_points/252,
                steps=n_points,
                seed=random.randint(1, 10000)
            )
        
        # Create fake dates
        dates = pd.date_range(start='2020-01-01', periods=len(synthetic_prices), freq='D')
        
        series_data.append({
            'type': 'synthetic',
            'prices': synthetic_prices,
            'dates': dates,
            'name': 'Synthetic Series',
            'description': 'Computer-generated using Geometric Brownian Motion'
        })
    
    # Shuffle so real and synthetic are mixed
    random.shuffle(series_data)
    
    # Assign labels
    for idx, data in enumerate(series_data):
        data['label'] = f"Series {chr(65 + idx)}"  # A, B, C, D
    
    return series_data

# Main game logic
if not st.session_state.game_started:
    # Instructions
    st.info("""
    ### 🎯 How to Play:
    
    1. **Choose your difficulty** in the sidebar
    2. **Click "Start New Game"** to generate time series charts
    3. **Examine each chart carefully** - look for patterns, volatility, trends
    4. **Make your guess** for each series: Is it REAL market data or SYNTHETIC?
    5. **Submit your guesses** to see how you did!
    
    ### 💡 Tips:
    
    - Real markets often have **regime changes** (calm periods followed by volatility)
    - Synthetic data tends to be **more uniform** in its randomness
    - Look for **extreme events** or **sharp reversals** that might indicate real crises
    - Remember: Sometimes they really do look identical! (That's the point of stochastic models)
    """)
    
    st.markdown("---")
    
    st.subheader("📚 What You'll Learn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **About Real Markets:**
        - Price movements appear random
        - Contains fat-tailed events (crashes)
        - Regime-dependent behavior
        """)
    
    with col2:
        st.markdown("""
        **About Stochastic Models:**
        - Geometric Brownian Motion (GBM)
        - Why random walks model prices
        - Limitations of simple models
        """)

else:
    # Generate game data if not exists
    if st.session_state.game_data is None:
        with st.spinner("Generating time series..."):
            st.session_state.game_data = generate_game_data(num_charts, difficulty)
    
    game_data = st.session_state.game_data
    
    if game_data is None:
        st.stop()
    
    # Display charts and collect guesses
    st.subheader("📊 Time Series Comparison")
    st.markdown("Examine each chart and decide: **Real market data** or **Synthetic**?")
    
    st.markdown("---")
    
    # Create grid of charts
    for idx, series_info in enumerate(game_data):
        label = series_info['label']
        prices = series_info['prices']
        dates = series_info['dates']
        
        # Create chart
        fig = go.Figure()
        
        # Color varies based on revealed state
        if st.session_state.revealed:
            line_color = COLOR_REAL if series_info['type'] == 'real' else COLOR_SYNTHETIC
        else:
            line_color = COLOR_NEUTRAL
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name=label,
            line=dict(width=2.5, color=line_color),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(int(line_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: %{y:.2f}<extra></extra>'
        ))
        
        # Build title with origin info if revealed
        if st.session_state.revealed and series_info['type'] == 'real':
            ticker_code = series_info.get('ticker', '')
            ticker_name = TICKER_NAMES.get(ticker_code, ticker_code)
            chart_title = f"<b>{label}</b> - {series_info['name']} ({ticker_name})"
        else:
            chart_title = f"<b>{label}</b>"
        
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=18, color='#ffffff')
            ),
            xaxis_title="Date",
            yaxis_title="Price (Indexed to 100)",
            height=350,
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=60),
            hovermode='x unified',
            plot_bgcolor='rgba(50,50,60,0.8)',
            paper_bgcolor='rgba(38,39,48,0.5)',
            font=dict(color='#ffffff'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
        
        with col2:
            st.markdown(f"### {label}")
            
            if not st.session_state.revealed:
                # Collect guess with styled buttons
                st.markdown("#### 🤔 Your Guess:")
                
                current_guess = st.session_state.user_guesses.get(label, None)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    # Padded text to ensure equal width
                    if st.button("📊 Real      ", key=f"btn_real_{idx}", use_container_width=True,
                                type="primary" if current_guess == "Real" else "secondary"):
                        st.session_state.user_guesses[label] = "Real"
                        st.rerun()
                with col_b:
                    if st.button("🤖 Synthetic", key=f"btn_synth_{idx}", use_container_width=True,
                                type="primary" if current_guess == "Synthetic" else "secondary"):
                        st.session_state.user_guesses[label] = "Synthetic"
                        st.rerun()
                
                if current_guess:
                    st.success(f"Selected: **{current_guess}**")
            else:
                # Show result with enhanced styling
                actual = "Real" if series_info['type'] == 'real' else "Synthetic"
                user_guess = st.session_state.user_guesses.get(label, "Unknown")
                
                correct = user_guess == actual
                
                # Result card
                if correct:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;">
                        <h3 style="color: #155724; margin: 0;">✅ Correct!</h3>
                        <p style="margin: 5px 0 0 0; font-size: 18px;"><b>{actual}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; padding: 15px; border-radius: 10px; border-left: 5px solid #dc3545;">
                        <h3 style="color: #721c24; margin: 0;">❌ Incorrect</h3>
                        <p style="margin: 5px 0 0 0; font-size: 16px;">You guessed: <b>{user_guess}</b></p>
                        <p style="margin: 5px 0 0 0; font-size: 18px;">Actually: <b>{actual}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                with st.expander("ℹ️ Details", expanded=False):
                    if series_info['type'] == 'real':
                        st.markdown(f"**{series_info['name']}**")
                        ticker_code = series_info.get('ticker', 'N/A')
                        ticker_name = TICKER_NAMES.get(ticker_code, ticker_code)
                        st.markdown(f"*Index: {ticker_name}*")
                        st.write(series_info['description'])
                        
                        # Add some statistics
                        returns = np.diff(np.log(series_info['prices']))
                        st.markdown(f"""  
                        **Key Statistics:**
                        - Total Return: {((series_info['prices'][-1] / series_info['prices'][0] - 1) * 100):.1f}%
                        - Volatility: {np.std(returns) * np.sqrt(252) * 100:.1f}% (annualized)
                        - Max Drawdown: {((np.min(series_info['prices']) / np.max(series_info['prices']) - 1) * 100):.1f}%
                        """)
                    else:
                        st.write("Computer-generated using **Geometric Brownian Motion (GBM)**")
                        st.write("This synthetic series mimics real market behavior using stochastic calculus.")
        
        st.markdown("---")
    
    # Submit button in main area (before results)
    if not st.session_state.revealed:
        st.markdown("")
        col_spacer1, col_submit, col_spacer2 = st.columns([1, 2, 1])
        with col_submit:
            if st.button("✅ Submit All Guesses", type="primary", use_container_width=True, key="submit_main"):
                if len(st.session_state.user_guesses) == len(game_data):
                    st.session_state.revealed = True
                    st.rerun()
                else:
                    st.error(f"⚠️ Please make a guess for all {len(game_data)} series before submitting!")
        st.markdown("---")
    
    # Show results summary
    if st.session_state.revealed:
        st.subheader("🎯 Your Results")
        
        correct_count = sum(
            1 for series_info in game_data
            if st.session_state.user_guesses.get(series_info['label']) == 
               ("Real" if series_info['type'] == 'real' else "Synthetic")
        )
        
        total_count = len(game_data)
        score_pct = (correct_count / total_count) * 100
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Circular gauge for score
            gauge_color = "#28a745" if score_pct >= 70 else "#ffc107" if score_pct >= 50 else "#dc3545"
            
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_pct,
                number={'suffix': "%", 'font': {'size': 32}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Accuracy Score", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2},
                    'bar': {'color': gauge_color, 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffebee'},
                        {'range': [50, 70], 'color': '#fff9c4'},
                        {'range': [70, 100], 'color': '#e8f5e9'}],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.8,
                        'value': 70}
                }
            ))
            
            gauge.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20))
            st.plotly_chart(gauge, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Performance Breakdown")
            
            st.metric("Correct Answers", f"{correct_count} / {total_count}")
            
            # Grade assessment
            if score_pct == 100:
                grade_emoji = "🎉"
                grade_text = "Perfect Score!"
                grade_msg = "You have excellent intuition for distinguishing real from synthetic data!"
            elif score_pct >= 70:
                grade_emoji = "👍"
                grade_text = "Great Job!"
                grade_msg = "You're good at spotting the differences between real and synthetic series."
            elif score_pct >= 50:
                grade_emoji = "🤔"
                grade_text = "Good Effort!"
                grade_msg = "Keep practicing - it's harder than it looks!"
            else:
                grade_emoji = "📚"
                grade_text = "Keep Learning!"
                grade_msg = "This is tricky! Try different difficulty levels to build intuition."
            
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-top: 10px;">
                <h2 style="margin: 0;">{grade_emoji} {grade_text}</h2>
                <p style="margin: 10px 0 0 0; font-size: 16px;">{grade_msg}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key insights
        st.subheader("💡 Key Takeaways")
        
        real_count = sum(1 for s in game_data if s['type'] == 'real')
        synthetic_count = len(game_data) - real_count
        
        st.info(f"""
        This game had **{real_count} real** and **{synthetic_count} synthetic** time series.
        
        **Why is this important?**
        
        - Financial models assume prices follow stochastic processes (like GBM)
        - If real and synthetic data are indistinguishable, the model captures market behavior well
        - However, real markets have features (fat tails, regime changes) that simple models miss
        - This is why risk management is crucial - models are approximations!
        """)

# Educational content
st.markdown("---")

with st.expander("📚 Understanding Stochastic Price Processes"):
    st.markdown("""
    ### What Are Stochastic Processes?
    
    A **stochastic process** is a mathematical model that describes how random variables evolve over time.
    In finance, we use these to model asset prices.
    
    ### Geometric Brownian Motion (GBM)
    
    The most common model for stock prices is **Geometric Brownian Motion**:
    
    $$dS_t = \\mu S_t dt + \\sigma S_t dW_t$$
    
    Where:
    - $S_t$ = price at time $t$
    - $\\mu$ = drift (expected return)
    - $\\sigma$ = volatility
    - $dW_t$ = random shock (Wiener process)
    
    ### Why Use GBM?
    
    **Advantages:**
    1. Prices never go negative (multiplicative process)
    2. Returns are normally distributed
    3. Captures randomness in markets
    4. Mathematically tractable
    
    **Limitations:**
    1. Assumes constant volatility (not true in real markets)
    2. No jumps or crashes (discontinuities)
    3. No regime changes
    4. Fat tails not captured
    
    ### Real Markets vs. Models
    
    Real financial markets exhibit:
    - **Volatility clustering**: Calm periods followed by volatile periods
    - **Fat tails**: Extreme events more common than normal distribution predicts
    - **Mean reversion**: Some tendency to return to long-term trends
    - **Asymmetry**: Crashes faster than rallies
    
    Despite these limitations, GBM remains the foundation of modern financial theory, including:
    - Black-Scholes option pricing
    - Portfolio optimization
    - Risk management models
    
    ### The Random Walk Hypothesis
    
    The similarity between real and synthetic data supports the **Efficient Market Hypothesis**:
    - Prices already reflect all available information
    - Future price movements are unpredictable (random)
    - You can't consistently "beat the market"
    
    However, market anomalies and behavioral factors suggest this isn't perfectly true!
    
    ### Advanced Models
    
    More sophisticated models include:
    - **GARCH**: Models changing volatility
    - **Jump diffusion**: Adds discontinuous jumps
    - **Stochastic volatility**: Volatility itself is random
    - **Regime-switching**: Different behavior in different market states
    
    ### Try This
    
    Play the game multiple times on different difficulty levels:
    - **Easy mode**: See obvious differences in pattern structure
    - **Hard mode**: Experience how matched volatility makes distinction nearly impossible
    
    This illustrates why financial modeling is both powerful and limited!
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Further Exploration
    
    **Books:**
    - "A Random Walk Down Wall Street" by Burton Malkiel
    - "The Misbehavior of Markets" by Benoit Mandelbrot
    
    **Concepts to explore:**
    - Brownian motion and diffusion processes
    - Ito's Lemma and stochastic calculus
    - Black-Scholes equation derivation
    - Market microstructure and high-frequency data
    """)

# Footer
st.markdown("---")
if st.button("🏠 Back to Home"):
    st.switch_page("Home.py")

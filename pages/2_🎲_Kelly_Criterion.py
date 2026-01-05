"""
Kelly Criterion Mini-Game
Learn optimal betting strategies through interactive simulations.
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
from utils.kelly_calculator import (
    kelly_fraction, 
    simulate_betting_strategy,
    compare_strategies
)

# Page configuration
st.set_page_config(
    page_title="Kelly Criterion Game",
    page_icon="🎲",
    layout="wide"
)

# Scroll to top if flag is set
if st.session_state.get('scroll_to_top', False):
    components.html(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_top = False

# Title
st.title("🎲 Kelly Criterion: Optimal Betting Strategy")

st.markdown("""
Imagine you have an edge in a repeated betting game. How much should you bet each time 
to maximize your long-term wealth growth while avoiding bankruptcy?

The **Kelly Criterion** provides the mathematically optimal answer.
""")

st.markdown("---")

# Initialize session state
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'game_params' not in st.session_state:
    st.session_state.game_params = None
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'simulation_counter' not in st.session_state:
    st.session_state.simulation_counter = 0

def generate_random_scenario(scenario_type):
    """Generate random betting scenario parameters based on selected type."""
    scenario_templates = {
        "Low Risk/Low Reward": {
            "win_prob_range": (0.55, 0.65),
            "win_mult_range": (0.8, 1.0),
            "description": "Conservative: High win probability, modest payout"
        },
        "Balanced": {
            "win_prob_range": (0.52, 0.58),
            "win_mult_range": (1.0, 1.5),
            "description": "Moderate: Balanced risk and reward"
        },
        "High Risk/High Reward": {
            "win_prob_range": (0.42, 0.48),
            "win_mult_range": (2.0, 3.0),
            "description": "Aggressive: Lower probability, high payout"
        },
        "Asymmetric Advantage": {
            "win_prob_range": (0.60, 0.70),
            "win_mult_range": (1.2, 2.0),
            "description": "Strong edge: Good odds, good payout"
        },
        "Marginal Edge": {
            "win_prob_range": (0.50, 0.53),
            "win_mult_range": (0.9, 1.2),
            "description": "Tight margins: Small edge, careful sizing needed"
        }
    }
    
    template = scenario_templates[scenario_type]
    
    # Randomize within the ranges
    win_prob = np.random.uniform(*template["win_prob_range"])
    win_mult = np.random.uniform(*template["win_mult_range"])
    
    return {
        'win_prob': round(win_prob, 2),
        'win_multiplier': round(win_mult, 1),
        'loss_multiplier': 1.0,
        'description': template['description'],
        'initial_capital': 1000,
        'n_bets': 100
    }

# Sidebar controls
with st.sidebar:
    st.header("🎮 Game Info")
    
    if st.session_state.game_started:
        st.info("🎲 Game in progress!")
        st.markdown("---")
        if st.button("↩️ Restart & Change Scenario", use_container_width=True):
            st.session_state.game_started = False
            st.session_state.game_params = None
            st.session_state.simulation_run = False
            st.rerun()
    else:
        st.info("💡 Configure your scenario in the main area, then start the game!")

# Main game area
if not st.session_state.game_started:
    # Game configuration
    st.markdown("### 🎮 Game Configuration")
    
    scenario_type = st.selectbox(
        "🎯 Select a Scenario Type:",
        ["Low Risk/Low Reward", "Balanced", "High Risk/High Reward", 
         "Asymmetric Advantage", "Marginal Edge"],
        help="Each scenario has different risk/reward characteristics. Try them all to build intuition!"
    )
    
    st.markdown("")
    
    # Start Game button in main area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎲 Start Game", type="primary", use_container_width=True, key="start_kelly_main"):
            st.session_state.game_started = True
            st.session_state.game_params = generate_random_scenario(scenario_type)
            st.session_state.simulation_run = False
            st.rerun()
    
    st.markdown("---")
    
    # Instructions before game starts
    st.info("""
    ### 🎯 How to Play:
    
    1. **Choose a scenario type** from the sidebar (Low Risk, Balanced, High Risk, etc.)
    2. **Click "Start Game"** to generate randomized parameters within that scenario
    3. **Study the scenario** - what are the odds and payoffs?
    4. **Decide your strategy** - what fraction of your capital will you bet?
    5. **Run the simulation** to see how your strategy performs vs Kelly optimal
    6. **Learn from the results** - see the distribution of outcomes
    7. **Try different scenarios** to build intuition about optimal betting!
    
    Your goal: Match or beat the Kelly Criterion strategy!
    """)
    
    st.markdown("---")
    
    st.subheader("📚 What is the Kelly Criterion?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        The Kelly Criterion tells you the **optimal fraction** of your capital to bet 
        when you have an edge in a repeated betting game.
        
        **Key Benefits:**
        - Maximizes long-term growth
        - Protects against bankruptcy
        - Balances risk and reward
        """)
    
    with col2:
        st.markdown("""
        **The Formula:**
        
        $$f^* = \\frac{{p(b+1) - 1}}{{b}}$$
        
        Where $p$ is win probability and $b$ is the win/loss ratio.
        """)

else:
    # New Scenario button in main area
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 New Scenario", use_container_width=True, key="new_scenario_main"):
            st.session_state.game_started = False
            st.session_state.game_params = None
            st.session_state.simulation_run = False
            st.rerun()
    
    st.markdown("---")
    
    # Game is active - show scenario and get user input
    params = st.session_state.game_params
    
    if params is None:
        st.error("Game parameters not initialized. Please click 'New Scenario' to restart.")
        st.stop()
    
    win_prob = params['win_prob']
    win_multiplier = params['win_multiplier']
    loss_multiplier = params['loss_multiplier']
    initial_capital = params['initial_capital']
    n_bets = params['n_bets']
    
    # Calculate Kelly fraction
    optimal_kelly = kelly_fraction(win_prob, win_multiplier, loss_multiplier)
    
    # Display scenario
    st.subheader("🎰 Your Betting Scenario")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Win Probability", f"{win_prob*100:.1f}%")
    
    with col2:
        st.metric("Win Amount", f"{win_multiplier:.1f}x bet")
    
    with col3:
        st.metric("Loss Amount", f"{loss_multiplier:.1f}x bet")
    
    with col4:
        st.metric("Starting Capital", f"${initial_capital:,}")
    
    st.info(f"**Scenario:** {params['description']}")
    
    st.markdown("---")
    
    # User input
    st.subheader("💭 Your Strategy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_bet_fraction = st.slider(
            "What fraction of your capital will you bet each round?",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            format="%.2f",
            help="Choose between 0 (bet nothing) and 1 (bet everything)"
        )
        
        st.markdown(f"""
        **Your choice:** Bet **{user_bet_fraction*100:.1f}%** of your current capital each round.
        
        - At start: ${initial_capital * user_bet_fraction:.2f} per bet
        - Kelly optimal: **{optimal_kelly*100:.1f}%**
        """)
    
    with col2:
        if user_bet_fraction > optimal_kelly * 1.5 and optimal_kelly > 0:
            st.warning("⚠️ You're betting more than 1.5x Kelly - high bankruptcy risk!")
        elif user_bet_fraction > optimal_kelly and optimal_kelly > 0:
            st.info("📊 You're over-betting compared to Kelly")
        elif user_bet_fraction < optimal_kelly * 0.5 and optimal_kelly > 0:
            st.info("🐢 You're being very conservative")
        elif abs(user_bet_fraction - optimal_kelly) < 0.05 and optimal_kelly > 0:
            st.success("🎯 Close to Kelly optimal!")
    
    st.markdown("---")
    
    # Run simulation button
    run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    if run_simulation:
        st.session_state.simulation_run = True
        st.session_state.simulation_counter += 1  # Increment to force re-simulation
    
    # Main simulation - use counter as key to force re-run
    if st.session_state.simulation_run and st.session_state.simulation_counter > 0:
        
        # Store current parameters to detect changes
        current_sim_key = f"{user_bet_fraction}_{st.session_state.simulation_counter}"
        
        st.subheader("📊 Results: Your Strategy vs Kelly Optimal")
        
        st.markdown("""
        We'll run simulations for two different timeframes to show how Kelly's advantage becomes 
        clearer with more repetitions:
        - **Short term** (100 bets): Some randomness, but trends emerge
        - **Long term** (1000 bets): Kelly's optimality becomes very clear
        """)
        
        # Run multiple simulations for both short and long term
        n_simulations = 100
        n_bets_short = 100
        n_bets_long = 1000
        
        with st.spinner("Running simulations for both timeframes..."):
            # Short term simulations
            user_results_short = []
            kelly_results_short = []
            for i in range(n_simulations):
                user_hist = simulate_betting_strategy(
                    initial_capital, user_bet_fraction, n_bets_short,
                    win_prob, win_multiplier, loss_multiplier, seed=None
                )
                kelly_hist = simulate_betting_strategy(
                    initial_capital, optimal_kelly, n_bets_short,
                    win_prob, win_multiplier, loss_multiplier, seed=None
                )
                user_results_short.append(user_hist)
                kelly_results_short.append(kelly_hist)
            
            # Long term simulations
            user_results_long = []
            kelly_results_long = []
            for i in range(n_simulations):
                user_hist = simulate_betting_strategy(
                    initial_capital, user_bet_fraction, n_bets_long,
                    win_prob, win_multiplier, loss_multiplier, seed=None
                )
                kelly_hist = simulate_betting_strategy(
                    initial_capital, optimal_kelly, n_bets_long,
                    win_prob, win_multiplier, loss_multiplier, seed=None
                )
                user_results_long.append(user_hist)
                kelly_results_long.append(kelly_hist)
        
        # Tab layout for short vs long term
        tab1, tab2 = st.tabs([f"📉 Short Term ({n_bets_short} bets)", f"📈 Long Term ({n_bets_long} bets)"])
        
        # Function to display results for a given timeframe
        def display_results(user_results, kelly_results, n_bets_display, tab_context):
            with tab_context:
                # Get one representative path for each
                user_path = user_results[0]
                kelly_path = kelly_results[0]
                
                # Extract final capitals
                user_finals = [r[-1] for r in user_results]
                kelly_finals = [r[-1] for r in kelly_results]
                
                # Plot 1: Single path comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Wealth Evolution (One Run)")
                    
                    fig1 = go.Figure()
                    
                    fig1.add_trace(go.Scatter(
                        x=list(range(len(user_path))),
                        y=user_path,
                        mode='lines',
                        name='Your Strategy',
                        line=dict(width=3, color='#FF6B6B'),
                        hovertemplate='<b>Your Strategy</b><br>Bet #%{x}<br>Capital: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig1.add_trace(go.Scatter(
                        x=list(range(len(kelly_path))),
                        y=kelly_path,
                        mode='lines',
                        name='Kelly Optimal',
                        line=dict(width=3, color='#4ECDC4'),
                        hovertemplate='<b>Kelly Optimal</b><br>Bet #%{x}<br>Capital: $%{y:,.2f}<extra></extra>'
                    ))
                    
                    fig1.add_hline(
                        y=initial_capital, 
                        line_dash="dash", 
                        line_color="gray",
                        annotation_text="Start",
                        annotation_position="right"
                    )
                    
                    fig1.update_layout(
                        xaxis_title="Bet Number",
                        yaxis_title="Capital ($)",
                        hovermode='x unified',
                        height=400,
                        template="plotly_dark",
                        showlegend=True,
                        plot_bgcolor='rgba(50,50,60,0.8)',
                        paper_bgcolor='rgba(38,39,48,0.5)',
                        font=dict(color='#ffffff'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown(f"### 📊 Final Wealth Distribution ({n_simulations} runs)")
                    
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Histogram(
                        x=user_finals,
                        name='Your Strategy',
                        opacity=0.75,
                        marker_color='#FF6B6B',
                        nbinsx=30,
                        hovertemplate='Capital: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig2.add_trace(go.Histogram(
                        x=kelly_finals,
                        name='Kelly Optimal',
                        opacity=0.75,
                        marker_color='#4ECDC4',
                        nbinsx=30,
                        hovertemplate='Capital: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig2.add_vline(
                        x=initial_capital,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Start"
                    )
                    
                    fig2.update_layout(
                        xaxis_title="Final Capital ($)",
                        yaxis_title="Frequency",
                        barmode='overlay',
                        height=400,
                        template="plotly_dark",
                        showlegend=True,
                        plot_bgcolor='rgba(50,50,60,0.8)',
                        paper_bgcolor='rgba(38,39,48,0.5)',
                        font=dict(color='#ffffff'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("---")
                
                # Statistical comparison
                st.markdown("### � Performance Metrics")
                
                # Calculate geometric mean (the correct metric for Kelly!)
                user_geo_mean = np.exp(np.mean(np.log([max(x, 1) for x in user_finals])))
                kelly_geo_mean = np.exp(np.mean(np.log([max(x, 1) for x in kelly_finals])))
                
                # Calculate arithmetic mean (can be misleading due to outliers)
                user_mean = np.mean(user_finals)
                kelly_mean = np.mean(kelly_finals)
                
                # Median (typical outcome)
                user_median = np.median(user_finals)
                kelly_median = np.median(kelly_finals)
                
                # Bankruptcy rate
                user_bankrupt = sum(1 for x in user_finals if x < initial_capital * 0.01) / n_simulations * 100
                kelly_bankrupt = sum(1 for x in kelly_finals if x < initial_capital * 0.01) / n_simulations * 100
                
                # Calculate average growth rate per bet from geometric mean
                user_growth_rate = (user_geo_mean/initial_capital)**(1/n_bets_display) - 1
                kelly_growth_rate = (kelly_geo_mean/initial_capital)**(1/n_bets_display) - 1
                
                # Important note about metrics
                st.info("📘 **Note:** Kelly optimizes the **geometric mean** (exponential growth rate), not the arithmetic mean. The arithmetic mean can be inflated by rare lucky runs and doesn't represent typical outcomes.")
                
                # Create visual comparison cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Your Strategy")
                    
                    # Circular gauge for growth rate
                    gauge_user = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=user_growth_rate * 100,
                        number={'suffix': "%", 'font': {'size': 24}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Growth Rate per Bet", 'font': {'size': 16}},
                        gauge={
                            'axis': {'range': [None, max(kelly_growth_rate * 100 * 1.5, user_growth_rate * 100 * 1.2)]},
                            'bar': {'color': "#FF6B6B"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, kelly_growth_rate * 100 * 0.5], 'color': '#ffebee'},
                                {'range': [kelly_growth_rate * 100 * 0.5, kelly_growth_rate * 100], 'color': '#fff9c4'},
                                {'range': [kelly_growth_rate * 100, kelly_growth_rate * 100 * 1.5], 'color': '#e8f5e9'}],
                            'threshold': {
                                'line': {'color': "green", 'width': 4},
                                'thickness': 0.75,
                                'value': kelly_growth_rate * 100}
                        }
                    ))
                    
                    gauge_user.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(gauge_user, use_container_width=True)
                    
                    # Key metrics as cards
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("💰 Geo Mean", f"${user_geo_mean:,.0f}", 
                                 f"{((user_geo_mean/initial_capital - 1) * 100):+.1f}%")
                        st.metric("📊 Median", f"${user_median:,.0f}")
                    with metric_col2:
                        # Bankruptcy risk indicator
                        if user_bankrupt > 20:
                            st.metric("🔥 Bankruptcy", f"{user_bankrupt:.0f}%", delta_color="inverse")
                        elif user_bankrupt > 5:
                            st.metric("⚠️ Bankruptcy", f"{user_bankrupt:.1f}%")
                        else:
                            st.metric("✅ Bankruptcy", f"{user_bankrupt:.1f}%")
                
                with col2:
                    st.markdown("#### 🏆 Kelly Optimal")
                    
                    # Circular gauge for growth rate
                    gauge_kelly = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=kelly_growth_rate * 100,
                        number={'suffix': "%", 'font': {'size': 24}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Growth Rate per Bet", 'font': {'size': 16}},
                        gauge={
                            'axis': {'range': [None, max(kelly_growth_rate * 100 * 1.5, user_growth_rate * 100 * 1.2)]},
                            'bar': {'color': "#4ECDC4"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, kelly_growth_rate * 100 * 0.5], 'color': '#ffebee'},
                                {'range': [kelly_growth_rate * 100 * 0.5, kelly_growth_rate * 100], 'color': '#fff9c4'},
                                {'range': [kelly_growth_rate * 100, kelly_growth_rate * 100 * 1.5], 'color': '#e8f5e9'}]
                        }
                    ))
                    
                    gauge_kelly.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(gauge_kelly, use_container_width=True)
                    
                    # Key metrics as cards
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("💰 Geo Mean", f"${kelly_geo_mean:,.0f}", 
                                 f"{((kelly_geo_mean/initial_capital - 1) * 100):+.1f}%")
                        st.metric("📊 Median", f"${kelly_median:,.0f}")
                    with metric_col2:
                        st.metric("✅ Bankruptcy", f"{kelly_bankrupt:.1f}%")
        
        # Display results for both timeframes
        display_results(user_results_short, kelly_results_short, n_bets_short, tab1)
        display_results(user_results_long, kelly_results_long, n_bets_long, tab2)
        
        st.markdown("---")
        
        # Overall performance feedback (using long-term results)
        user_finals_long = [r[-1] for r in user_results_long]
        kelly_finals_long = [r[-1] for r in kelly_results_long]
        user_mean_long = np.mean(user_finals_long)
        kelly_mean_long = np.mean(kelly_finals_long)
        user_bankrupt_long = sum(1 for x in user_finals_long if x < initial_capital * 0.01) / n_simulations * 100
        kelly_bankrupt_long = sum(1 for x in kelly_finals_long if x < initial_capital * 0.01) / n_simulations * 100
        
        # Performance feedback
        st.subheader("💡 How Did You Do?")
        
        if user_mean_long > kelly_mean_long * 0.95:
            st.success(f"""
            **Excellent!** Your strategy performed very well in the long term.
            - Your long-term mean return: {((user_mean_long/initial_capital - 1) * 100):.1f}%
            - Kelly long-term mean return: {((kelly_mean_long/initial_capital - 1) * 100):.1f}%
            
            You achieved similar or better growth to the Kelly optimal strategy!
            """)
        elif user_mean_long > kelly_mean_long * 0.75:
            st.info(f"""
            **Good job!** Your strategy was reasonable but left some growth on the table in the long run.
            - Your long-term mean return: {((user_mean_long/initial_capital - 1) * 100):.1f}%
            - Kelly long-term mean return: {((kelly_mean_long/initial_capital - 1) * 100):.1f}%
            
            Kelly achieved {((kelly_mean_long/user_mean_long - 1) * 100):.1f}% more growth over 1000 bets.
            """)
        else:
            st.warning(f"""
            **Room for improvement.** Your strategy significantly underperformed Kelly in the long term.
            - Your long-term mean return: {((user_mean_long/initial_capital - 1) * 100):.1f}%
            - Kelly long-term mean return: {((kelly_mean_long/initial_capital - 1) * 100):.1f}%
            
            Consider adjusting your bet size closer to the Kelly optimal fraction.
            """)
        
        if user_bankrupt_long > kelly_bankrupt_long * 2 and user_bankrupt_long > 5:
            st.error(f"""
            ⚠️ **High Bankruptcy Risk:** Your strategy went bankrupt in {user_bankrupt_long:.1f}% of long-term simulations, 
            compared to {kelly_bankrupt_long:.1f}% for Kelly. You may be over-betting!
            """)
        
        st.info("""
        **Key Insight:** Notice how Kelly's advantage becomes clearer in the long term (1000 bets) 
        compared to the short term (100 bets). This demonstrates that Kelly maximizes **geometric mean growth** - 
        it's the optimal strategy for long-term wealth accumulation!
        
        🔄 Try a new scenario to explore different betting situations!
        """)
        
        # Educational content - collapsible
        st.markdown("---")
        
        with st.expander("📚 Learn the Mathematics Behind the Kelly Criterion"):
            st.markdown(f"""
            ### The Kelly Criterion Formula
            
            The optimal fraction of your capital to bet is given by:
            
            $$f^* = \\frac{{p \\cdot w - q \\cdot l}}{{w \\cdot l}}$$
            
            Where:
            - $f^*$ = optimal fraction to bet
            - $p$ = probability of winning
            - $q$ = probability of losing (1-p)
            - $w$ = win multiplier
            - $l$ = loss multiplier
            
            ### Why It Works
            
            The Kelly Criterion maximizes the **expected logarithmic growth** of your capital. This means:
            
            1. **Long-term Growth**: Maximizes your wealth over many repeated bets
            2. **Bankruptcy Protection**: Never bets so much that you risk total ruin
            3. **Optimal Risk/Reward**: Balances growth potential with risk management
            
            ### Example Calculation
            
            With your current scenario:
            - Win probability: {win_prob:.1%}
            - Win multiplier: {win_multiplier:.2f}
            - Loss multiplier: {loss_multiplier:.2f}
            
            $$f^* = \\frac{{{win_prob:.2f} \\times {win_multiplier:.2f} - {1-win_prob:.2f} \\times {loss_multiplier:.2f}}}{{{win_multiplier:.2f} \\times {loss_multiplier:.2f}}} = {optimal_kelly:.1%}$$
            
            ### Important Notes
            
            - **Half Kelly**: Many professionals use half the Kelly fraction for more conservative betting
            - **Overbet Risk**: Betting more than Kelly dramatically increases bankruptcy risk
            - **Real Markets**: In practice, estimating win probability is the hard part!
            
            ### Applications in Finance
            
            The Kelly Criterion is used by:
            - Professional gamblers and poker players
            - Hedge fund managers for position sizing
            - Options traders for optimal leverage
            - Any situation with repeated bets and known odds
            
            ### Further Reading
            
            - Original paper: J. L. Kelly Jr., "A New Interpretation of Information Rate" (1956)
            - Book: "Fortune's Formula" by William Poundstone
            - Application: Edward Thorp's use in blackjack and stock market
            """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Interactive Challenge
    
    Try these scenarios to build intuition:
    
    1. **No Edge** (Win Prob = 50%, Win/Loss = 1.0): Kelly says bet 0%!
    2. **Strong Edge** (Win Prob = 70%, Win/Loss = 1.0): Kelly recommends 40%
    3. **High Payoff** (Win Prob = 55%, Win/Loss = 2.0): Bigger bets are optimal
    4. **Marginal Edge** (Win Prob = 51%, Win/Loss = 1.0): Bet only 2%
    """)

# Navigation buttons at the bottom
st.markdown("---")
st.markdown("---")

bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    if st.button("🔄 Play Again", use_container_width=True, type="primary", key="play_again_bottom"):
        st.session_state.game_started = False
        st.session_state.game_params = None
        st.session_state.simulation_run = False
        st.session_state.scroll_to_top = True
        st.rerun()

with bottom_col2:
    if st.button("🏠 Back to Main Home Page", use_container_width=True, type="secondary", key="back_to_home_bottom"):
        st.switch_page("Home.py")

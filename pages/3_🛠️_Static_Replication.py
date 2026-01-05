"""
Static Replication Game
Learn how any European option payoff can be replicated using vanilla calls and puts.
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from utils.option_math import (
    portfolio_payoff, get_named_payoff, get_option_description,
    compute_static_replication, call_payoff
)

# Page configuration
st.set_page_config(
    page_title="Static Replication Game",
    page_icon="🛠️",
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

# Initialize session state
if 'rep_game_started' not in st.session_state:
    st.session_state.rep_game_started = False
if 'rep_level' not in st.session_state:
    st.session_state.rep_level = 1
if 'rep_show_solution' not in st.session_state:
    st.session_state.rep_show_solution = False

# Title and introduction
st.title("🛠️ Static Replication: Building Any Payoff")

if not st.session_state.rep_game_started:
    st.markdown("""
    ### Welcome to the Static Replication Workshop!
    
    #### 🎯 The Challenge You Face
    
    Imagine you work at an investment bank. A client wants to buy an exotic derivative with a complex payoff 
    (like a "strangle" or "butterfly"). Your job is to:
    1. **Price it** - what should the client pay?
    2. **Hedge it** - how do you protect the bank if you sell it?
    
    The solution: **Static Replication** - break down the exotic payoff into simple building blocks that 
    have known market prices!
    """)
    
    st.markdown("---")
    
    # Explanation of European options
    st.markdown("#### 📖 Financial Derivatives: The Basics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What's a European Option?**
        
        A European option is a financial contract that pays out based on a stock's price at a **specific future date** 
        (the maturity). Unlike American options, you can only exercise European options at maturity, not before.
        
        **Two main types:**
        - **Call Option**: Right to BUY the stock at a fixed price (strike)
        - **Put Option**: Right to SELL the stock at a fixed price (strike)
        
        In this workshop, we'll focus on **call options** as building blocks.
        """)
    
    with col2:
        st.markdown("""
        **Why Static Hedging?**
        
        When a bank sells an exotic option, it faces risk. Two hedging approaches exist:
        
        1. **Dynamic Hedging** (complex):
           - Continuously rebalance a portfolio as markets move
           - Requires constant trading, high transaction costs
           - Sensitive to market volatility
        
        2. **Static Hedging** (elegant):
           - Buy a portfolio ONCE and hold to maturity
           - No rebalancing needed!
           - Perfect hedge if replication is exact
        
        This workshop teaches the static approach.
        """)
    
    st.markdown("---")
    
    # Visual explanation of call option
    st.markdown("#### 📊 Understanding Call Options")
    
    # Create a simple call option payoff diagram
    S_demo = np.linspace(70, 130, 100)
    K_demo = 100
    call_payoff_demo = np.maximum(S_demo - K_demo, 0)
    
    fig_demo = go.Figure()
    
    fig_demo.add_trace(go.Scatter(
        x=S_demo,
        y=call_payoff_demo,
        name='Call Option Payoff',
        line=dict(color='royalblue', width=4),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.2)'
    ))
    
    fig_demo.add_vline(x=K_demo, line_dash="dash", line_color="red", 
                      annotation_text="Strike = $100", annotation_position="top")
    
    fig_demo.add_annotation(x=115, y=15, text="In the money<br>(profitable)",
                           showarrow=True, arrowhead=2, arrowcolor="green")
    fig_demo.add_annotation(x=85, y=5, text="Out of the money<br>(worthless at expiry)",
                           showarrow=True, arrowhead=2, arrowcolor="gray")
    
    fig_demo.update_layout(
        title="Call Option Payoff: max(Stock Price - Strike, 0)",
        xaxis_title="Stock Price at Maturity ($)",
        yaxis_title="Payoff ($)",
        template="plotly_dark",
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(50,50,60,0.8)',
        paper_bgcolor='rgba(38,39,48,0.5)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig_demo, use_container_width=True)
    
    st.markdown("""
    A call option gives you the **right** (but not obligation) to buy a stock at a fixed price (the **strike**) on a future date (the **maturity**).
    
    **Key Points:**
    - 🟢 **Strike Price**: The price you can buy the stock at ($100 in the example above)
    - 📅 **Maturity Date**: When the option expires (e.g., 1 year from now)
    - 💰 **Payoff**: At maturity, the option is worth max(Stock Price - Strike, 0)
      - If stock = $110, payoff = $10 (you buy at $100, sell at $110)
      - If stock = $90, payoff = $0 (you don't exercise)
    
    **Why options are valuable TODAY:**  
    Even if a call is currently "out of the money" (stock < strike), it still has value because the stock *might* go up before maturity!
    """)
    
    st.markdown("---")
    
    # The pricing problem
    st.markdown("#### 💰 The Pricing Problem")
    
    st.info("""
    **How do we price an exotic option that doesn't trade in the market?**
    
    The **no-arbitrage principle** gives us the answer:
    
    > If two portfolios have identical payoffs at maturity, they must have the same price today.
    
    **Therefore:**
    1. Find a portfolio of vanilla options (calls/puts) that replicates the exotic payoff
    2. The exotic option's price = the sum of vanilla option prices in the replicating portfolio
    3. Market prices for vanilla options are observable → we can price the exotic!
    
    This is exactly what you'll practice in this workshop.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    **The Static Replication Toolkit:**  
    
    Any exotic payoff can be built by combining:
    - **The underlying stock** - linear payoff: if stock goes up $1, you make $1
    - **Call options at different strikes** - kinked payoffs that "turn on" above the strike
    - **Cash** - constant payoff: shifts everything up or down
    
    It's like building any shape with LEGO blocks!
    
    **What you'll learn in this workshop:**
    - ✓ How to decompose exotic payoffs into vanilla building blocks
    - ✓ Why this technique allows us to price complex derivatives
    - ✓ How static hedging protects banks from exotic option risk
    - ✓ The mathematical foundation (Breeden-Litzenberger theorem)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Your Goal:**
        
        Build portfolios that match exotic option payoffs using only vanilla calls at different strikes.
        
        **What You'll Learn:**
        - How option payoffs combine
        - The building blocks of financial engineering
        - Why any payoff can be replicated
        - Real market applications
        """)
    
    with col2:
        st.markdown("""
        **🎮 How It Works:**
        
        1. Choose a target exotic option (Digital Call, Strangle, etc.)
        2. Adjust quantities of vanilla calls at available strikes
        3. See your portfolio payoff match the target in real-time
        4. Discover how markets price and hedge exotic derivatives
        """)
    
    st.markdown("---")
    
    if st.button("🚀 Start Workshop", type="primary", use_container_width=True):
        st.session_state.rep_game_started = True
        st.rerun()

else:
    # Game is active
    
    # Sidebar: Level selection
    st.sidebar.header("📚 Workshop Levels")
# Define levels dictionary (outside sidebar so it's accessible everywhere)
levels = {
    1: {
        'name': 'Butterfly Spread',
        'payoff_type': 'butterfly',
        'strikes': [90, 95, 100, 105, 110],
        'difficulty': 'Beginner',
        'hint': 'This one only needs calls at different strikes. You\'ll need positive and negative quantities to create the peak.',
        'mode': 'interactive',
        'allow_stock': False
    },
    2: {
        'name': 'Long Strangle', 
        'payoff_type': 'strangle',
        'strikes': [90, 95, 100, 105, 110],
        'difficulty': 'Intermediate',
        'hint': 'Try combining the stock with call options. Think about what happens when the stock moves far from $100 in either direction.',
        'mode': 'interactive',
        'allow_stock': True
    },
    3: {
        'name': 'Power Call (Demonstration)',
        'payoff_type': 'power_call',
        'strikes': list(range(85, 121, 2)),  # Every $2
        'difficulty': 'Demo',
        'hint': 'Watch how adding more strikes progressively improves the approximation.',
        'mode': 'animation',
        'allow_stock': False
    }
}

# Sidebar
with st.sidebar:
    st.header("🎮 Game Info")
    current_level = levels[st.session_state.rep_level]
    st.info(f"🎯 Current: Level {st.session_state.rep_level}\n{current_level['name']}")
    st.markdown("---")
    if st.button("↩️ Change Level", use_container_width=True, key="change_level_sidebar"):
        # Scroll to top by rerunning
        st.rerun()

# Level selection in main area
st.markdown("### 🎮 Select a Level")

level_col1, level_col2, level_col3 = st.columns(3)

for idx, (level_id, level_data) in enumerate(levels.items()):
    if level_data['difficulty'] == 'Beginner':
        difficulty_emoji = "🟢"
    elif level_data['difficulty'] == 'Intermediate':
        difficulty_emoji = "🟡"
    elif level_data['difficulty'] == 'Advanced':
        difficulty_emoji = "🔴"
    else:
        difficulty_emoji = "🎬"
    
    level_col = [level_col1, level_col2, level_col3][idx]
    with level_col:
        button_label = f"{difficulty_emoji} Level {level_id}\n{level_data['name']}"
        if st.button(
            button_label,
            use_container_width=True,
            type="primary" if st.session_state.rep_level == level_id else "secondary",
            key=f"level_{level_id}_main"
        ):
            st.session_state.rep_level = level_id
            st.session_state.rep_show_solution = False
            st.rerun()

st.markdown("---")

# Current level
current_level = levels[st.session_state.rep_level]

# Header
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader(f"Level {st.session_state.rep_level}: {current_level['name']}")
with col2:
    st.metric("Difficulty", current_level['difficulty'])
with col3:
    st.metric("Available Strikes", len(current_level['strikes']))

# Option description
option_info = get_option_description(current_level['payoff_type'])

with st.expander("ℹ️ About This Option", expanded=True):
    st.markdown(f"**{option_info['name']}**")
    st.markdown(option_info['description'])
    st.info(f"**Real Market Use:** {option_info['use_case']}")

st.markdown("---")

# Setup stock price grid
S = np.linspace(70, 130, 300)

# Get target payoff
params = {'K': 100, 'K1': 90, 'K2': 110, 'Km': 100}
target_payoff = get_named_payoff(current_level['payoff_type'], S, params)

# Check if this is animation mode
if current_level['mode'] == 'animation':
    # Animation mode - demonstrate progressive approximation
    st.markdown("### 🎬 Watch How It Works")
    st.info("This level demonstrates how using more strikes creates a better approximation. Use the slider to see the progression.")
    
    strikes = current_level['strikes']
    
    # Slider to control how many strikes to use
    n_strikes_to_use = st.slider(
        "Number of strikes to use in replication:",
            min_value=3,
            max_value=len(strikes),
            value=5,
            step=1,
            key="animation_slider"
        )
        
    # Use subset of strikes
    active_strikes = strikes[:n_strikes_to_use]
    
    # Compute optimal replication for this subset
    optimal_weights = compute_static_replication(
        target_payoff, S, active_strikes, option_type='call'
    )
    
    user_positions = [
        {'type': 'call', 'K': K, 'q': optimal_weights[K]} 
        for K in active_strikes
    ]
    
    st.markdown(f"**Using {n_strikes_to_use} strikes:** {active_strikes[:3]}...{active_strikes[-1] if len(active_strikes) > 3 else ''}")
        
else:
    # Interactive mode - user adjusts quantities
    st.markdown("### 🔧 Build Your Replicating Portfolio")
    
    with st.expander("💡 Hint", expanded=False):
        st.markdown(current_level['hint'])
    
    # Create columns for strike inputs
    strikes = current_level['strikes']
    n_strikes = len(strikes)
    allow_stock = current_level.get('allow_stock', False)
    
    user_positions = []
    
    if allow_stock:
        # Add stock and cash position inputs first
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💵 Cash Position")
            cash_qty = st.slider(
                "Cash Amount ($)",
                min_value=-200.0,
                max_value=200.0,
                value=0.0,
                step=10.0,
                key=f"cash_level_{st.session_state.rep_level}",
                help="Add or subtract a constant amount to the entire payoff"
            )
            st.caption(f"${cash_qty:.0f}")
            if cash_qty != 0:
                user_positions.append({'type': 'cash', 'K': 0, 'q': cash_qty})
        
        with col2:
            st.markdown("#### 📊 Stock Position")
            stock_qty = st.slider(
                "Quantity of Stock",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                key=f"stock_level_{st.session_state.rep_level}",
                help="Positive = long stock, Negative = short stock"
            )
            st.caption(f"{stock_qty:.1f} shares")
            if stock_qty != 0:
                user_positions.append({'type': 'stock', 'K': 0, 'q': stock_qty})
        
        st.markdown("#### 📞 Call Options")
    
    # Call options
    strikes_per_row = 5
    n_rows = (n_strikes + strikes_per_row - 1) // strikes_per_row
    
    for row in range(n_rows):
        cols = st.columns(strikes_per_row)
        for col_idx in range(strikes_per_row):
            strike_idx = row * strikes_per_row + col_idx
            if strike_idx < n_strikes:
                K = strikes[strike_idx]
                with cols[col_idx]:
                    q = st.slider(
                        f"K=${K}",
                        min_value=-3.0,
                        max_value=3.0,
                        value=0.0,
                        step=0.1,
                        key=f"call_{K}_level_{st.session_state.rep_level}",
                        label_visibility="visible"
                    )
                    if q != 0:
                        st.caption(f"{q:+.1f}")
                        user_positions.append({'type': 'call', 'K': K, 'q': q})
                    else:
                        st.caption("—")

# Calculate user portfolio payoff
user_payoff = portfolio_payoff(S, user_positions)

# Calculate error metric
error = np.sqrt(np.mean((target_payoff - user_payoff)**2))
max_error = np.sqrt(np.mean(target_payoff**2))
accuracy = max(0, 100 * (1 - error / max_error)) if max_error > 0 else 0

st.markdown("---")

# Visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Payoff Diagram")
    
    fig = go.Figure()
    
    # Target payoff
    fig.add_trace(go.Scatter(
        x=S, 
        y=target_payoff,
        name='🎯 Target Payoff',
        line=dict(color='rgba(200, 200, 200, 0.8)', width=4, dash='dash'),
        hovertemplate='S=%{x:.0f}<br>Target=%{y:.2f}<extra></extra>'
    ))
    
    # User portfolio
    fig.add_trace(go.Scatter(
        x=S,
        y=user_payoff,
        name='🛠️ Your Portfolio',
        line=dict(color='royalblue', width=3),
        hovertemplate='S=%{x:.0f}<br>Your Payoff=%{y:.2f}<extra></extra>'
    ))
    
    # Error area (filled)
    fig.add_trace(go.Scatter(
        x=np.concatenate([S, S[::-1]]),
        y=np.concatenate([target_payoff, user_payoff[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 100, 100, 0.2)',
        line=dict(width=0),
        name='Replication Error',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add strike markers with color coding for active positions
    active_strikes = {p['K'] for p in user_positions if p['type'] == 'call' and abs(p['q']) > 0.01}
    for K in strikes:
        if K in active_strikes:
            fig.add_vline(
                x=K,
                line_dash="dash",
                line_color="royalblue",
                opacity=0.6,
                annotation_text=f"K={K}",
                annotation_position="top"
            )
        else:
            fig.add_vline(
                x=K,
                line_dash="dot",
                line_color="lightgray",
                opacity=0.2
            )
    
    fig.update_layout(
        xaxis_title="Stock Price at Expiration ($)",
        yaxis_title="Payoff ($)",
        hovermode='x unified',
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        plot_bgcolor='rgba(50,50,60,0.8)',
        paper_bgcolor='rgba(38,39,48,0.5)',
        font=dict(color='#ffffff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📈 Replication Quality")
    
    # Create circular gauge for accuracy
    import plotly.graph_objects as go
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': '#ffebee'},
                {'range': [70, 90], 'color': '#fff9c4'},
                {'range': [90, 100], 'color': '#e8f5e9'}],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 95}}
    ))
    
    gauge_fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        title={'text': "Accuracy Score", 'x': 0.5, 'xanchor': 'center'}
    )
    
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feedback (only for interactive mode)
    if current_level['mode'] == 'interactive':
        if accuracy >= 98:
            st.success("🎉 **Excellent!** Perfect replication!")
            st.balloons()
        elif accuracy >= 90:
            st.success("✅ **Great!** Very close match!")
        elif accuracy >= 70:
            st.warning("🤏 **Good progress!** Keep adjusting...")
        else:
            st.info("💪 **Keep trying!** Adjust the quantities.")
    else:
        # Animation mode - show informative message
        if accuracy >= 95:
            st.success("✨ **Excellent approximation!** With enough strikes, we get nearly perfect replication.")
        elif accuracy >= 80:
            st.info("📈 **Good approximation.** Add more strikes to improve further!")
        else:
            st.warning("⚠️ **Rough approximation.** Try adding more strikes using the slider above.")
    
    st.markdown("---")
    
    # Portfolio composition visualization
    st.markdown("**Portfolio Composition:**")
    
    # Count position types
    call_positions = [p for p in user_positions if p['type'] == 'call' and abs(p['q']) > 0.01]
    has_stock = any(p['type'] == 'stock' and abs(p['q']) > 0.01 for p in user_positions)
    has_cash = any(p['type'] == 'cash' and abs(p['q']) > 0.01 for p in user_positions)
    
    # Create composition indicator
    comp_cols = st.columns([1, 1, 1])
    with comp_cols[0]:
        if has_stock:
            st.success("✓ Stock")
        else:
            st.info("○ Stock")
    with comp_cols[1]:
        if has_cash:
            st.success("✓ Cash")
        else:
            st.info("○ Cash")
    with comp_cols[2]:
        if call_positions:
            st.success(f"✓ {len(call_positions)} Call(s)")
        else:
            st.info("○ Calls")
    
    st.markdown("**Position Details:**")
    active_positions = [p for p in user_positions if abs(p['q']) > 0.01]
    if active_positions:
        if current_level['mode'] == 'animation':
            st.markdown(f"*Showing optimal weights for {len(active_positions)} strikes*")
            if len(active_positions) <= 8:
                for p in active_positions:
                    sign = "+" if p['q'] > 0 else ""
                    if p['type'] == 'stock':
                        st.markdown(f"- {sign}{p['q']:.1f} × Stock")
                    elif p['type'] == 'cash':
                        st.markdown(f"- ${p['q']:.0f} Cash")
                    else:
                        st.markdown(f"- {sign}{p['q']:.1f} × Call(K=${p['K']})")
            else:
                st.markdown(f"*{len(active_positions)} positions (see weights in chart)*")
        else:
            for p in active_positions:
                sign = "+" if p['q'] > 0 else ""
                if p['type'] == 'stock':
                    st.markdown(f"- {sign}{p['q']:.1f} × Stock")
                elif p['type'] == 'cash':
                    st.markdown(f"- ${p['q']:.0f} Cash")
                else:
                    st.markdown(f"- {sign}{p['q']:.1f} × Call(K=${p['K']})")
    else:
        st.markdown("*No positions yet*")
    
    st.markdown("---")
    
    # Show solution button (only in interactive mode and for call-only levels)
    if current_level['mode'] == 'interactive':
        if current_level.get('allow_stock', False):
            st.info("💡 **Hint:** For this payoff, the natural solution uses puts (or equivalently, stock + calls). The exact combination depends on your approach!")
        else:
            if st.button("💡 Show Theoretical Solution", use_container_width=True):
                st.session_state.rep_show_solution = True
    else:
        st.info("💡 **Note:** Move the slider to see how more strikes improve the approximation!")

# Show solution if requested
if st.session_state.rep_show_solution and current_level['mode'] == 'interactive':
    st.markdown("---")
    st.markdown("### 🎓 Theoretical Solution")
    
    # Compute optimal replication
    optimal_weights = compute_static_replication(
        target_payoff, S, strikes, option_type='call'
    )
    
    optimal_positions = [
        {'type': 'call', 'K': K, 'q': w} 
        for K, w in optimal_weights.items()
    ]
    optimal_payoff = portfolio_payoff(S, optimal_positions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optimal Weights:**")
        for K, w in sorted(optimal_weights.items()):
            if abs(w) > 0.01:
                sign = "+" if w > 0 else ""
                st.markdown(f"- {sign}{w:.3f} × Call(K=${K})")
    
    with col2:
        st.markdown("**Key Insight:**")
        st.info("""
        The weights are determined by the **second derivative** of the target payoff.
        
        This is the **Breeden-Litzenberger formula**:
        
        Any payoff f(S) can be written as:
        
        $$f(S) = \\int_0^\\infty f''(K) \\cdot (S-K)^+ dK$$
        
        In practice, we use discrete strikes as an approximation.
        """)
    
    # Pricing illustration
    if current_level['mode'] == 'interactive' and len([p for p in user_positions if abs(p['q']) > 0.01]) > 0:
        st.markdown("---")
        st.markdown("### 💰 How This Relates to Pricing")
        
        with st.expander("📖 Click to see pricing calculation", expanded=False):
            st.markdown("""
            In real markets, this replication technique is used to **price exotic options**.
            
            **The No-Arbitrage Principle:**  
            If two portfolios have identical payoffs at maturity, they must have the same price today.
            
            Therefore:
            $$\\text{Price(Exotic)} = \\sum_{i} w_i \\times \\text{Price(Call}_i\\text{)} + w_S \\times S_0 + w_C \\times PV(\\text{Cash})$$
            
            Where:
            - $w_i$ are the replication weights for calls
            - $w_S$ is the stock position weight
            - $w_C$ is the cash amount
            - $PV(\\text{Cash})$ is the present value of cash (discounted from maturity to today)
            """)
            
            st.markdown("#### Example Calculation")
            st.markdown("**Assumed market parameters:**")
            
            # Create example prices (simplified Black-Scholes-like)
            current_stock_price = 100
            risk_free_rate = 0.05
            time_to_maturity = 1.0  # 1 year
            discount_factor = np.exp(-risk_free_rate * time_to_maturity)
            
            st.markdown(f"- Current stock price: **${current_stock_price}**")
            st.markdown(f"- Risk-free rate: **{risk_free_rate*100:.1f}%** per year")
            st.markdown(f"- Time to maturity: **{time_to_maturity} year**")
            st.markdown(f"- Discount factor: **{discount_factor:.4f}** (for discounting cash)")
            
            st.markdown("---")
            
            # Simplified option prices (Black-Scholes inspired)
            # Use a simple formula: Price = Intrinsic + Time Value
            # Time value depends on moneyness and volatility
            implied_vol = 0.20  # 20% annualized volatility
            
            example_prices = {
                'stock': current_stock_price,
                'cash_pv': discount_factor,  # Present value of $1 at maturity
            }
            
            for K in strikes:
                # Intrinsic value
                intrinsic = max(current_stock_price - K, 0)
                
                # Time value (simplified): decreases with distance from ATM
                # Even deep OTM options have some time value
                moneyness = abs(current_stock_price - K) / current_stock_price
                time_value = 8.0 * np.exp(-2 * moneyness) * np.sqrt(time_to_maturity)
                
                # Total option price
                option_price = intrinsic + time_value
                example_prices[f'call_{K}'] = option_price
            
            # Calculate portfolio value
            portfolio_value = 0
            st.markdown("**Your Portfolio Value Today:**")
            
            active_pos = [p for p in user_positions if abs(p['q']) > 0.01]
            if active_pos:
                for p in active_pos:
                    if p['type'] == 'stock':
                        value = p['q'] * example_prices['stock']
                        portfolio_value += value
                        st.markdown(f"- {p['q']:.1f} × Stock @ **${example_prices['stock']:.2f}** = **${value:.2f}**")
                    elif p['type'] == 'cash':
                        value = p['q'] * example_prices['cash_pv']
                        portfolio_value += value
                        st.markdown(f"- ${p['q']:.0f} Cash (at maturity) × **{example_prices['cash_pv']:.4f}** (PV factor) = **${value:.2f}**")
                        st.caption(f"   ↳ Cash must be discounted because you receive it in {time_to_maturity} year, not today")
                    elif p['type'] == 'call':
                        price = example_prices[f'call_{p["K"]}']
                        value = p['q'] * price
                        portfolio_value += value
                        st.markdown(f"- {p['q']:.1f} × Call(K=${p['K']}) @ **${price:.2f}** = **${value:.2f}**")
                
                st.markdown("---")
                st.metric("💰 Total Portfolio Value (Fair Price Today)", f"${portfolio_value:.2f}")
                
                st.success(f"""
                **Key Insight:** If your portfolio perfectly replicates the exotic option's payoff, 
                then the exotic option should trade at **${portfolio_value:.2f}** in a no-arbitrage market!
                
                This is how investment banks price complex derivatives:
                1. Decompose the exotic payoff into vanilla building blocks
                2. Observe market prices for each building block
                3. Sum up: Price(Exotic) = Σ(weights × market prices)
                4. Account for time value of money (discount future cash flows)
                """)
            else:
                st.markdown("*Add some positions to see the pricing calculation*")
    
    # Educational section
    st.markdown("---")
    
    with st.expander("📚 Why This Matters in Real Markets"):
        st.markdown("""
        ### Static Replication in Practice
        
        **1. Pricing Exotic Options**
        - Banks don't have pricing models for every exotic option
        - Instead, they decompose the payoff into vanilla calls/puts
        - Price each vanilla using Black-Scholes
        - Sum up to get the exotic option price
        
        **2. Hedging Exotic Books**
        - If a bank sells an exotic option, they need to hedge it
        - Instead of dynamic hedging (complex!), they can use static replication
        - Buy the replicating portfolio once and hold to expiration
        - No rebalancing needed!
        
        **3. The Breeden-Litzenberger Theorem**
        - Mathematically proves that ANY European payoff can be replicated
        - Uses a continuum of strikes (in practice, we approximate with discrete strikes)
        - The formula: weights are proportional to the second derivative of the payoff
        
        **4. Extracting Risk-Neutral Probabilities**
        - Option prices reveal market expectations about future stock prices
        - By observing call/put prices at all strikes, we can extract the full probability distribution
        - This is how traders gauge "implied volatility" across strikes (the "vol smile")
        
        ### As You Add More Strikes...
        
        In Level 4, notice how using more strikes (every $2 instead of every $5) gives a better approximation.
        
        In the limit, with strikes at **every possible price**, you get **perfect replication**.
        
        This is the theoretical foundation for derivative pricing!
        """)
    
    with st.expander("🧮 The Mathematics"):
        st.markdown(r"""
        ### The Static Replication Formula
        
        For any twice-differentiable European payoff $f(S_T)$, we can write:
        
        $$f(S_T) = f(F) + f'(F)(S_T - F) + \int_0^F f''(K) \cdot P(K) \, dK + \int_F^\infty f''(K) \cdot C(K) \, dK$$
        
        Where:
        - $F$ = forward price
        - $C(K)$ = Call with strike $K$
        - $P(K)$ = Put with strike $K$
        - $f'', f'$ = derivatives of the payoff function
        
        **Key Insights:**
        1. The "weight" on each option is the **second derivative** of the payoff
        2. We use **puts** for strikes below the forward, **calls** above
        3. We also need a position in the forward/stock (the $f'(F)$ term) and cash (the $f(F)$ term)
        
        **For Discrete Approximation:**
        
        With strikes $K_1, K_2, \ldots, K_n$, we approximate:
        
        $$f(S_T) \approx \sum_{i=1}^n w_i \cdot C(K_i)$$
        
        Where $w_i \approx f''(K_i) \cdot \Delta K$ (the second derivative times the strike spacing).
        
        As $\Delta K \to 0$, the approximation becomes exact!
        """)

# Navigation buttons at the bottom
st.markdown("---")
st.markdown("---")

bottom_col1, bottom_col2 = st.columns(2)

with bottom_col1:
    if st.button("🔄 Play Again", use_container_width=True, type="primary", key="play_again_bottom"):
        st.session_state.rep_game_started = False
        st.session_state.rep_level = 1
        st.session_state.rep_show_solution = False
        st.session_state.scroll_to_top = True
        st.rerun()

with bottom_col2:
    if st.button("🏠 Back to Main Home Page", use_container_width=True, type="secondary", key="back_to_home_bottom"):
        st.switch_page("Home.py")

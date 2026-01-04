"""
Math Finance Playground
Interactive mini-games to illustrate financial mathematics concepts.
"""

import streamlit as st
from utils.data_loader import get_period_summary

# Page configuration
st.set_page_config(
    page_title="Math Finance Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("📊 Math Finance Playground")
st.markdown("### Explore some financial mathematics concepts through interactive mini-games")

st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Welcome! This app helps you understand some concepts in financial mathematics through 
    hands-on experience. Choose a mini-game from the sidebar to get started.
    
    Each game takes about **5 minutes** and provides immediate feedback on your decisions.
    After playing, you can explore the mathematical theory behind the concepts.
    """)

with col2:
    st.info("💡 **Tip**: Each game is designed for exploration. Don't worry about getting everything right!")

st.markdown("---")

# Game cards
st.markdown("## 🎮 Available Mini-Games")

game1_col, game2_col = st.columns(2)

with game1_col:
    st.markdown("### 📈 Guess the Real Series")
    st.markdown("""
    **What you'll learn:**
    - Stochastic price behavior
    - Why markets appear random
    - Synthetic vs. real financial data
    
    **How it works:**
    View several time series charts mixing real market data with computer-generated 
    synthetic series. Can you tell which is real?
    """)
    
    if st.button("📈 Play Guess the Series Game", use_container_width=True, type="primary"):
        st.switch_page("pages/1_📈_Guess_The_Series.py")

with game2_col:
    st.markdown("### 🎲 Kelly Criterion Game")
    st.markdown("""
    **What you'll learn:**
    - Optimal betting strategies
    - Risk management in investing
    - Why "betting it all" is dangerous
    
    **How it works:**
    Simulate repeated bets with different strategies and see how your wealth evolves over time.
    Compare Kelly optimal betting vs. fixed percentages and aggressive over-betting.
    """)
    
    if st.button("🎲 Play Kelly Criterion Game", use_container_width=True):
        st.switch_page("pages/2_🎲_Kelly_Criterion.py")

st.markdown("---")

# Third game row
game3_col, game4_col = st.columns(2)

with game3_col:
    st.markdown("### 🛠️ Static Replication")
    st.markdown("""
    **What you'll learn:**
    - How to build any option payoff
    - Breeden-Litzenberger theorem
    - Real market pricing and hedging
    
    **How it works:**
    Match exotic option payoffs by combining vanilla calls at different strikes.
    See how traders replicate and price complex derivatives in practice.
    """)
    
    if st.button("🛠️ Play Static Replication Game", use_container_width=True):
        st.switch_page("pages/3_🛠️_Static_Replication.py")

with game4_col:
    st.markdown("### 🚧 More Games Coming Soon!")
    st.markdown("""
    **Future topics:**
    - Portfolio optimization
    - Risk-return tradeoffs
    - Volatility and Greeks
    
    **Stay tuned!**
    More interactive games are in development.
    """)

st.markdown("---")

# Data information
st.markdown("## 📊 About the Data")

try:
    summary_df = get_period_summary()
    
    if summary_df['Data Available'].str.contains('❌').any():
        st.warning("""
        ⚠️ Some historical data files are missing. 
        Please run the data download script first:
        
        ```bash
        python scripts/download_data.py
        ```
        """)
    else:
        st.success("✅ All historical data files are available!")
    
    # with st.expander("📋 View Available Market Periods"):
    #     st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    #     st.markdown("""
    #     **Market Regimes:**
    #     - **Crisis**: Periods of high volatility and sharp declines
    #     - **Stability**: Low volatility, sideways trading
    #     - **Mixed**: Complete market cycles including booms and busts
    #     - **Bull**: Clear upward trends
    #     """)

except Exception as e:
    st.error(f"Error loading data information: {str(e)}")
    st.info("""
    This is normal if you haven't downloaded the data yet.
    Run: `python scripts/download_data.py`
    """)

st.markdown("---")

# About section
with st.expander("## ℹ️ About This App"):
    st.markdown("""
    ### Math Finance Playground
    
    This educational application was created to help enthusiasts explore 
    some concepts in financial mathematics through interactive mini-games.
    
    **Created by:** Camilo García Trillos (UCL)  
    **With assistance from:** Anthropic's Claude Sonnet 4.5
    
    **Technologies:** Python, Streamlit, Plotly, NumPy, Pandas
    """)


st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Educational use only | Market data provided by Yahoo Finance</p>
    <p>Built with Streamlit and Python</p>
</div>
""", unsafe_allow_html=True)

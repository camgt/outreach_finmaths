---
title: Math Finance Playground
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 📊 Math Finance Playground

Interactive mini-games to explore financial mathematics concepts through hands-on experience.

## 🎮 Available Games

1. **📈 Guess the Real Series** - Can you distinguish real market data from synthetic stochastic processes?
2. **🎲 Kelly Criterion** - Learn optimal betting strategies and risk management
3. **🛠️ Static Replication** - Build exotic option payoffs using vanilla calls

## 🚀 Features

- Interactive visualizations with Plotly
- Real historical market data from multiple crisis and stability periods
- Educational explanations of mathematical concepts
- Mobile-friendly design for tablets

## 📚 Technologies

- Python 3.11
- Streamlit
- Plotly
- NumPy, Pandas, SciPy

## 👨‍💼 Created By

**Camilo García Trillos** (UCL)  
With assistance from Anthropic's Claude Sonnet 4.5

## 📄 License

MIT License - Educational use only

Market data provided by Yahoo Finance

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run Home.py
```

## Data Setup

Before running the app for the first time, download the historical market data:

```bash
python scripts/download_data.py
```

This downloads static historical data for various market periods (2008 crisis, COVID-19, etc.) and saves them as CSV files.

## Project Structure

```
outreach_finmaths/
├── Home.py                      # Landing page
├── pages/
│   ├── 1_🎲_Kelly_Criterion.py  # Kelly betting game
│   └── 2_📈_Guess_The_Series.py # Series identification game
├── utils/
│   ├── kelly_calculator.py      # Kelly criterion calculations
│   ├── financial_models.py      # Time series generation (GBM)
│   └── data_loader.py           # Data loading utilities
├── data/
│   ├── raw/                     # Historical market data CSVs
│   └── data_config.json         # Metadata for market periods
├── scripts/
│   └── download_data.py         # One-time data download
└── requirements.txt
```

## Deployment

This app can be deployed to Streamlit Cloud for free browser-based access on tablets and lightweight computers.

## License

Educational use. Market data provided by Yahoo Finance.

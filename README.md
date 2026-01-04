# Financial Mathematics Educational App

Interactive mini-games to illustrate financial mathematics concepts for a general audience.

## Features

### 🎲 Kelly Criterion Game
Learn optimal investment strategies through interactive betting simulations. Compare Kelly optimal betting vs fixed percentage and over-betting strategies.

### 📈 Guess the Real Series
Can you distinguish real market data from synthetic stochastic processes? Test your intuition about market randomness and learn about stochastic price behavior.

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

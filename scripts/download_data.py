"""
Download historical market data for the educational app.
Run this script once to populate the data/raw/ directory with CSV files.
"""

import yfinance as yf
from pathlib import Path
import json

# Load configuration
config_path = Path(__file__).parent.parent / "data" / "data_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Create data directory
data_dir = Path(__file__).parent.parent / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

print("📊 Downloading historical market data...")
print("=" * 60)

successful_downloads = 0
failed_downloads = []

for period_key, period_info in config['periods'].items():
    ticker = period_info['ticker']
    start = period_info['start']
    end = period_info['end']
    name = period_info['name']
    
    print(f"\n📈 {name}")
    print(f"   Ticker: {ticker} | Period: {start} to {end}")
    
    try:
        # Download data
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            print(f"   ⚠️  No data returned for {ticker}")
            failed_downloads.append((period_key, "No data returned"))
            continue
        
        # Save to CSV
        output_file = data_dir / f"{period_key}.csv"
        df.to_csv(output_file)
        
        print(f"   ✅ Downloaded {len(df)} days of data")
        print(f"   💾 Saved to: {output_file.name}")
        successful_downloads += 1
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        failed_downloads.append((period_key, str(e)))

print("\n" + "=" * 60)
print(f"\n📦 Summary:")
print(f"   ✅ Successful: {successful_downloads}/{len(config['periods'])}")

if failed_downloads:
    print(f"   ❌ Failed: {len(failed_downloads)}")
    for period_key, error in failed_downloads:
        print(f"      - {period_key}: {error}")
else:
    print(f"   🎉 All data downloaded successfully!")

print("\n✨ Data download complete. You can now run the Streamlit app.")
print("   Run: streamlit run Home.py")

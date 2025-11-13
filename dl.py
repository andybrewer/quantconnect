from ib_insync import IB, Stock, util
from datetime import datetime, timedelta
import pandas as pd
import os
import zipfile

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)  # TWS paper trading port

contract = Stock('QQQ', 'SMART', 'USD')

# Create output directory
output_dir = 'data/equity/usa/minute/qqq'
os.makedirs(output_dir, exist_ok=True)

# Download minute data day by day
target_start = datetime(2022, 12, 1)  # Start 30 days before 2023-01-01 for warmup
target_end = datetime(2024, 7, 31, 23, 59, 59)

print(f"Downloading QQQ minute data from {target_start.date()} to {target_end.date()}...")
print(f"Checking existing files and building download list...\n")

# Get all business days in the range
all_dates = pd.date_range(target_start, target_end, freq='B')  # Business days only
print(f"Total business days in range: {len(all_dates)}")

# Check which files already exist
existing_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
existing_dates = set()
for d in all_dates:
    if f"{d.strftime('%Y%m%d')}_trade.zip" in existing_files:
        existing_dates.add(d)

print(f"Already have data for: {len(existing_dates)} days")

# Get dates that need to be downloaded
dates_to_download = [d for d in all_dates if d not in existing_dates]
print(f"Need to download: {len(dates_to_download)} days\n")

if not dates_to_download:
    print("All data already downloaded!")
else:
    print(f"Downloading {len(dates_to_download)} days...\n")
    request_count = 0
    
    for date in dates_to_download:
        request_count += 1
        date_str = date.strftime('%Y%m%d')
        
        print(f"[{request_count}/{len(dates_to_download)}] Downloading {date.strftime('%Y-%m-%d')}...", end=' ')
        
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=date.strftime('%Y%m%d 23:59:59'),
                durationStr='1 D',  # 1 day of data
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True,
                timeout=30
            )
            
            if bars:
                df = util.df(bars)
                
                # Convert to LEAN format
                lean_data = []
                for _, row in df.iterrows():
                    time_ms = int((row['date'] - row['date'].normalize()).total_seconds() * 1000)
                    open_price = int(row['open'] * 10000)
                    high_price = int(row['high'] * 10000)
                    low_price = int(row['low'] * 10000)
                    close_price = int(row['close'] * 10000)
                    volume = int(row['volume'])
                    lean_data.append(f"{time_ms},{open_price},{high_price},{low_price},{close_price},{volume}")
                
                # Save to zip file
                zip_filename = f"{date_str}_trade.zip"
                zip_path = os.path.join(output_dir, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    csv_content = '\n'.join(lean_data)
                    zipf.writestr(f"{date_str}_trade.csv", csv_content)
                
                print(f"✓ {len(bars)} bars saved")
            else:
                print(f"⚠ No data returned (likely holiday/market closed)")
                
        except Exception as e:
            print(f"✗ Error: {e}")

print(f"\n{'='*60}")
print(f"Download complete!")
print(f"Total requests made: {request_count}")
print(f"Data saved to: {output_dir}")
print(f"{'='*60}\n")

ib.disconnect()

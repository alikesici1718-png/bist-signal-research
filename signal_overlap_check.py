"""Diagnostic: overlap between volume-spike and extreme-down signals.

Measures how many (symbol, date) events are shared between the
volume_spike_2x and extreme_down signals to determine whether they are
independent hypotheses or one phenomenon counted twice (relevant for the
BH-FDR correction in comprehensive_scan.py). Console output only.
"""
import os
import pandas as pd
import numpy as np

def read_stock_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = {}
    for f in files:
        if f == 'USDTRY.csv' or f == 'USDTRY=X.csv':
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['Date'])
        df = df.set_index('Date')
        dfs[symbol] = df
    return dfs

def compute_signals(dfs):
    signals = {}
    for symbol, df in dfs.items():
        if len(df) < 250:
            continue
        close = df['Close']
        volume = df['Volume'] if 'Volume' in df.columns else None
        returns = np.log(close / close.shift(1)).dropna()
        if len(returns) < 250:
            continue
        
        # Volume spike: volume > 2x 20-day rolling median
        if volume is not None:
            vol_median = volume.rolling(20, min_periods=10).median()
            vol_spike = (volume > 2 * vol_median).reindex(returns.index, fill_value=False)
        else:
            vol_spike = pd.Series(False, index=returns.index)
        
        # Extreme down reversal: today's return in bottom 5% of last 60 days
        rolling_bottom_5 = returns.rolling(60, min_periods=30).quantile(0.05)
        extreme_down = (returns < rolling_bottom_5).reindex(returns.index, fill_value=False)
        
        signals[symbol] = pd.DataFrame({
            'volume_spike': vol_spike,
            'extreme_down': extreme_down
        })
    return signals

def main():
    data_dir = 'data'
    signals = compute_signals(read_stock_data(data_dir))
    
    all_overlap = []
    all_vol_only = []
    all_ext_only = []
    
    for symbol, sig in signals.items():
        vol_dates = sig.index[sig['volume_spike']].tolist()
        ext_dates = sig.index[sig['extreme_down']].tolist()
        
        overlap = set(vol_dates) & set(ext_dates)
        vol_only = set(vol_dates) - set(ext_dates)
        ext_only = set(ext_dates) - set(vol_dates)
        
        all_overlap.extend([(symbol, d) for d in overlap])
        all_vol_only.extend([(symbol, d) for d in vol_only])
        all_ext_only.extend([(symbol, d) for d in ext_only])
    
    total_vol = len(all_overlap) + len(all_vol_only)
    total_ext = len(all_overlap) + len(all_ext_only)
    
    print(f"Total volume_spike events: {total_vol}")
    print(f"Total extreme_down events: {total_ext}")
    print(f"Overlap (both same day): {len(all_overlap)}")
    print(f"Volume-only events: {len(all_vol_only)}")
    print(f"Extreme-only events: {len(all_ext_only)}")
    print()
    print(f"Overlap % of volume_spike: {100*len(all_overlap)/total_vol:.1f}%")
    print(f"Overlap % of extreme_down: {100*len(all_overlap)/total_ext:.1f}%")
    print()
    
    # Show top overlap dates (most symbols hit on same day)
    from collections import Counter
    date_counts = Counter([d for _, d in all_overlap])
    print("Top 10 overlap dates (most symbols):")
    for date, count in date_counts.most_common(10):
        print(f"  {date.strftime('%Y-%m-%d')}: {count} symbols")
    
    print()
    print("Top 10 volume-only dates:")
    date_counts_vol = Counter([d for _, d in all_vol_only])
    for date, count in date_counts_vol.most_common(10):
        print(f"  {date.strftime('%Y-%m-%d')}: {count} symbols")
    
    print()
    print("Top 10 extreme-only dates:")
    date_counts_ext = Counter([d for _, d in all_ext_only])
    for date, count in date_counts_ext.most_common(10):
        print(f"  {date.strftime('%Y-%m-%d')}: {count} symbols")

if __name__ == '__main__':
    main()
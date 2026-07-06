import os
import pandas as pd
import numpy as np
from scipy import stats

def read_stock_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = {}
    for f in files:
        if f in ('USDTRY.csv', 'USDTRY=X.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['Date'])
        df = df.set_index('Date')
        dfs[symbol] = df
    return dfs

def compute_avg_volume(dfs, last_n_days=60):
    avg_vols = {}
    for symbol, df in dfs.items():
        if 'Volume' not in df.columns or len(df) < last_n_days:
            continue
        avg_vols[symbol] = df['Volume'].tail(last_n_days).mean()
    return avg_vols

def main():
    data_dir = 'data'
    dfs = read_stock_data(data_dir)
    
    # Liquid filter (top 50%)
    avg_vols = compute_avg_volume(dfs)
    median_vol = np.median(list(avg_vols.values()))
    liquid_symbols = [s for s, v in avg_vols.items() if v >= median_vol]
    
    close_df = pd.DataFrame({s: dfs[s]['Close'] for s in liquid_symbols if len(dfs[s]) >= 250})
    close_df = close_df.dropna(axis=1, thresh=int(0.8 * len(close_df)))
    close_df = close_df.dropna(axis=0, thresh=int(0.8 * len(close_df.columns)))
    
    returns = np.log(close_df / close_df.shift(1)).dropna()
    market_ret = returns.mean(axis=1)
    
    # Signal
    q05 = returns.rolling(60, min_periods=30).quantile(0.05)
    sig = returns < q05
    
    # 1. EVENT FREQUENCY PER SYMBOL
    print("=== EVENT FREQUENCY PER SYMBOL ===")
    event_counts = sig.sum().sort_values(ascending=False)
    print(f"Total symbols: {len(event_counts)}")
    print(f"Mean events per symbol: {event_counts.mean():.1f}")
    print(f"Median: {event_counts.median():.1f}")
    print(f"Max: {event_counts.max()} (symbol: {event_counts.idxmax()})")
    print(f"Symbols with >30 events: {(event_counts > 30).sum()}")
    print(f"Symbols with <10 events: {(event_counts < 10).sum()}")
    print("\nTop 10 most frequent:")
    print(event_counts.head(10).to_string())
    print()
    
    # 2. GAP BETWEEN EVENTS (same symbol)
    print("=== GAP BETWEEN CONSECUTIVE EVENTS (same symbol) ===")
    all_gaps = []
    for col in returns.columns:
        dates = returns.index[sig[col].fillna(False)]
        if len(dates) >= 2:
            gaps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            all_gaps.extend(gaps)
    
    if all_gaps:
        print(f"Mean gap: {np.mean(all_gaps):.1f} days")
        print(f"Median gap: {np.median(all_gaps):.1f} days")
        print(f"Events within 10 days: {sum(1 for g in all_gaps if g <= 10)} / {len(all_gaps)} ({100*sum(1 for g in all_gaps if g <= 10)/len(all_gaps):.1f}%)")
        print(f"Events within 20 days: {sum(1 for g in all_gaps if g <= 20)} / {len(all_gaps)} ({100*sum(1 for g in all_gaps if g <= 20)/len(all_gaps):.1f}%)")
    print()
    
    # 3. POST-EVENT DRIFT CONDITIONAL ON MARKET STATE
    print("=== POST-EVENT DRIFT (h=5) BY MARKET STATE ===")
    
    # h=5 future excess
    h = 5
    future_mkt = market_ret.shift(-h).rolling(h).sum()
    
    # Event day market return buckets
    results = []
    for col in returns.columns:
        event_dates = returns.index[sig[col].fillna(False)]
        for d in event_dates:
            if d not in future_mkt.index or pd.isna(future_mkt.loc[d]):
                continue
            # Market return on event day
            mkt_day = market_ret.loc[d] if d in market_ret.index else np.nan
            if pd.isna(mkt_day):
                continue
            
            # Future excess
            f_stock = returns[col].shift(-h).rolling(h).sum().loc[d]
            f_excess = f_stock - future_mkt.loc[d]
            
            results.append({
                'symbol': col,
                'date': d,
                'mkt_day': mkt_day * 100,
                'future_excess_5d': f_excess * 100
            })
    
    df_res = pd.DataFrame(results)
    
    # Bucket by event-day market return
    df_res['mkt_bucket'] = pd.cut(df_res['mkt_day'],
                                   bins=[-np.inf, -5, -2, 0, 2, np.inf],
                                   labels=['<-5%', '-5to-2%', '-2to0%', '0to2%', '>2%'])
    
    summary = df_res.groupby('mkt_bucket')['future_excess_5d'].agg(['count', 'mean', 'std', 'median'])
    print(summary.round(3))
    print()
    
    # T-test per bucket
    print("T-test (H0: mean excess = 0):")
    for bucket in df_res['mkt_bucket'].cat.categories:
        subset = df_res[df_res['mkt_bucket'] == bucket]['future_excess_5d'].dropna()
        if len(subset) > 5:
            t, p = stats.ttest_1samp(subset, 0)
            p_one = p / 2 if t < 0 else 1 - p / 2
            print(f"  {bucket}: n={len(subset)}, mean={subset.mean():+.3f}%, t={t:.2f}, p_one={p_one:.4f}")
    
    # 4. CAPACITY: Events per day distribution
    print("\n=== CAPACITY: EVENTS PER DAY ===")
    daily_counts = sig.sum(axis=1)
    daily_counts = daily_counts[daily_counts > 0]
    print(f"Days with events: {len(daily_counts)} / {len(returns)} ({100*len(daily_counts)/len(returns):.1f}%)")
    print(f"Mean events per day: {daily_counts.mean():.1f}")
    print(f"Median: {daily_counts.median():.1f}")
    print(f"Max events single day: {daily_counts.max()} on {daily_counts.idxmax().strftime('%Y-%m-%d')}")
    print(f"Days with >50 events: {(daily_counts > 50).sum()}")
    print(f"Days with >100 events: {(daily_counts > 100).sum()}")

if __name__ == '__main__':
    main()
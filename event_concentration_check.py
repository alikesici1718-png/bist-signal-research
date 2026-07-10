"""Diagnostic: are signal events concentrated in a few symbols or dates?

Checks whether volume-spike / extreme-down event counts cluster in a
handful of (mostly illiquid) symbols or calendar dates, which would make
pooled t-stats unreliable. Reports per-symbol and per-date concentration
plus liquidity (avg volume) stratification. Diagnostic only — produces
console output, no result CSV.
"""
import os
import pandas as pd
import numpy as np
from collections import Counter

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
    
    # Filter liquid (top 50% by volume) - same as liquidity_filtered_scan
    avg_vols = compute_avg_volume(dfs)
    if not avg_vols:
        print("No volume data")
        return
    
    median_vol = np.median(list(avg_vols.values()))
    liquid_symbols = [s for s, v in avg_vols.items() if v >= median_vol]
    
    # Build returns panel for liquid symbols with sufficient history
    close_df = pd.DataFrame({s: dfs[s]['Close'] for s in liquid_symbols if len(dfs[s]) >= 250})
    close_df = close_df.dropna(axis=1, thresh=int(0.8 * len(close_df)))
    close_df = close_df.dropna(axis=0, thresh=int(0.8 * len(close_df.columns)))
    
    returns = np.log(close_df / close_df.shift(1)).dropna()
    market_ret = returns.mean(axis=1)
    
    # extreme_down signal: bottom 5% of 60-day rolling
    q05 = returns.rolling(60, min_periods=30).quantile(0.05)
    sig = returns < q05
    
    # Collect all events
    events = []
    for col in returns.columns:
        event_dates = returns.index[sig[col].fillna(False)]
        for d in event_dates:
            events.append({
                'symbol': col,
                'date': d,
                'stock_return': returns.loc[d, col] * 100,  # in percent
                'market_return': market_ret.loc[d] * 100 if d in market_ret.index else np.nan
            })
    
    df_events = pd.DataFrame(events)
    print(f"Total extreme_down events in liquid universe: {len(df_events)}")
    print(f"Unique dates with events: {df_events['date'].nunique()}")
    print(f"Symbols in universe: {len(returns.columns)}")
    print()
    
    # 1. DATE CONCENTRATION
    date_counts = df_events['date'].value_counts().sort_values(ascending=False)
    print("=== TOP 20 EVENT DATES (most symbols hit) ===")
    for date, count in date_counts.head(20).items():
        mkt_ret = market_ret.loc[date] * 100 if date in market_ret.index else 0
        print(f"  {date.strftime('%Y-%m-%d')}: {count:3d} symbols | market: {mkt_ret:+.2f}%")
    print()
    
    # Cumulative concentration
    top_5_dates = date_counts.head(5).sum()
    top_10_dates = date_counts.head(10).sum()
    print(f"Top 5 dates cover: {top_5_dates} events ({100*top_5_dates/len(df_events):.1f}%)")
    print(f"Top 10 dates cover: {top_10_dates} events ({100*top_10_dates/len(df_events):.1f}%)")
    print()
    
    # 2. MARKET RETURN ON EVENT DAYS
    print("=== MARKET RETURN DISTRIBUTION ON EVENT DAYS ===")
    event_market_rets = df_events['market_return'].dropna()
    print(f"  Mean market return on event days: {event_market_rets.mean():+.3f}%")
    print(f"  Median market return: {event_market_rets.median():+.3f}%")
    print(f"  Std: {event_market_rets.std():.3f}%")
    print(f"  % of event days with market < -1%: {100*(event_market_rets < -1).sum()/len(event_market_rets):.1f}%")
    print(f"  % of event days with market < -2%: {100*(event_market_rets < -2).sum()/len(event_market_rets):.1f}%")
    print()
    
    # 3. IS IT JUST MARKET BETA?
    print("=== STOCK vs MARKET RETURN ON EVENT DAYS ===")
    clean = df_events.dropna(subset=['stock_return', 'market_return'])
    print(f"  Correlation(stock_ret, market_ret) on event days: {clean['stock_return'].corr(clean['market_return']):.3f}")
    
    # Excess return on event day itself
    clean['excess'] = clean['stock_return'] - clean['market_return']
    print(f"  Mean excess return on event day: {clean['excess'].mean():+.3f}%")
    print(f"  Median excess return: {clean['excess'].median():+.3f}%")
    print(f"  % negative excess: {100*(clean['excess'] < 0).sum()/len(clean):.1f}%")
    print()
    
    # 4. POST-EVENT DRIFT BY MARKET CONDITION
    print("=== POST-EVENT PERFORMANCE BY MARKET CONDITION ===")
    print("(This requires forward-looking data, showing event-day market return bucket)")
    
    # Bucket event days by market return
    clean['mkt_bucket'] = pd.cut(clean['market_return'], 
                                  bins=[-np.inf, -3, -1, 0, 1, np.inf],
                                  labels=['<-3%', '-3to-1%', '-1to0%', '0to1%', '>1%'])
    print(clean.groupby('mkt_bucket')['excess'].agg(['count', 'mean', 'median']).round(3))

if __name__ == '__main__':
    main()
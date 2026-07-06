import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

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

def align_data(dfs, min_days=250):
    # Filter by length
    valid = {s: df for s, df in dfs.items() if len(df) >= min_days}
    if not valid:
        return None, None
    
    # Compute avg volume and filter top 50% liquid
    avg_vols = compute_avg_volume(valid)
    if not avg_vols:
        return None, None
    
    median_vol = np.median(list(avg_vols.values()))
    liquid_symbols = [s for s, v in avg_vols.items() if v >= median_vol]
    
    print(f"Symbols with volume data: {len(avg_vols)}")
    print(f"Median 60-day avg volume: {median_vol:,.0f}")
    print(f"Liquid symbols (>= median): {len(liquid_symbols)}")
    
    # Build close price panel for liquid symbols
    close_df = pd.DataFrame({s: valid[s]['Close'] for s in liquid_symbols})
    close_df = close_df.dropna(axis=1, thresh=int(0.8 * len(close_df)))
    close_df = close_df.dropna(axis=0, thresh=int(0.8 * len(close_df.columns)))
    
    print(f"Aligned panel: {close_df.shape[1]} symbols, {close_df.shape[0]} days")
    return close_df, liquid_symbols

def run_test(close_df, signal_func, label, horizons=[1,5,20]):
    returns = np.log(close_df / close_df.shift(1)).dropna()
    market_ret = returns.mean(axis=1)
    
    # Build signal
    sig = signal_func(returns)
    
    results = []
    for h in horizons:
        future_mkt = market_ret.shift(-h).rolling(h).sum()
        future_excess = {}
        
        for col in returns.columns:
            future_stock = returns[col].shift(-h).rolling(h).sum()
            valid = sig[col].dropna() & future_stock.notna() & future_mkt.notna()
            if valid.sum() < 10:
                continue
            events = future_stock[valid] - future_mkt[valid]
            future_excess[col] = events.mean()
        
        if not future_excess:
            continue
        
        values = list(future_excess.values())
        t_stat, p_val = stats.ttest_1samp(values, 0)
        p_val = p_val / 2 if t_stat < 0 else 1 - p_val / 2  # one-sided
        
        results.append({
            'signal': label,
            'horizon': h,
            'mean_bps': np.mean(values) * 10000,
            't_stat': t_stat,
            'p_value': p_val,
            'n_events': len(values)
        })
    return results

def main():
    data_dir = 'data'
    dfs = read_stock_data(data_dir)
    close_df, symbols = align_data(dfs)
    
    if close_df is None:
        print("No data")
        return
    
    returns = np.log(close_df / close_df.shift(1)).dropna()
    
    # Signals
    def vol_spike(r):
        vol = r.abs().rolling(20, min_periods=10).mean()  # proxy: use return vol as volume proxy for speed
        # Actually we need real volume - let's skip if no volume in panel
        # Better: use return-based extreme for now
        return pd.DataFrame(False, index=r.index, columns=r.columns)
    
    # Since we filtered by volume but don't have cross-sectional volume panel easily,
    # let's use the two signals we care about with return-based proxies:
    
    # extreme_down: bottom 5% of 60-day rolling
    def extreme_down(r):
        q05 = r.rolling(60, min_periods=30).quantile(0.05)
        return r < q05
    
    # large_down: daily return < -5%
    def large_down(r):
        return r < -0.05
    
    all_results = []
    
    # extreme_down
    sig = extreme_down(returns)
    for h in [1, 5, 20]:
        future_mkt = returns.mean(axis=1).shift(-h).rolling(h).sum()
        excess = {}
        for col in returns.columns:
            f = returns[col].shift(-h).rolling(h).sum()
            valid = sig[col].dropna() & f.notna() & future_mkt.notna()
            if valid.sum() < 3:
                continue
            ev = (f[valid] - future_mkt[valid]).dropna()
            if len(ev) > 0:
                excess[col] = ev.mean()
        if excess:
            vals = list(excess.values())
            t, p = stats.ttest_1samp(vals, 0)
            p = p / 2 if t < 0 else 1 - p / 2
            all_results.append({
                'signal': 'extreme_down_reversal', 'horizon': h,
                'mean_bps': np.mean(vals)*10000, 't_stat': t,
                'p_value': p, 'n_events': len(vals)
            })
    
    # volume_spike proxy: day with >2x avg absolute return (high volatility = high volume proxy)
    vol_proxy = (returns.abs() > 2 * returns.abs().rolling(20, min_periods=10).mean())
    for h in [1, 5, 20]:
        future_mkt = returns.mean(axis=1).shift(-h).rolling(h).sum()
        excess = {}
        for col in returns.columns:
            f = returns[col].shift(-h).rolling(h).sum()
            valid = vol_proxy[col].dropna() & f.notna() & future_mkt.notna()
            if valid.sum() < 3:
                continue
            ev = (f[valid] - future_mkt[valid]).dropna()
            if len(ev) > 0:
                excess[col] = ev.mean()
        if excess:
            vals = list(excess.values())
            t, p = stats.ttest_1samp(vals, 0)
            p = p / 2 if t < 0 else 1 - p / 2
            all_results.append({
                'signal': 'volume_spike_proxy', 'horizon': h,
                'mean_bps': np.mean(vals)*10000, 't_stat': t,
                'p_value': p, 'n_events': len(vals)
            })
    
    # BH correction
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        _, q_vals, _, _ = multipletests(df['p_value'].values, method='fdr_bh', alpha=0.10)
        df['q_value_bh'] = q_vals
        df['significant'] = df['q_value_bh'] < 0.10
        
        print("\n=== LIQUIDITY-FILTERED RESULTS (Top 50% by volume) ===\n")
        print(df.to_string(index=False))
        
        sig = df[df['significant']]
        if len(sig) > 0:
            print("\n--- Significant after BH (q<0.10) ---")
            print(sig.to_string(index=False))
        else:
            print("\n--- No significant results after BH correction ---")
    else:
        print("No results")

if __name__ == '__main__':
    main()
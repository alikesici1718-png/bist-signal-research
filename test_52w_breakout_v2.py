"""52-week-high breakout test with regression controls.

Hypothesis: closes at/near the 52-week high predict positive forward
excess returns. Method: panel of breakout events, next-open entry, excess
return vs equal-weighted market, OLS with controls (volume, volatility)
in addition to plain t-tests. Console output.
Result: no robust edge after market adjustment (superseded by the
multi-lookback version, test_multi_lookback_breakout.py).
"""
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

def build_panels(dfs, min_history=252):
    valid = {s: df for s, df in dfs.items() if len(df) >= min_history + 25}
    
    close_df = pd.DataFrame({s: df['Close'] for s, df in valid.items()})
    open_df = pd.DataFrame({s: df['Open'] for s, df in valid.items()})
    volume_df = pd.DataFrame({s: df['Volume'] for s, df in valid.items()})
    
    common_idx = close_df.index.intersection(open_df.index).intersection(volume_df.index)
    close_df = close_df.loc[common_idx]
    open_df = open_df.loc[common_idx]
    volume_df = volume_df.loc[common_idx]
    
    close_df = close_df.dropna(axis=1, thresh=int(0.8 * len(close_df)))
    open_df = open_df[close_df.columns]
    volume_df = volume_df[close_df.columns]
    
    close_df = close_df.dropna(axis=0, thresh=int(0.8 * len(close_df.columns)))
    open_df = open_df.loc[close_df.index]
    volume_df = volume_df.loc[close_df.index]
    
    return close_df, open_df, volume_df

def signal_52w_high_breakout(close_df):
    high_52w = close_df.rolling(252, min_periods=126).max()
    breakout = (close_df > high_52w.shift(1)) & (close_df.shift(1) <= high_52w.shift(1))
    return breakout.fillna(False)

def dynamic_volume_filter(volume_df, method='median', top_n=None):
    vol_avg_60 = volume_df.rolling(60, min_periods=30).mean()
    
    if method == 'median':
        daily_median = vol_avg_60.median(axis=1)
        is_liquid = vol_avg_60.ge(daily_median, axis=0)
    elif method == 'top_n' and top_n:
        is_liquid = pd.DataFrame(False, index=vol_avg_60.index, columns=vol_avg_60.columns)
        for date in vol_avg_60.index:
            row = vol_avg_60.loc[date].dropna()
            if len(row) >= top_n:
                top_symbols = row.nlargest(top_n).index
                is_liquid.loc[date, top_symbols] = True
    else:
        is_liquid = pd.DataFrame(True, index=vol_avg_60.index, columns=vol_avg_60.columns)
    
    return is_liquid.fillna(False)

def run_test(close_df, open_df, signal_df, period_mask, tx_cost_bps=0, horizons=[5, 10, 20]):
    open_entry = open_df.shift(-1)
    
    for h in horizons:
        open_exit = open_df.shift(-(1 + h))
        forward_ret = np.log(open_exit / open_entry)
        
        mkt_ret = forward_ret.mean(axis=1)
        excess = forward_ret.sub(mkt_ret, axis=0)
        
        # FIX: period_mask can be Series or ndarray, handle both
        mask_arr = np.asarray(period_mask).reshape(-1, 1)
        period_df = pd.DataFrame(
            np.tile(mask_arr, (1, len(close_df.columns))),
            index=close_df.index, columns=close_df.columns)
        
        valid = signal_df & period_df & open_entry.notna() & open_exit.notna()
        valid = valid.fillna(False)
        
        daily_port = []
        for date in valid.index:
            mask_today = valid.loc[date].fillna(False)
            symbols_today = valid.columns[mask_today].tolist()
            if len(symbols_today) == 0:
                continue
            
            port_excess = excess.loc[date, symbols_today].mean()
            daily_port.append({
                'date': date,
                'n_signals': len(symbols_today),
                'excess_bps': port_excess * 10000 - tx_cost_bps
            })
        
        if len(daily_port) < 10:
            continue
        
        df_port = pd.DataFrame(daily_port).set_index('date')
        returns = df_port['excess_bps'].values
        
        X = np.ones(len(returns))
        model = sm.OLS(returns, X)
        try:
            nw = model.fit(cov_type='HAC', cov_kwds={'maxlags': h})
            t_stat = nw.tvalues[0]
            p_val = nw.pvalues[0]
        except Exception:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
        
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        
        yield {
            'horizon': h,
            'n_days': len(returns),
            'mean_excess_bps': returns.mean(),
            'median_excess_bps': np.median(returns),
            't_stat': t_stat,
            'p_one': p_one,
            'std_bps': returns.std(),
            'avg_signals_per_day': df_port['n_signals'].mean(),
            'max_signals_day': df_port['n_signals'].max()
        }

def main():
    data_dir = 'data'
    dfs = read_stock_data(data_dir)
    close_df, open_df, volume_df = build_panels(dfs)
    
    print(f"Panel: {close_df.shape[1]} symbols, {close_df.shape[0]} days")
    print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print()
    
    signal_raw = signal_52w_high_breakout(close_df)
    
    liquid_median = dynamic_volume_filter(volume_df, method='median')
    liquid_top30 = dynamic_volume_filter(volume_df, method='top_n', top_n=30)
    
    split_date = pd.Timestamp('2025-01-01')
    full_mask = pd.Series(True, index=close_df.index)
    older_mask = close_df.index < split_date
    recent_mask = close_df.index >= split_date
    
    tx_scenarios = [
        ('Gross (0 bps)', 0),
        ('Low (40 bps)', 40),
        ('Medium (70 bps)', 70),
        ('High (100 bps)', 100),
    ]
    
    configs = [
        ('All Symbols', signal_raw, full_mask),
        ('All Symbols (Older)', signal_raw, older_mask),
        ('All Symbols (Recent)', signal_raw, recent_mask),
        ('Dynamic Median Vol', signal_raw & liquid_median, full_mask),
        ('Dynamic Median Vol (Older)', signal_raw & liquid_median, older_mask),
        ('Dynamic Median Vol (Recent)', signal_raw & liquid_median, recent_mask),
        ('Dynamic Top 30 Vol', signal_raw & liquid_top30, full_mask),
        ('Dynamic Top 30 Vol (Older)', signal_raw & liquid_top30, older_mask),
        ('Dynamic Top 30 Vol (Recent)', signal_raw & liquid_top30, recent_mask),
    ]
    
    for name, sig, mask in configs:
        print(f"\n{'='*70}")
        print(f"=== {name} ===")
        print(f"{'='*70}")
        
        for tx_name, tx_cost in tx_scenarios:
            res = list(run_test(close_df, open_df, sig, mask, tx_cost_bps=tx_cost, horizons=[5, 10, 20]))
            if res:
                df = pd.DataFrame(res)
                print(f"\n  {tx_name}:")
                print(df[['horizon', 'n_days', 'mean_excess_bps', 't_stat', 'p_one', 'median_excess_bps', 'avg_signals_per_day']].to_string(index=False))
            else:
                print(f"\n  {tx_name}: Insufficient events (<10 days)")

if __name__ == '__main__':
    main()
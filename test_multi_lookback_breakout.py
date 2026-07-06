import os
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats

def read_stock_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = {}
    for f in files:
        if f in ('USDTRY.csv', 'USDTRY=X.csv', 'XU100.IS.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['Date'])
        df = df.set_index('Date')
        dfs[symbol] = df
    return dfs

def get_benchmark(data_dir, start, end):
    bench_path = os.path.join(data_dir, 'XU100.IS.csv')
    if os.path.exists(bench_path):
        df = pd.read_csv(bench_path, parse_dates=['Date']).set_index('Date')
        if 'Open' in df.columns:
            return df['Open'].squeeze()
    try:
        bench = yf.download('XU100.IS', start=start.strftime('%Y-%m-%d'), 
                           end=end.strftime('%Y-%m-%d'), progress=False)
        if not bench.empty and 'Open' in bench.columns:
            bench_df = bench[['Open']] if isinstance(bench, pd.DataFrame) else bench.to_frame('Open')
            bench_df.to_csv(bench_path)
            return bench_df['Open'].squeeze()
    except Exception:
        pass
    return None

def build_panels(dfs, min_history=200):
    valid = {s: df for s, df in dfs.items() if len(df) >= min_history}
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

def signal_high_breakout(close_df, lookback):
    high_rolling = close_df.rolling(lookback, min_periods=int(lookback*0.5)).max()
    breakout = (close_df > high_rolling.shift(1)) & (close_df.shift(1) <= high_rolling.shift(1))
    return breakout.fillna(False)

def dynamic_volume_filter(volume_df):
    vol_avg_60 = volume_df.rolling(60, min_periods=30).mean()
    daily_median = vol_avg_60.median(axis=1)
    return vol_avg_60.ge(daily_median, axis=0).fillna(False)

def run_test(close_df, open_df, signal_df, period_mask, bench_open, roundtrip_cost_bps=0, horizons=[5, 10, 20]):
    open_entry = open_df.shift(-1)
    bench_aligned = bench_open.reindex(open_df.index).ffill() if bench_open is not None else None
    
    for h in horizons:
        open_exit = open_df.shift(-(1 + h))
        forward_ret = np.log(open_exit / open_entry)
        
        if bench_aligned is not None:
            bench_entry = bench_aligned.shift(-1)
            bench_exit = bench_aligned.shift(-(1 + h))
            mkt_ret = np.log(bench_exit / bench_entry).squeeze()
        else:
            mkt_ret = forward_ret.mean(axis=1)
        
        # KONTROL: mkt_ret tipi
        if h == 5:
            print(f"      [DEBUG] type(mkt_ret)={type(mkt_ret)}")
        
        excess = forward_ret.sub(mkt_ret, axis=0)
        
        mask_arr = np.asarray(period_mask).reshape(-1, 1)
        period_df = pd.DataFrame(
            np.tile(mask_arr, (1, len(close_df.columns))),
            index=close_df.index, columns=close_df.columns)
        
        valid = signal_df & period_df & open_entry.notna() & open_exit.notna()
        valid = valid.fillna(False)
        
        valid_long = valid.stack()
        valid_long = valid_long[valid_long].index.to_frame(index=False)
        valid_long.columns = ['date', 'symbol']
        if len(valid_long) == 0:
            continue
        
        excess_long = excess.stack().reset_index()
        excess_long.columns = ['date', 'symbol', 'excess']
        
        merged = valid_long.merge(excess_long, on=['date', 'symbol'], how='left')
        merged = merged.dropna(subset=['excess'])
        if len(merged) == 0:
            continue
        
        merged['excess_bps'] = merged['excess'] * 10000 - roundtrip_cost_bps
        
        daily = merged.groupby('date')['excess_bps'].agg(['mean', 'count']).reset_index()
        daily.columns = ['date', 'port_excess_bps', 'n_signals']
        daily = daily.set_index('date')
        
        if len(daily) < 10:
            continue
        
        returns = daily['port_excess_bps'].values
        
        X = np.ones(len(returns))
        model = sm.OLS(returns, X)
        try:
            nw = model.fit(cov_type='HAC', cov_kwds={'maxlags': h})
            t_stat = nw.tvalues[0]
            p_val = nw.pvalues[0]
        except Exception:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
        
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        
        trade_mean = merged['excess_bps'].mean()
        trade_median = merged['excess_bps'].median()
        trade_win_rate = (merged['excess_bps'] > 0).mean() * 100
        
        yield {
            'horizon': h,
            'n_days': len(returns),
            'mean_excess_bps': returns.mean(),
            'median_excess_bps': np.median(returns),
            't_stat': t_stat,
            'p_one': p_one,
            'std_bps': returns.std(),
            'avg_signals_per_day': daily['n_signals'].mean(),
            'max_signals_day': daily['n_signals'].max(),
            'trade_mean_bps': trade_mean,
            'trade_median_bps': trade_median,
            'trade_win_rate': trade_win_rate,
            'n_trades': len(merged)
        }

def main():
    data_dir = 'data'
    dfs = read_stock_data(data_dir)
    close_df, open_df, volume_df = build_panels(dfs)
    
    bench_open = get_benchmark(data_dir, close_df.index[0], close_df.index[-1])
    
    # KONTROL: bench_open tipi
    print(f"="*70)
    print("KONTROL: bench_open")
    print(f"  type: {type(bench_open)}")
    print(f"  shape: {getattr(bench_open, 'shape', 'N/A')}")
    print(f"  head:\n{bench_open.head()}")
    print(f"="*70)
    print()
    
    print(f"Benchmark: {'BIST100 (XU100.IS)' if bench_open is not None else 'Panel equal-weighted'}")
    print(f"Panel: {close_df.shape[1]} symbols, {close_df.shape[0]} days")
    print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print("NOTE: Survivorship bias exists.")
    print()
    
    liquid_filter = dynamic_volume_filter(volume_df)
    
    split_date = pd.Timestamp('2025-01-01')
    full_mask = pd.Series(True, index=close_df.index)
    older_mask = close_df.index < split_date
    recent_mask = close_df.index >= split_date
    
    tx_scenarios = [('Gross', 0), ('RoundTrip 40bps', 40), ('RoundTrip 70bps', 70), ('RoundTrip 100bps', 100)]
    lookbacks = [('20w', 50), ('30w', 75), ('40w', 100), ('52w', 252)]
    
    for lb_name, lb_days in lookbacks:
        signal_raw = signal_high_breakout(close_df, lb_days)
        sig_liquid = signal_raw & liquid_filter
        
        print(f"\n{'='*70}")
        print(f"=== LOOKBACK: {lb_name} ({lb_days} days) ===")
        
        for period_name, mask in [('Full', full_mask), ('Older', older_mask), ('Recent', recent_mask)]:
            print(f"\n  --- {period_name} ---")
            
            for tx_name, tx_cost in tx_scenarios:
                res = list(run_test(close_df, open_df, sig_liquid, mask, bench_open,
                                   roundtrip_cost_bps=tx_cost, horizons=[5, 10, 20]))
                if res:
                    df = pd.DataFrame(res)
                    print(f"    {tx_name}: h={df['horizon'].tolist()}")
                    print(f"      daily_mean={df['mean_excess_bps'].round(1).tolist()}, p={df['p_one'].round(4).tolist()}")
                    print(f"      trade_mean={df['trade_mean_bps'].round(1).tolist()}, win_rate={df['trade_win_rate'].round(1).tolist()}%")
                    print(f"      sig/day={df['avg_signals_per_day'].round(1).tolist()}, n_trades={df['n_trades'].tolist()}")
                else:
                    print(f"    {tx_name}: Insufficient events")

if __name__ == '__main__':
    main()
import os
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from scipy import stats

# ==================== DATA LOADING ====================

def _find_date_col(df):
    for col in df.columns:
        if col.lower() in ('date', 'tarih', 'datetime', 'timestamp', 'time'):
            return col
    return None

def read_stock_data(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    dfs = {}
    for f in files:
        if f in ('USDTRY.csv', 'USDTRY=X.csv', 'XU100.IS.csv'):
            continue
        symbol = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, f))
        
        date_col = _find_date_col(df)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            
        dfs[symbol] = df
    return dfs

def get_benchmark(data_dir, start, end):
    bench_path = os.path.join(data_dir, 'XU100.IS.csv')
    if os.path.exists(bench_path):
        try:
            # Try yfinance multi-header format first
            df = pd.read_csv(bench_path, header=[0, 1])
            # Flatten multi-level columns
            df.columns = [' '.join(col).strip() if col[1] not in ['nan', 'NaN'] else col[0] 
                          for col in df.columns.values]
            # Find date column
            date_col = _find_date_col(df)
            if not date_col:
                date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
            # Find Open column (yfinance names it like "Open XU100.IS" or just "Open")
            open_col = next((c for c in df.columns if 'Open' in c), None)
            if open_col:
                return df[open_col].squeeze()
        except Exception:
            # Fallback to standard single-header CSV
            try:
                df = pd.read_csv(bench_path)
                date_col = _find_date_col(df)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
                else:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                if 'Open' in df.columns:
                    return df['Open'].squeeze()
            except Exception as e2:
                print(f"Warning: Local benchmark read failed ({e2}), trying yfinance...")
    
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

def build_panels(dfs, min_history=300):
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

# ==================== SIGNAL FAMILIES ====================

def signal_momentum(close_df, lookback, skip):
    past_end = close_df.shift(skip)
    past_start = close_df.shift(lookback + skip)
    past_ret = np.log(past_end / past_start)
    rank = past_ret.rank(axis=1, pct=True)
    return (rank >= 0.90).fillna(False)

def signal_mean_reversion(close_df, lookback):
    sma = close_df.rolling(lookback, min_periods=int(lookback * 0.5)).mean()
    std = close_df.rolling(lookback, min_periods=int(lookback * 0.5)).std()
    zscore = (close_df - sma) / std
    return ((zscore.shift(1) >= -2) & (zscore < -2)).fillna(False)

def signal_52w_high(close_df, lookback, volume_df=None):
    high_roll = close_df.rolling(lookback, min_periods=int(lookback * 0.5)).max()
    breakout = (close_df > high_roll.shift(1)) & (close_df.shift(1) <= high_roll.shift(1))
    
    if volume_df is not None:
        vol_avg = volume_df.rolling(20, min_periods=10).mean()
        vol_confirm = volume_df > vol_avg
        vol_confirm = vol_confirm.reindex(close_df.index).reindex(columns=close_df.columns).fillna(False)
        return (breakout & vol_confirm).fillna(False)
    
    return breakout.fillna(False)

def signal_volatility_compression(close_df, lookback, width):
    sma = close_df.rolling(lookback, min_periods=int(lookback * 0.5)).mean()
    std = close_df.rolling(lookback, min_periods=int(lookback * 0.5)).std()
    upper = sma + width * std
    lower = sma - width * std
    band_width = (upper - lower) / sma
    compression = band_width <= band_width.rolling(20, min_periods=10).quantile(0.2)
    breakout = (close_df > upper.shift(1)) & (close_df.shift(1) <= upper.shift(1))
    return (compression & breakout).fillna(False)

# ==================== TEST INFRASTRUCTURE ====================

def dynamic_volume_filter(volume_df):
    vol_avg_60 = volume_df.rolling(60, min_periods=30).mean()
    daily_median = vol_avg_60.median(axis=1)
    return vol_avg_60.ge(daily_median, axis=0).fillna(False)

def run_single_test(close_df, open_df, signal_df, period_mask, bench_open,
                    roundtrip_cost_bps=0, horizon=10):
    open_entry = open_df.shift(-1)
    open_exit = open_df.shift(-(1 + horizon))
    forward_ret = np.log(open_exit / open_entry)

    if bench_open is not None:
        bench_aligned = bench_open.reindex(open_df.index).ffill()
        bench_entry = bench_aligned.shift(-1)
        bench_exit = bench_aligned.shift(-(1 + horizon))
        mkt_ret = np.log(bench_exit / bench_entry).squeeze()
    else:
        mkt_ret = forward_ret.mean(axis=1)

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
        return None, None

    excess_long = excess.stack().reset_index()
    excess_long.columns = ['date', 'symbol', 'excess']

    merged = valid_long.merge(excess_long, on=['date', 'symbol'], how='left')
    merged = merged.dropna(subset=['excess'])
    if len(merged) == 0:
        return None, None

    merged['excess_bps'] = merged['excess'] * 10000 - roundtrip_cost_bps

    daily = merged.groupby('date')['excess_bps'].agg(['mean', 'count']).reset_index()
    daily.columns = ['date', 'port_excess_bps', 'n_signals']
    daily = daily.set_index('date')

    if len(daily) < 10:
        return None, None

    returns = daily['port_excess_bps'].values

    X = np.ones(len(returns))
    model = sm.OLS(returns, X)
    try:
        nw = model.fit(cov_type='HAC', cov_kwds={'maxlags': horizon})
        t_stat = nw.tvalues[0]
        p_val = nw.pvalues[0]
    except Exception:
        t_stat, p_val = stats.ttest_1samp(returns, 0)

    p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2

    summary = {
        'n_days': len(returns),
        'mean_excess_bps': returns.mean(),
        't_stat': t_stat,
        'p_one': p_one,
        'avg_signals_per_day': daily['n_signals'].mean(),
        'n_trades': len(merged),
        'win_rate': (merged['excess_bps'] > 0).mean() * 100
    }

    return summary, daily

# ==================== WALK-FORWARD ====================

def walk_forward(close_df, open_df, volume_df, bench_open,
                 train_days=756, test_days=126, step_days=63,
                 cost_bps=0):
    liquid_filter = dynamic_volume_filter(volume_df)

    family_configs = [
        ('Momentum', signal_momentum, [
            {'lookback': 252, 'skip': 20},
            {'lookback': 126, 'skip': 20},
            {'lookback': 63, 'skip': 20},
        ]),
        ('MeanReversion', signal_mean_reversion, [
            {'lookback': 5},
            {'lookback': 10},
            {'lookback': 20},
        ]),
        ('52wHigh', signal_52w_high, [
            {'lookback': 252},
        ]),
        ('VolCompression', signal_volatility_compression, [
            {'lookback': 20, 'width': 1.5},
            {'lookback': 20, 'width': 2.0},
            {'lookback': 40, 'width': 2.0},
        ]),
    ]

    horizons = [5, 10, 20, 40]
    total_models = sum(len(param_list) * len(horizons) for _, _, param_list in family_configs)
    n = len(close_df)

    all_fold_results = []

    start = train_days
    fold_idx = 1
    while start + test_days <= n:
        is_end = start
        oos_start = start
        oos_end = start + test_days

        is_close = close_df.iloc[:is_end]
        is_open = open_df.iloc[:is_end]
        is_liquid = liquid_filter.iloc[:is_end]
        is_bench = bench_open.iloc[:is_end] if bench_open is not None else None

        oos_open = open_df.iloc[oos_start:oos_end]
        oos_liquid = liquid_filter.iloc[oos_start:oos_end]
        oos_bench = bench_open.iloc[:oos_end] if bench_open is not None else None

        combined_close = close_df.iloc[:oos_end]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}: IS [{close_df.index[0].date()} - {close_df.index[is_end-1].date()}] "
              f"({is_end} days) | OOS [{close_df.index[oos_start].date()} - {close_df.index[oos_end-1].date()}] "
              f"({test_days} days)")
        print(f"{'='*70}")

        best_per_family = {}

        for family_name, signal_func, param_list in family_configs:
            best_t = -np.inf
            best_config = None
            best_is_summary = None
            best_is_daily = None

            for params in param_list:
                for horizon in horizons:
                    if family_name == '52wHigh':
                        signal_is = signal_func(is_close, volume_df=volume_df.iloc[:is_end], **params)
                    else:
                        signal_is = signal_func(is_close, **params)
                    
                    sig_liquid = signal_is & is_liquid

                    is_mask = pd.Series(True, index=is_close.index)
                    is_summary, is_daily = run_single_test(
                        is_close, is_open, sig_liquid, is_mask,
                        is_bench, roundtrip_cost_bps=cost_bps, horizon=horizon
                    )

                    if is_summary and is_summary['t_stat'] > best_t:
                        best_t = is_summary['t_stat']
                        best_config = {**params, 'horizon': horizon}
                        best_is_summary = is_summary
                        best_is_daily = is_daily

            if best_config:
                best_per_family[family_name] = {
                    'func': signal_func,
                    'params': best_config,
                    'is_summary': best_is_summary,
                    'is_daily': best_is_daily
                }
                param_str = '-'.join([f"{k}{v}" for k, v in best_config.items()])
                print(f"  IS BEST [{family_name}] -> {param_str} | "
                      f"mean={best_is_summary['mean_excess_bps']:.2f} bps, "
                      f"t={best_is_summary['t_stat']:.2f}, "
                      f"n={best_is_summary['n_trades']}")

        fold_oos_entries = []

        for family_name, config in best_per_family.items():
            signal_params = {k: v for k, v in config['params'].items() if k != 'horizon'}
            
            if family_name == '52wHigh':
                signal_full = config['func'](combined_close, volume_df=volume_df.iloc[:oos_end], **signal_params)
            else:
                signal_full = config['func'](combined_close, **signal_params)
            
            signal_oos = signal_full.iloc[oos_start:oos_end]
            sig_liquid = signal_oos & oos_liquid

            oos_mask = pd.Series(True, index=oos_open.index)
            oos_summary, oos_daily = run_single_test(
                close_df.iloc[oos_start:oos_end], oos_open, sig_liquid, oos_mask,
                oos_bench, roundtrip_cost_bps=cost_bps,
                horizon=config['params']['horizon']
            )

            if oos_summary:
                is_mean = config['is_summary']['mean_excess_bps']
                oos_mean = oos_summary['mean_excess_bps']
                if is_mean > 0:
                    decay = 100 * (1 - oos_mean / is_mean)
                else:
                    decay = np.nan

                all_fold_results.append({
                    'fold': fold_idx,
                    'fold_start': close_df.index[oos_start].date(),
                    'fold_end': close_df.index[oos_end-1].date(),
                    'family': family_name,
                    'params': '-'.join([f"{k}{v}" for k, v in config['params'].items() if k != 'horizon']),
                    'horizon': config['params']['horizon'],
                    'is_mean': is_mean,
                    'oos_mean': oos_mean,
                    'decay_pct': decay,
                    'is_t': config['is_summary']['t_stat'],
                    **oos_summary,
                    'oos_daily': oos_daily
                })

                fold_oos_entries.append({
                    'family': family_name,
                    'is_mean': is_mean,
                    'oos_mean': oos_mean,
                    'oos_t': oos_summary['t_stat'],
                    'decay': decay
                })

                print(f"  OOS      [{family_name}] -> "
                      f"mean={oos_mean:.2f} bps, "
                      f"t={oos_summary['t_stat']:.2f}, "
                      f"n={oos_summary['n_trades']}, "
                      f"win={oos_summary['win_rate']:.1f}%, "
                      f"decay={decay:.1f}%")

        if fold_oos_entries:
            print(f"\n  Fold {fold_idx} Summary ({total_models} models tested, per-family winner selected)")
            print(f"  {'Family':<15} {'IS Mean':>10} {'OOS Mean':>10} {'Decay %':>10} {'OOS t':>8}")
            print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
            for entry in fold_oos_entries:
                print(f"  {entry['family']:<15} {entry['is_mean']:>10.2f} {entry['oos_mean']:>10.2f} "
                      f"{entry['decay']:>10.1f} {entry['oos_t']:>8.2f}")
            
            winner = max(fold_oos_entries, key=lambda x: x['oos_t'])
            print(f"\n  >>> WINNER: {winner['family']} (OOS t={winner['oos_t']:.2f})")

        start += step_days
        fold_idx += 1

    return pd.DataFrame(all_fold_results)

# ==================== MAIN ====================

def main():
    data_dir = 'data'
    dfs = read_stock_data(data_dir)
    close_df, open_df, volume_df = build_panels(dfs)

    bench_open = get_benchmark(data_dir, close_df.index[0], close_df.index[-1])
    print(f"Benchmark: {'BIST100' if bench_open is not None else 'Panel mean'}")
    print(f"Panel: {close_df.shape[1]} symbols, {close_df.shape[0]} days")
    print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print("NOTE: All signals are LONG-ONLY. Walk-forward: expanding window, family-level selection.")
    print("IS signals computed ONLY on IS data. OOS signals computed on full history (IS+OOS).")

    # Auto-adjust train_days based on available data
    available_days = len(close_df)
    desired_train = 756
    test_days = 126
    step_days = 63
    
    if available_days < desired_train + test_days:
        print(f"\nWARNING: Only {available_days} days available. Need {desired_train}+{test_days} for default walk-forward.")
        train_days = max(252, available_days - test_days - 1)
        print(f"Auto-adjusted train_days to {train_days} (minimum viable for momentum).")
        if train_days < 252:
            print("ERROR: Not enough data. Need at least 378 days (252 train + 126 test).")
            print("Please download more historical data (e.g., 10 years).")
            return
    else:
        train_days = desired_train

    for cost_bps in [0, 15, 30, 45]:
        print(f"\n{'#'*70}")
        print(f"# WALK-FORWARD WITH {cost_bps} bps ROUND-TRIP COST")
        print(f"{'#'*70}")

        results_df = walk_forward(
            close_df, open_df, volume_df, bench_open,
            train_days=train_days, test_days=test_days, step_days=step_days,
            cost_bps=cost_bps
        )

        if results_df.empty:
            print("No walk-forward results.")
            continue

        print(f"\n{'='*70}")
        print("AGGREGATE OOS RESULTS BY FAMILY")
        print(f"{'='*70}")

        for family in results_df['family'].unique():
            fam = results_df[results_df['family'] == family]

            daily_list = fam['oos_daily'].tolist()
            if not daily_list or all(d is None for d in daily_list):
                continue

            combined_daily = pd.concat([d for d in daily_list if d is not None])
            if len(combined_daily) < 10:
                continue

            returns = combined_daily['port_excess_bps'].values
            X = np.ones(len(returns))
            model = sm.OLS(returns, X)
            try:
                med_horizon = fam['horizon'].median()
                nw = model.fit(cov_type='HAC', cov_kwds={'maxlags': int(med_horizon)})
                agg_t = nw.tvalues[0]
                agg_p = nw.pvalues[0]
            except Exception:
                agg_t, agg_p = stats.ttest_1samp(returns, 0)

            agg_p_one = agg_p / 2 if agg_t > 0 else 1 - agg_p / 2
            avg_decay = fam['decay_pct'].mean()

            print(f"\n{family}:")
            print(f"  Folds: {len(fam)}")
            print(f"  Total OOS days: {len(combined_daily)}")
            print(f"  Total trades: {fam['n_trades'].sum()}")
            print(f"  Mean excess: {returns.mean():.2f} bps/day")
            print(f"  t-stat: {agg_t:.2f}")
            print(f"  p-one: {agg_p_one:.4f}")
            print(f"  Win rate: {(returns > 0).mean()*100:.1f}%")
            print(f"  Avg signals/day: {combined_daily['n_signals'].mean():.1f}")
            print(f"  Avg decay IS->OOS: {avg_decay:.1f}%")
            print(f"  Most common params: {fam['params'].mode().values[0] if not fam['params'].mode().empty else 'N/A'}")
            print(f"  Most common horizon: {fam['horizon'].mode().values[0] if not fam['horizon'].mode().empty else 'N/A'}")

        out_file = f"walkforward_results_{cost_bps}bps.csv"
        save_df = results_df.drop(columns=['oos_daily'], errors='ignore')
        save_df.to_csv(out_file, index=False)
        print(f"\nSaved: {out_file}")

if __name__ == '__main__':
    main()
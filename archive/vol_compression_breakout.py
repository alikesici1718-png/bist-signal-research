import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
import logging

log_file = 'logs/vol_compression_breakout.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')


def read_data(data_dir):
    data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name != 'DSTKF.csv':
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            df = df.set_index('Date')
            data[symbol] = df
    return data


def compute_signals_for_symbol(df, vol_lookback, compression_pct, breakout_lookback, horizon):
    close = df['Close']
    high = df['High']
    log_ret = np.log(close / close.shift(1))

    volatility = log_ret.rolling(vol_lookback).std()

    vol_percentile = volatility.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    is_compressed = vol_percentile <= compression_pct

    rolling_high = high.shift(1).rolling(breakout_lookback).max()
    is_breakout = high > rolling_high

    signal = is_compressed & is_breakout

    open_ = df['Open']
    entry = open_.shift(-1)
    exit_ = open_.shift(-(horizon + 1))
    forward_return = np.log(exit_ / entry)

    return signal, forward_return


def run_backtest(data, vol_lookback, compression_pct, breakout_lookback, horizon, date_filter=None):
    all_signal_returns = []

    for symbol, df in data.items():
        if date_filter is not None:
            df = df.loc[(df.index >= date_filter[0]) & (df.index < date_filter[1])]
        if len(df) < vol_lookback + breakout_lookback + horizon + 60:
            continue

        signal, forward_return = compute_signals_for_symbol(
            df, vol_lookback, compression_pct, breakout_lookback, horizon
        )

        signal_dates = signal[signal].index
        for d in signal_dates:
            if d in forward_return.index and not pd.isna(forward_return.loc[d]):
                all_signal_returns.append({
                    'symbol': symbol,
                    'date': d,
                    'forward_return': forward_return.loc[d]
                })

    if len(all_signal_returns) < 5:
        return None

    returns_df = pd.DataFrame(all_signal_returns)
    returns = returns_df['forward_return'].values * 10000

    mean_bps = returns.mean()
    t_stat, p_value = ttest_1samp(returns, 0)

    symbol_counts = returns_df['symbol'].value_counts()
    max_symbol_share = symbol_counts.max() / len(returns_df) if len(returns_df) > 0 else np.nan

    return {
        'mean_return_bps': mean_bps,
        't_stat': t_stat,
        'p_value': p_value,
        'signal_count': len(returns_df),
        'unique_symbols': returns_df['symbol'].nunique(),
        'max_symbol_share': max_symbol_share
    }


def main():
    data_dir = 'data'
    data = read_data(data_dir)

    if not data:
        logging.error('No data found')
        return

    vol_lookback = 20
    compression_pct = 0.20
    breakout_lookbacks = [10, 20]
    horizons = [5, 10, 20]

    all_dates = sorted(set().union(*[df.index for df in data.values()]))
    mid_date = all_dates[len(all_dates) // 2]
    start_date = all_dates[0]
    end_date = all_dates[-1] + pd.Timedelta(days=1)

    results = []

    for breakout_lookback in breakout_lookbacks:
        for horizon in horizons:
            res_full = run_backtest(data, vol_lookback, compression_pct, breakout_lookback, horizon)
            if res_full:
                results.append({'breakout_lookback': breakout_lookback, 'horizon': horizon,
                                 'period': 'full', **res_full})

            res_older = run_backtest(data, vol_lookback, compression_pct, breakout_lookback, horizon,
                                      date_filter=(start_date, mid_date))
            if res_older:
                results.append({'breakout_lookback': breakout_lookback, 'horizon': horizon,
                                 'period': 'older', **res_older})

            res_recent = run_backtest(data, vol_lookback, compression_pct, breakout_lookback, horizon,
                                       date_filter=(mid_date, end_date))
            if res_recent:
                results.append({'breakout_lookback': breakout_lookback, 'horizon': horizon,
                                 'period': 'recent', **res_recent})

    results_df = pd.DataFrame(results)

    print('Volatility Compression Breakout Results')
    if not results_df.empty:
        print(results_df[['breakout_lookback', 'horizon', 'period', 'mean_return_bps',
                           't_stat', 'p_value', 'signal_count', 'unique_symbols',
                           'max_symbol_share']].to_string(index=False))
    else:
        print('No results generated - insufficient signals across all combinations')

    results_df.to_csv('vol_compression_breakout_results.csv', index=False)
    logging.info('Results saved')


if __name__ == '__main__':
    main()
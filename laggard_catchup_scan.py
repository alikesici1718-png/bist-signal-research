"""Laggard catch-up test: do stocks that lag a market rally catch up?

Hypothesis: when the equal-weighted market rallies over a lookback window,
the worst-performing (laggard) stocks subsequently outperform.
Method: cross-sectional laggard selection, next-open entry, excess return
vs equal-weighted market, t-test on overlapping and non-overlapping
samples across lookback/horizon grid. Output: laggard_catchup_results.csv.
Result: no edge — e.g. 5d/5d excess +3.6 bps, p=0.77.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import logging

log_file = 'logs/laggard_catchup_scan.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')


def read_data(data_dir):
    close_data = {}
    open_data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name != 'DSTKF.csv':
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            df = df.set_index('Date')
            close_data[symbol] = df['Close']
            open_data[symbol] = df['Open']
    return pd.DataFrame(close_data), pd.DataFrame(open_data)


def compute_cross_sectional_zscore(log_returns, lookback):
    cum_return = log_returns.rolling(lookback).sum()
    row_mean = cum_return.mean(axis=1)
    row_std = cum_return.std(axis=1)
    z_score = cum_return.sub(row_mean, axis=0).div(row_std, axis=0)
    return z_score


def compute_forward_return(open_prices, horizon):
    entry = open_prices.shift(-1)
    exit_ = open_prices.shift(-(horizon + 1))
    return np.log(exit_ / entry)


def run_test(z_score, forward_return, non_overlap=False, horizon=None):
    dates = z_score.index
    if non_overlap and horizon:
        dates = dates[::horizon]

    laggard_returns = []
    market_returns = []

    n_laggards = max(1, int(round(z_score.shape[1] * 0.2)))

    for date in dates:
        z_row = z_score.loc[date]
        fwd_row = forward_return.loc[date]

        if z_row.isna().all() or fwd_row.isna().all():
            continue

        laggard_symbols = z_row.nsmallest(n_laggards).index
        lag_vals = fwd_row.loc[laggard_symbols].dropna()
        mkt_vals = fwd_row.dropna()

        if len(lag_vals) == 0 or len(mkt_vals) == 0:
            continue

        laggard_returns.extend(lag_vals.tolist())
        market_returns.extend(mkt_vals.tolist())

    laggard_returns = np.array(laggard_returns)
    market_returns = np.array(market_returns)

    if len(laggard_returns) < 2 or len(market_returns) < 2:
        return None

    excess_bps = (laggard_returns.mean() - market_returns.mean()) * 10000
    t_stat, p_value = ttest_ind(laggard_returns, market_returns, equal_var=False)

    return {
        'excess_return_bps': excess_bps,
        't_stat': t_stat,
        'p_value': p_value,
        'signal_count': len(laggard_returns)
    }


def main():
    data_dir = 'data'
    df, open_df = read_data(data_dir)

    if df.empty:
        logging.error('No data found in data_dir')
        return

    df = df.dropna()

    lookbacks = [5, 10, 20]
    horizons = [5, 10, 20]

    log_returns_full = np.log(df / df.shift(1)).dropna()
    open_full = open_df.reindex(log_returns_full.index)

    mid_point = len(log_returns_full) // 2
    older = log_returns_full.iloc[:mid_point]
    recent = log_returns_full.iloc[mid_point:]

    splits = {
        'full': log_returns_full,
        'older': older,
        'recent': recent
    }
    open_splits = {
        'full': open_full,
        'older': open_full.iloc[:mid_point],
        'recent': open_full.iloc[mid_point:]
    }

    results = []

    for lookback in lookbacks:
        for horizon in horizons:
            for split_name, log_returns in splits.items():
                if len(log_returns) < lookback + horizon + 5:
                    continue

                z_score = compute_cross_sectional_zscore(log_returns, lookback)
                fwd_return = compute_forward_return(open_splits[split_name], horizon)

                res_overlap = run_test(z_score, fwd_return, non_overlap=False)
                if res_overlap:
                    results.append({
                        'lookback': lookback,
                        'horizon': horizon,
                        'split_type': split_name,
                        'sampling': 'overlapping',
                        **res_overlap
                    })

                if split_name == 'full':
                    res_non_overlap = run_test(z_score, fwd_return, non_overlap=True, horizon=horizon)
                    if res_non_overlap:
                        results.append({
                            'lookback': lookback,
                            'horizon': horizon,
                            'split_type': split_name,
                            'sampling': 'non_overlapping',
                            **res_non_overlap
                        })

    results_df = pd.DataFrame(results)

    print('Laggard Catchup Results')
    print(results_df[['lookback', 'horizon', 'split_type', 'sampling',
                       'excess_return_bps', 't_stat', 'p_value', 'signal_count']]
          .to_string(index=False))

    results_df.to_csv('laggard_catchup_results.csv', index=False)
    logging.info('Results saved to laggard_catchup_results.csv')


if __name__ == '__main__':
    main()
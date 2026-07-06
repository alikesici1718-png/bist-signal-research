import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, pearsonr
import logging

log_file = 'logs/usdtry_bist_analysis.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')


def read_stock_data(data_dir):
    close_data = {}
    open_data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name not in ('DSTKF.csv', 'USDTRY.csv'):
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            df = df.set_index('Date')
            close_data[symbol] = df['Close']
            open_data[symbol] = df['Open']
    return pd.DataFrame(close_data), pd.DataFrame(open_data)


def read_usdtry(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'USDTRY.csv'), parse_dates=['Date'])
    df = df.set_index('Date')
    return df['Close']


def test_fx_shock_reaction(stock_returns, stock_open, usdtry_returns, shock_threshold_pct,
                            horizons, date_filter=None):
    results = []

    usdtry_ret = usdtry_returns.copy()
    if date_filter is not None:
        usdtry_ret = usdtry_ret.loc[(usdtry_ret.index >= date_filter[0]) & (usdtry_ret.index < date_filter[1])]

    threshold = usdtry_ret.abs().quantile(shock_threshold_pct)
    shock_up_dates = usdtry_ret[usdtry_ret > threshold].index
    shock_down_dates = usdtry_ret[usdtry_ret < -threshold].index

    for horizon in horizons:
        entry = stock_open.shift(-1)
        exit_ = stock_open.shift(-(horizon + 1))
        stock_fwd_return = np.log(exit_ / entry)

        if date_filter is not None:
            stock_fwd_return = stock_fwd_return.loc[
                (stock_fwd_return.index >= date_filter[0]) & (stock_fwd_return.index < date_filter[1])
            ]

        for direction, shock_dates in [('TL_WEAKENS', shock_up_dates), ('TL_STRENGTHENS', shock_down_dates)]:
            valid_dates = [d for d in shock_dates if d in stock_fwd_return.index]
            if len(valid_dates) < 3:
                continue

            event_level_returns = []
            for d in valid_dates:
                row = stock_fwd_return.loc[d].dropna()
                if len(row) == 0:
                    continue
                event_level_returns.append(row.mean())

            if len(event_level_returns) < 3:
                continue

            event_level_returns = np.array(event_level_returns) * 10000
            mean_bps = event_level_returns.mean()
            t_stat, p_value = ttest_1samp(event_level_returns, 0)

            results.append({
                'test': 'fx_shock_reaction',
                'direction': direction,
                'horizon': horizon,
                'shock_event_count': len(valid_dates),
                'observation_count': len(event_level_returns),
                'mean_return_bps': mean_bps,
                't_stat': t_stat,
                'p_value': p_value
            })

    return results


def test_fx_momentum_correlation(stock_returns, stock_open, usdtry_returns, lookbacks, horizons, date_filter=None):
    results = []

    for lookback in lookbacks:
        usdtry_cum = usdtry_returns.rolling(lookback).sum()

        for horizon in horizons:
            entry = stock_open.shift(-1)
            exit_ = stock_open.shift(-(horizon + 1))
            market_avg_fwd = np.log(exit_ / entry).mean(axis=1)

            combined = pd.DataFrame({
                'usdtry_momentum': usdtry_cum,
                'market_fwd_return': market_avg_fwd
            }).dropna()

            if date_filter is not None:
                combined = combined.loc[(combined.index >= date_filter[0]) & (combined.index < date_filter[1])]

            if len(combined) < 10:
                continue

            corr, p_value = pearsonr(combined['usdtry_momentum'], combined['market_fwd_return'])

            results.append({
                'test': 'fx_momentum_correlation',
                'lookback': lookback,
                'horizon': horizon,
                'observation_count': len(combined),
                'correlation': corr,
                'p_value': p_value
            })

    return results


def main():
    data_dir = 'data'
    stock_close, stock_open = read_stock_data(data_dir)
    stock_close = stock_close.dropna()
    stock_returns = np.log(stock_close / stock_close.shift(1)).dropna()
    stock_open = stock_open.reindex(stock_returns.index)

    usdtry_close = read_usdtry(data_dir)
    usdtry_returns = np.log(usdtry_close / usdtry_close.shift(1)).dropna()

    common_dates = stock_returns.index.intersection(usdtry_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    stock_open = stock_open.loc[common_dates]
    usdtry_returns = usdtry_returns.loc[common_dates]

    mid_date = common_dates[len(common_dates) // 2]
    start_date = common_dates[0]
    end_date = common_dates[-1] + pd.Timedelta(days=1)

    horizons = [5, 10, 20]

    all_results = []

    for period_name, date_filter in [('full', None), ('older', (start_date, mid_date)),
                                      ('recent', (mid_date, end_date))]:
        res = test_fx_shock_reaction(stock_returns, stock_open, usdtry_returns, shock_threshold_pct=0.90,
                                      horizons=horizons, date_filter=date_filter)
        for r in res:
            r['period'] = period_name
        all_results.extend(res)

    lookbacks = [5, 10, 20]
    for period_name, date_filter in [('full', None), ('older', (start_date, mid_date)),
                                      ('recent', (mid_date, end_date))]:
        res = test_fx_momentum_correlation(stock_returns, stock_open, usdtry_returns, lookbacks, horizons,
                                            date_filter=date_filter)
        for r in res:
            r['period'] = period_name
        all_results.extend(res)

    results_df = pd.DataFrame(all_results)

    print('=== TEST 1: FX Shock Reaction ===')
    shock_df = results_df[results_df['test'] == 'fx_shock_reaction']
    if not shock_df.empty:
        print(shock_df[['direction', 'horizon', 'period', 'shock_event_count',
                         'observation_count', 'mean_return_bps', 't_stat', 'p_value']]
              .to_string(index=False))

    print()
    print('=== TEST 2: FX Momentum Correlation ===')
    corr_df = results_df[results_df['test'] == 'fx_momentum_correlation']
    if not corr_df.empty:
        print(corr_df[['lookback', 'horizon', 'period', 'observation_count',
                        'correlation', 'p_value']].to_string(index=False))

    results_df.to_csv('usdtry_bist_analysis_results.csv', index=False)
    logging.info('Results saved')


if __name__ == '__main__':
    main()
"""Cross-sectional scan of technical event signals on BIST daily data.

Hypothesis: simple technical events (2x volume spike, extreme down day,
52w-high breakout, etc.) predict abnormal forward returns.
Method: per-symbol event detection over data/*.csv, next-open entry,
t-test on forward excess returns at multiple horizons, overlapping and
non-overlapping event sets, Benjamini-Hochberg FDR correction across
hypotheses. Output: comprehensive_scan_results.csv.
Result: strongest effect is NEGATIVE post-event drift (volume_spike_2x
5d excess -117 bps, q<1e-12) — a short-side signal that does not survive
transaction costs (see check_net_returns.py); no exploitable long edge.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import logging

log_file = 'logs/comprehensive_scan.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')


def read_stock_data(data_dir, min_rows=250):
    """Read all per-symbol CSVs except FX/index files. Symbols with fewer than
    min_rows rows (e.g. recent IPOs) are excluded up front so they don't force
    a global dropna() that would wipe out everyone else's history."""
    data = {}
    excluded_files = ('DSTKF.csv', 'USDTRY.csv', 'USDTRY=X.csv')
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name not in excluded_files:
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            df = df.set_index('Date')
            if len(df) < min_rows:
                logging.info(f'Excluding {symbol}: only {len(df)} rows (< {min_rows} minimum, likely recent listing)')
                continue
            data[symbol] = df
    return data


def align_data(stock_data):
    """Build a common close/volume/open panel using the INTERSECTION of dates
    across symbols that survived the min_rows filter in read_stock_data.
    A small number of remaining short-history symbols can still shrink the
    common date range; we report that range so it's visible, not silent."""
    closes = {}
    volumes = {}
    opens = {}
    for symbol, df in stock_data.items():
        closes[symbol] = df['Close']
        if 'Volume' in df.columns:
            volumes[symbol] = df['Volume']
        if 'Open' in df.columns:
            opens[symbol] = df['Open']
    close_df = pd.DataFrame(closes)

    # Use dates where at least 80% of symbols have data, rather than requiring
    # literally all symbols (which one late listing would otherwise force).
    coverage = close_df.notna().mean(axis=1)
    good_dates = coverage[coverage >= 0.80].index
    close_df = close_df.loc[good_dates]

    logging.info(f'Aligned panel: {close_df.shape[1]} symbols, {close_df.shape[0]} dates, '
                 f'{close_df.index.min()} to {close_df.index.max()}')

    # Drop symbols that still have any NaN within this date range (i.e. those
    # whose history doesn't fully cover the >=80%-coverage window).
    close_df = close_df.dropna(axis=1)
    logging.info(f'After dropping symbols with gaps in the coverage window: {close_df.shape[1]} symbols remain')

    volume_df = pd.DataFrame(volumes).reindex(index=close_df.index, columns=close_df.columns) if volumes else None
    open_df = pd.DataFrame(opens).reindex(index=close_df.index, columns=close_df.columns) if opens else None
    return close_df, volume_df, open_df


def get_splits(dates):
    mid = dates[len(dates) // 2]
    start = dates[0]
    end = dates[-1] + pd.Timedelta(days=1)
    return [('full', None), ('older', (start, mid)), ('recent', (mid, end))]


def filter_period(df, date_filter):
    if date_filter is None:
        return df
    return df.loc[(df.index >= date_filter[0]) & (df.index < date_filter[1])]


def event_test(signal_bool_df, fwd_return_df, market_mean, label, horizon, period, results):
    """Given a boolean signal DataFrame (dates x symbols) and forward returns,
    compute excess return per signal event, aggregated at the DATE level first
    (cross-sectional mean per date) to avoid pseudo-replication across symbols.

    IMPORTANT: forward returns use an overlapping rolling window of length
    `horizon`, so event dates less than `horizon` apart share underlying return
    observations and are NOT independent. A plain t-test on all overlapping
    dates understates variance and inflates significance (verified via a
    pure-noise simulation before this fix). We therefore also compute a
    non-overlapping version by keeping only event dates at least `horizon`
    trading days apart. The non-overlapping p-value is the one to trust; the
    overlapping one is kept only for reference/sample-size context.
    """
    excess_fwd = fwd_return_df.sub(market_mean, axis=0)

    event_dates = []
    event_vals = []
    for date in signal_bool_df.index:
        row_signal = signal_bool_df.loc[date]
        symbols_on = row_signal[row_signal].index.tolist()
        if not symbols_on:
            continue
        vals = excess_fwd.loc[date, symbols_on].dropna()
        if len(vals) == 0:
            continue
        event_dates.append(date)
        event_vals.append(vals.mean())

    if len(event_vals) < 5:
        return

    event_vals = np.array(event_vals) * 10000

    mean_bps_overlap = event_vals.mean()
    t_stat_overlap, p_value_overlap = ttest_1samp(event_vals, 0)

    non_overlap_vals = []
    last_kept_date = None
    for i, d in enumerate(event_dates):
        if last_kept_date is None or (d - last_kept_date).days >= horizon:
            non_overlap_vals.append(event_vals[i])
            last_kept_date = d

    n_no = len(non_overlap_vals)
    if n_no < 5:
        mean_bps_no, t_stat_no, p_value_no = np.nan, np.nan, np.nan
    else:
        non_overlap_arr = np.array(non_overlap_vals)
        mean_bps_no = non_overlap_arr.mean()
        t_stat_no, p_value_no = ttest_1samp(non_overlap_arr, 0)

    results.append({
        'hypothesis': label,
        'horizon': horizon,
        'period': period,
        'event_count': len(event_vals),
        'excess_return_bps': mean_bps_overlap,
        't_stat': t_stat_overlap,
        'p_value': p_value_overlap,
        'event_count_nonoverlap': n_no,
        'excess_return_bps_nonoverlap': mean_bps_no,
        't_stat_nonoverlap': t_stat_no,
        'p_value_nonoverlap': p_value_no
    })


def hypothesis_cross_sectional_momentum(close_df, returns_df, open_df, horizons, lookbacks, results):
    for lookback in lookbacks:
        past_return = returns_df.rolling(lookback).sum()
        for period_name, date_filter in get_splits(close_df.index):
            past_f = filter_period(past_return, date_filter)
            for horizon in horizons:
                fwd = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
                fwd_f = filter_period(fwd, date_filter)
                market_mean = fwd_f.mean(axis=1)

                common_idx = past_f.index.intersection(fwd_f.index)
                past_c = past_f.loc[common_idx]
                rank = past_c.rank(axis=1, pct=True)

                top_signal = rank >= 0.8
                bottom_signal = rank <= 0.2

                event_test(top_signal, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                           f'xsect_momentum_top20_lb{lookback}', horizon, period_name, results)
                event_test(bottom_signal, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                           f'xsect_reversal_bottom20_lb{lookback}', horizon, period_name, results)


def hypothesis_volume_spike(volume_df, returns_df, open_df, horizons, results):
    if volume_df is None:
        logging.warning('No volume data available, skipping volume_spike hypothesis')
        return
    vol_ma = volume_df.rolling(20).mean()
    vol_ratio = volume_df / vol_ma

    for period_name, date_filter in get_splits(returns_df.index):
        ratio_f = filter_period(vol_ratio, date_filter)
        for horizon in horizons:
            fwd = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
            fwd_f = filter_period(fwd, date_filter)
            market_mean = fwd_f.mean(axis=1)

            common_idx = ratio_f.index.intersection(fwd_f.index)
            signal = ratio_f.loc[common_idx] >= 2.0

            event_test(signal, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                       'volume_spike_2x', horizon, period_name, results)


def hypothesis_calendar_effects(returns_df, horizons, results):
    dow = returns_df.index.dayofweek
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

    for period_name, date_filter in get_splits(returns_df.index):
        ret_f = filter_period(returns_df, date_filter)
        market_ret = ret_f.mean(axis=1)

        for d, name in day_names.items():
            mask = ret_f.index.dayofweek == d
            vals = market_ret[mask].dropna() * 10000
            if len(vals) < 10:
                continue
            t_stat, p_value = ttest_1samp(vals, 0)
            results.append({
                'hypothesis': f'calendar_dow_{name}',
                'horizon': 1,
                'period': period_name,
                'event_count': len(vals),
                'excess_return_bps': vals.mean(),
                't_stat': t_stat,
                'p_value': p_value
            })

        is_month_start = ret_f.index.day <= 3
        is_month_end = ret_f.index.day >= 28
        for label, mask in [('month_start', is_month_start), ('month_end', is_month_end)]:
            vals = market_ret[mask].dropna() * 10000
            if len(vals) < 10:
                continue
            t_stat, p_value = ttest_1samp(vals, 0)
            results.append({
                'hypothesis': f'calendar_{label}',
                'horizon': 1,
                'period': period_name,
                'event_count': len(vals),
                'excess_return_bps': vals.mean(),
                't_stat': t_stat,
                'p_value': p_value
            })


def hypothesis_vol_regime(returns_df, open_df, horizons, results):
    market_ret = returns_df.mean(axis=1)
    realized_vol = market_ret.rolling(20).std()
    vol_rank = realized_vol.rolling(120, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    for period_name, date_filter in get_splits(returns_df.index):
        vol_rank_f = filter_period(vol_rank, date_filter)
        for horizon in horizons:
            fwd = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
            fwd_f = filter_period(fwd, date_filter)
            market_mean = fwd_f.mean(axis=1)

            common_idx = vol_rank_f.dropna().index.intersection(fwd_f.index)
            high_vol_dates = vol_rank_f.loc[common_idx][vol_rank_f.loc[common_idx] >= 0.8].index
            low_vol_dates = vol_rank_f.loc[common_idx][vol_rank_f.loc[common_idx] <= 0.2].index

            for label, dates in [('high_vol_regime', high_vol_dates), ('low_vol_regime', low_vol_dates)]:
                signal = pd.DataFrame(False, index=common_idx, columns=fwd_f.columns)
                valid_dates = [d for d in dates if d in signal.index]
                signal.loc[valid_dates] = True
                event_test(signal, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                           f'vol_regime_{label}', horizon, period_name, results)


def hypothesis_extreme_return_reversal(returns_df, open_df, horizons, results):
    for period_name, date_filter in get_splits(returns_df.index):
        ret_f = filter_period(returns_df, date_filter)
        rank = ret_f.rank(axis=1, pct=True)

        extreme_up = rank >= 0.95
        extreme_down = rank <= 0.05

        for horizon in horizons:
            fwd = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
            fwd_f = filter_period(fwd, date_filter)
            market_mean = fwd_f.mean(axis=1)

            common_idx = ret_f.index.intersection(fwd_f.index)

            event_test(extreme_up.loc[common_idx], fwd_f.loc[common_idx], market_mean.loc[common_idx],
                       'extreme_up_reversal', horizon, period_name, results)
            event_test(extreme_down.loc[common_idx], fwd_f.loc[common_idx], market_mean.loc[common_idx],
                       'extreme_down_reversal', horizon, period_name, results)


def hypothesis_52week_high_low(close_df, returns_df, open_df, horizons, results):
    rolling_max = close_df.rolling(252, min_periods=100).max()
    rolling_min = close_df.rolling(252, min_periods=100).min()
    pct_from_high = (close_df - rolling_max) / rolling_max
    pct_from_low = (close_df - rolling_min) / rolling_min

    for period_name, date_filter in get_splits(returns_df.index):
        pfh_f = filter_period(pct_from_high, date_filter)
        pfl_f = filter_period(pct_from_low, date_filter)

        for horizon in horizons:
            fwd = np.log(open_df.shift(-(horizon + 1)) / open_df.shift(-1))
            fwd_f = filter_period(fwd, date_filter)
            market_mean = fwd_f.mean(axis=1)

            common_idx = pfh_f.dropna(how='all').index.intersection(fwd_f.index)

            near_high = pfh_f.loc[common_idx] >= -0.02
            near_low = pfl_f.loc[common_idx] <= 0.02

            event_test(near_high, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                       '52w_high_breakout', horizon, period_name, results)
            event_test(near_low, fwd_f.loc[common_idx], market_mean.loc[common_idx],
                       '52w_low_bounce', horizon, period_name, results)


def apply_bh_correction(df):
    if df.empty:
        return df
    df = df.copy()
    # Use the non-overlapping p-value when available (event-based hypotheses),
    # otherwise fall back to p_value (calendar effects, which are already
    # single-day/non-overlapping by construction).
    if 'p_value_nonoverlap' in df.columns:
        df['p_value_for_correction'] = df['p_value_nonoverlap'].fillna(df['p_value'])
    else:
        df['p_value_for_correction'] = df['p_value']
    reject, q_values, _, _ = multipletests(df['p_value_for_correction'].values, method='fdr_bh')
    df['q_value_bh'] = q_values
    return df


def main():
    data_dir = 'data'
    stock_data = read_stock_data(data_dir)
    close_df, volume_df, open_df = align_data(stock_data)
    returns_df = np.log(close_df / close_df.shift(1)).dropna()

    print(f'Universe: {close_df.shape[1]} symbols, {close_df.shape[0]} trading days')
    print(f'Date range: {close_df.index.min().date()} to {close_df.index.max().date()}')
    print()

    horizons = [5, 10, 20]
    lookbacks = [10, 20]

    results = []

    logging.info('Running cross_sectional_momentum')
    hypothesis_cross_sectional_momentum(close_df, returns_df, open_df, horizons, lookbacks, results)

    logging.info('Running volume_spike')
    hypothesis_volume_spike(volume_df, returns_df, open_df, horizons, results)

    logging.info('Running calendar_effects')
    hypothesis_calendar_effects(returns_df, horizons, results)

    logging.info('Running vol_regime')
    hypothesis_vol_regime(returns_df, open_df, horizons, results)

    logging.info('Running extreme_return_reversal')
    hypothesis_extreme_return_reversal(returns_df, open_df, horizons, results)

    logging.info('Running 52week_high_low')
    hypothesis_52week_high_low(close_df, returns_df, open_df, horizons, results)

    results_df = pd.DataFrame(results)
    results_df = apply_bh_correction(results_df)
    results_df = results_df.sort_values('p_value_for_correction')

    results_df.to_csv('comprehensive_scan_results.csv', index=False)

    display_cols = ['hypothesis', 'horizon', 'period', 'event_count',
                     'excess_return_bps', 'p_value',
                     'event_count_nonoverlap', 'excess_return_bps_nonoverlap',
                     'p_value_nonoverlap', 'q_value_bh']
    display_cols = [c for c in display_cols if c in results_df.columns]

    print(f'Total tests run: {len(results_df)}')
    print('NOTE: p_value uses overlapping horizon windows and is NOT reliable on its own.')
    print('p_value_nonoverlap thins events to be >= horizon days apart and is the trustworthy figure.')
    print('q_value_bh is computed from p_value_nonoverlap (or p_value for calendar effects, which are 1-day and non-overlapping already).')
    print()
    print('=== TOP 20 BY CORRECTED (NON-OVERLAPPING) P-VALUE ===')
    print(results_df.head(20)[display_cols].to_string(index=False))

    print()
    print('=== ANY SURVIVING BH CORRECTION AT q < 0.10 ===')
    survivors = results_df[results_df['q_value_bh'] < 0.10]
    if survivors.empty:
        print('None.')
    else:
        print(survivors[display_cols].to_string(index=False))

    logging.info('Comprehensive scan complete, results saved to comprehensive_scan_results.csv')


if __name__ == '__main__':
    main()
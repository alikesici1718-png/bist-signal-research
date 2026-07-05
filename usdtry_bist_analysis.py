def test_fx_shock_reaction(stock_returns, usdtry_returns, shock_threshold_pct,
                            horizons, date_filter=None):
    results = []

    usdtry_ret = usdtry_returns.copy()
    if date_filter is not None:
        usdtry_ret = usdtry_ret.loc[(usdtry_ret.index >= date_filter[0]) & (usdtry_ret.index < date_filter[1])]

    threshold = usdtry_ret.abs().quantile(shock_threshold_pct)
    shock_up_dates = usdtry_ret[usdtry_ret > threshold].index
    shock_down_dates = usdtry_ret[usdtry_ret < -threshold].index

    for horizon in horizons:
        stock_fwd_return = stock_returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))

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

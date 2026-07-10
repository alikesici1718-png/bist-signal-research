"""Diagnostic: list USDTRY shock dates used by the FX-shock event study.

Flags days where the absolute USDTRY log return exceeds its 90th
percentile and prints the TL-strengthening shock dates with magnitudes,
for manual verification against known macro events (CBRT interventions,
elections). Feeds usdtry_bist_analysis.py. Console output only.
"""
import os
import pandas as pd
import numpy as np

def read_usdtry(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'USDTRY.csv'), parse_dates=['Date'])
    df = df.set_index('Date')
    return df['Close']


def main():
    data_dir = 'data'
    usdtry_close = read_usdtry(data_dir)
    usdtry_returns = np.log(usdtry_close / usdtry_close.shift(1)).dropna()

    threshold = usdtry_returns.abs().quantile(0.90)
    shock_down_dates = usdtry_returns[usdtry_returns < -threshold]

    print(f'Threshold (90th percentile abs return): {threshold:.6f}')
    print(f'Number of TL_STRENGTHENS shock events: {len(shock_down_dates)}')
    print()
    print('Dates and magnitude:')
    print(shock_down_dates.sort_values().to_string())

    print()
    print('Year distribution:')
    print(shock_down_dates.index.year.value_counts().sort_index())

    print()
    print('Month distribution:')
    year_month = shock_down_dates.index.to_period('M')
    print(year_month.value_counts().sort_index())


if __name__ == '__main__':
    main()
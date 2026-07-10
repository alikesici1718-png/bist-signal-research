"""Pairwise cointegration scan across BIST symbols (pairs-trading candidates).

Hypothesis: some BIST stock pairs are cointegrated and thus candidates for
mean-reversion pairs trading. Method: Engle-Granger cointegration test on
price pairs with Benjamini-Hochberg FDR correction across all pairs.
Output: cointegration_results.csv (screening only — no pairs backtest here).
"""
import os
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests
import numpy as np
import logging

# Log dosyasını oluştur
log_file = 'logs/cointegration_scan.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(data_dir):
    data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            data[symbol] = df['Close']
    return pd.DataFrame(data)

def calculate_cointegration(symbol1, symbol2):
    try:
        result = coint(symbol1, symbol2)
        p_value = result[1]
        test_statistic = result[0]
        if p_value < 0.05:
            feasibility = 'FEASIBLE_WEAK'
        elif p_value < 0.01:
            feasibility = 'FEASIBLE_STRONG'
        else:
            feasibility = 'NOT_FEASIBLE'
        return symbol1, symbol2, p_value, test_statistic, feasibility
    except Exception as e:
        logging.error(f'Error calculating cointegration for {symbol1} and {symbol2}: {e}')
        return symbol1, symbol2, None, None, 'ERROR'

def main():
    data_dir = 'data'
    
    # Data oku
    df = read_data(data_dir)
    
    if df.empty:
        logging.error('No data found in the data directory')
        return
    
    # Ortak tarih aralığına sahip satırları kullan
    common_dates = df.dropna().index
    df = df.loc[common_dates]
    
    # Tüm ikili kombinasyonları tara
    results = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            symbol1 = df.columns[i]
            symbol2 = df.columns[j]
            result = calculate_cointegration(df[symbol1], df[symbol2])
            p_value, test_statistic, feasibility = result[2:]
            results.append({'symbol1': str(symbol1), 'symbol2': str(symbol2), 'p_value': p_value, 'test_statistic': test_statistic})
    
    # Sonuçları DataFrame'e topla
    coint_results = pd.DataFrame(results)
    
    # Benjamini-Hochberg düzeltmesi ekle
    reject_bh, pvals_corrected_bh, _, _ = multipletests(coint_results['p_value'], alpha=0.05, method='fdr_bh')
    coint_results['q_value_bh'] = pvals_corrected_bh
    
    # Bonferroni düzeltmesi ekle
    reject_bonf, pvals_corrected_bonf, _, _ = multipletests(coint_results['p_value'], alpha=0.05, method='bonferroni')
    coint_results['q_value_bonferroni'] = pvals_corrected_bonf
    
    # Kaç çift BH q<0.05'i geçiyor, kaç çift Bonferroni'yi geçiyor ayrı ayrı say ve yazdır
    bh_count = (coint_results['q_value_bh'] < 0.05).sum()
    bonferroni_count = (coint_results['q_value_bonferroni'] < 0.05).sum()
    print(f'Number of pairs passing BH q<0.05: {bh_count}')
    print(f'Number of pairs passing Bonferroni q<0.05: {bonferroni_count}')
    
    # Economic sanity check ekle
    for index, row in coint_results.iterrows():
        symbol1 = row['symbol1']
        symbol2 = row['symbol2']
        hedge_ratio = -row['test_statistic'] / row['p_value']
        spread = df[symbol1] - hedge_ratio * df[symbol2]
        std_dev = spread.std()
        daily_change_mean = spread.diff().abs().mean() * 100
        if daily_change_mean < 5:
            print(f'Warning: Economic sanity check failed for {symbol1} and {symbol2}. Daily change mean is too small.')
    
    # Sonuçları kaydet
    coint_results.to_csv('cointegration_results.csv', index=False)

if __name__ == '__main__':
    main()

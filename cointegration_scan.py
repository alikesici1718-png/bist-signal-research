import os
import pandas as pd
from statsmodels.tsa.stattools import coint
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
            results.append({'symbol1': str(symbol1), 'symbol2': str(symbol2), 'p_value': p_value, 'test_statistic': test_stat})
    
    # Sonuçları DataFrame'e topla
    coint_results = pd.DataFrame(results)
    
    # p-value'ya göre sırala ve en anlamlı ilk 20 çifti konsola yazdır
    top_20_pairs = coint_results.sort_values(by='p_value').head(20)
    print('Top 20 Cointegrated Pairs:')
    print(top_20_pairs[['symbol1', 'symbol2', 'p_value', 'test_statistic', 'feasibility']].to_string(index=False))
    
    # Tüm sonuçları kaydet
    coint_results.to_csv('cointegration_results.csv', index=False)
    
    # Özet istatistik yazdır
    summary = {
        'total_pairs': len(coint_results),
        'feasible_strong_count': (coint_results['feasibility'] == 'FEASIBLE_STRONG').sum(),
        'feasible_weak_count': (coint_results['feasibility'] == 'FEASIBLE_WEAK').sum()
    }
    print('Summary Statistics:')
    for key, value in summary.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()

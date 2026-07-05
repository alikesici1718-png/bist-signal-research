import os
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import logging

# Log dosyasını oluştur
log_file = 'logs/laggard_catchup_scan.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(data_dir):
    data = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv') and file_name != 'DSTKF.csv':
            symbol = os.path.basename(file_name).split('.')[0]
            df = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=['Date'])
            data[symbol] = df['Close']
    return pd.DataFrame(data)

def calculate_log_returns(df):
    log_returns = np.log(df / df.shift(1))
    return log_returns.dropna()

def calculate_cross_sectional_z_scores(log_returns, lookback):
    z_scores = {}
    for date in log_returns.index:
        subset = log_returns.loc[:date].tail(lookback)
        mean = subset.mean()
        std_dev = subset.std()
        z_scores[date] = (log_returns.loc[date] - mean) / std_dev
    return pd.Series(z_scores)

def calculate_forward_returns(log_returns, horizon):
    forward_returns = {}
    for date in log_returns.index:
        if date + timedelta(days=horizon) in log_returns.index:
            forward_returns[date] = log_returns.loc[date + timedelta(days=horizon)]
    return pd.Series(forward_returns)

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
    
    lookbacks = [5, 10, 20]
    horizons = [5, 10, 20]
    
    results = []
    
    for lookback in lookbacks:
        for horizon in horizons:
            log_returns = calculate_log_returns(df)
            z_scores = calculate_cross_sectional_z_scores(log_returns, lookback)
            
            # Laggard tanımı: en düşük %20'lik dilim (yaklaşık en düşük 6 hisse)
            laggards = z_scores.nsmallest(6).index
            
            forward_returns = calculate_forward_returns(log_returns, horizon)
            laggard_forward_returns = forward_returns.loc[laggards]
            
            # TÜM piyasanın aynı dönemdeki ortalama forward getirisinden yüksek mi?
            market_forward_return = forward_returns.mean()
            excess_return = (laggard_forward_returns - market_forward_return) * 100000
            
            t_stat, p_value = ttest_ind(laggard_forward_returns, market_forward_return)
            
            results.append({
                'lookback': lookback,
                'horizon': horizon,
                'excess_return_mean': excess_return.mean(),
                't_stat': t_stat,
                'p_value': p_value,
                'signal_count': len(laggard_forward_returns)
            })
    
    # Sonuçları DataFrame'e topla
    results_df = pd.DataFrame(results)
    
    # Non-overlapping test için: sinyaller horizon kadar aralıklarla örneklensin (overlap'ten kaynaklanan sahte güven riskini azaltmak için), hem overlapping hem non-overlapping sonucu ayrı ayrı raporla
    overlapping_results = results_df.copy()
    non_overlapping_results = results_df.iloc[::horizon].copy()
    
    # Veriyi ilk yarı (older) ve ikinci yarı (recent) olarak ikiye bölüp her iki dönemde de sonucu ayrı raporla (recency-first audit mantığı - sadece son dönemde mi çalışıyor yoksa tutarlı mı)
    older_results = results_df.iloc[:len(results_df)//2].copy()
    recent_results = results_df.iloc[len(results_df)//2:].copy()
    
    # Konsola özet tablo yazdır
    print('Laggard Catchup Results')
    print(results_df[['lookback', 'horizon', 'excess_return_mean', 't_stat', 'p_value', 'signal_count']].to_string(index=False))
    
    # Sonuçları kaydet
    results_df.to_csv('laggard_catchup_results.csv', index=False)
    overlapping_results.to_csv('laggard_catchup_overlapping_results.csv', index=False)
    non_overlapping_results.to_csv('laggard_catchup_non_overlapping_results.csv', index=False)
    older_results.to_csv('laggard_catchup_older_results.csv', index=False)
    recent_results.to_csv('laggard_catchup_recent_results.csv', index=False)

if __name__ == '__main__':
    main()

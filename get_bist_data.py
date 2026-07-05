import yfinance as yf
import pandas as pd
import os
import time
import logging

# Log dosyasını oluştur
log_file = 'logs/get_bist_data.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_symbols(file_path):
    with open(file_path, 'r') as file:
        symbols = [line.strip() for line in file if line.strip()]
    return symbols

def download_data(symbol):
    try:
        data = yf.download(symbol + '.IS', period='2y', group_by='ticker')
        
        # Eğer data.columns bir MultiIndex ise, düzleştir (flatten)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        
        # CSV'ye yazmadan önce index'i sıfırla (reset_index) ki Date bir kolon olsun
        data.reset_index(inplace=True)
        
        # Sadece belirtilen kolonları koruyarak diğerlerini kaldır
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[columns_to_keep]
        
        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{symbol}.csv', index=False)
        logging.info(f'Data downloaded and saved for {symbol}')
    except Exception as e:
        logging.error(f'Error downloading data for {symbol}: {e}')

def main():
    symbols_file = 'config/symbols.txt'
    symbols = read_symbols(symbols_file)
    
    if not symbols:
        logging.error('No symbols found in the config file')
        return
    
    for symbol in symbols:
        download_data(symbol)
        time.sleep(0.5)  # Rate limiting

if __name__ == '__main__':
    main()

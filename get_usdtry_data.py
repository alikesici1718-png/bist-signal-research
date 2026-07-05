import yfinance as yf
import pandas as pd
import os
import time
import logging

# Log file creation
log_file = 'logs/get_usdtry_data.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(symbol):
    try:
        data = yf.download(symbol, period='2y', auto_adjust=True)
        if data.empty:
            logging.warning(f'No data found for {symbol}')
            return
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.reset_index()
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[c for c in columns_to_keep if c in data.columns]]
        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{symbol}.csv', index=False)
        logging.info(f'Data downloaded and saved for {symbol}')
    except Exception as e:
        logging.error(f'Error downloading data for {symbol}: {e}')

def main():
    symbol = 'USDTRY=X'
    download_data(symbol)

if __name__ == '__main__':
    main()

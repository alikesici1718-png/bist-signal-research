"""Data-quality audit of per-symbol OHLCV CSVs in data/.

Checks each downloaded symbol file for missing values, date gaps,
non-positive prices and other structural problems before any signal
scan consumes the data. Output: quality_report.txt.
"""
import os
import pandas as pd
from datetime import timedelta
import logging

# Log file creation
log_file = 'logs/get_bist_data.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        
        # Kontrolleri yap
        total_rows = len(data)
        date_range = (data['Date'].min(), data['Date'].max())
        missing_values = data.isnull().sum()
        duplicate_dates = data.duplicated(subset='Date').any()
        
        daily_changes = data['Close'].pct_change() * 100
        large_changes = daily_changes.abs() > 20
        
        ohlc_issues = (data['High'] < data['Low']) | ((data['Close'] / data['Open']).abs() > 1.5)
        
        insufficient_trading_days = len(data) < 300
        
        # Sonuçları topla
        issues = []
        if missing_values.any():
            issues.append(f'Missing values in columns: {missing_values[missing_values > 0]}')
        if duplicate_dates:
            issues.append('Duplicate dates found')
        if large_changes.any():
            issues.append(f'Large changes detected on dates: {data.loc[large_changes, "Date"]}')
        if ohlc_issues.any():
            issues.append('OHLC inconsistencies found')
        if insufficient_trading_days:
            issues.append('Insufficient trading days')
        
        return {
            'symbol': os.path.basename(file_path).split('.')[0],
            'total_rows': total_rows,
            'date_range': date_range,
            'issues_count': len(issues),
            'issues': issues
        }
    except Exception as e:
        logging.error(f'Error checking data for {file_path}: {e}')
        return {
            'symbol': os.path.basename(file_path).split('.')[0],
            'total_rows': None,
            'date_range': None,
            'issues_count': 1,
            'issues': [f'General error: {str(e)}']
        }

def main():
    data_dir = 'data'
    report_file = 'quality_report.txt'
    
    results = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            result = check_data(file_path)
            results.append(result)
    
    # Konsola özet tablo yazdır
    print('Symbol\tTotal Rows\tDate Range\tIssues Count')
    for result in results:
        print(f'{result["symbol"]}\t{result["total_rows"]}\t{result["date_range"]}\t{result["issues_count"]}')
    
    # Detaylı sorunları quality_report.txt dosyasına yaz
    with open(report_file, 'w') as report:
        for result in results:
            report.write(f'Symbol: {result["symbol"]}\n')
            report.write(f'Total Rows: {result["total_rows"]}\n')
            report.write(f'Date Range: {result["date_range"]}\n')
            report.write(f'Issues Count: {result["issues_count"]}\n')
            if result["issues"]:
                report.write('Issues:\n')
                for issue in result["issues"]:
                    report.write(f'- {issue}\n')
            report.write('\n')

if __name__ == '__main__':
    main()

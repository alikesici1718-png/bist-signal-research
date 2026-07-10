import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

end_date = datetime.today()
start_date = end_date - timedelta(days=365)

ticker = yf.Ticker("USDTRY=X")
df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")

df = df[["Close"]].reset_index()
df.columns = ["date", "close_price"]
df["date"] = pd.to_datetime(df["date"]).dt.date

output_path = "data/carry_trade/usdtry_price.csv"
df.to_csv(output_path, index=False)

print(f"Satır sayısı: {len(df)}")
print(f"Tarih aralığı: {df['date'].min()} — {df['date'].max()}")
print(f"Kaydedildi: {output_path}")

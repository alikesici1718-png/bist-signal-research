"""Debug script: inspect kap_data/kap_financial_report_dates.csv
(row counts, date formats, year/period distributions) to validate the
KAP fetch output. No analysis output.
"""
import pandas as pd

df = pd.read_csv("kap_data/kap_financial_report_dates.csv")

print("=== Genel bilgi ===")
print(f"Toplam satir: {len(df)}")
print(f"Benzersiz sembol: {df['ticker'].nunique()}")
print(f"Sutunlar: {df.columns.tolist()}")
print()

print("=== İlk 10 satır ===")
print(df.head(10).to_string())
print()

print("=== publish_date formatı örnekleri ===")
print(df['publish_date'].head(5).tolist())
print()

print("=== year dağılımı ===")
print(df['year'].value_counts().sort_index())
print()

print("=== period dağılımı ===")
print(df['period'].value_counts())
print()

print("=== title örnekleri (ilk 10 benzersiz) ===")
print(df['title'].dropna().unique()[:10])
print()

print("=== Sembol başına ortalama bildirim sayısı ===")
print(df.groupby('ticker').size().describe())

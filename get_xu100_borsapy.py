"""
get_xu100_borsapy.py

BIST100 (XU100) endeksinin tam gecmisini borsapy ile ceker ve
data_market/ klasorune standart formatta (Date, Open, High, Low, Close, Volume)
kaydeder. Bu, comprehensive_scan.py / diagnose_signals.py'deki esit-agirlikli
market_ret proxy'sinin yerini alacak gercek piyasa benchmarki icin kullanilacak.

Kullanim:
    python get_xu100_borsapy.py
"""

import os
import pandas as pd
import borsapy as bp

OUTPUT_PATH = "data_market/XU100.csv"
os.makedirs("data_market", exist_ok=True)

index = bp.Index("XU100")
df = index.history(period="max")

if df is None or df.empty:
    print("HATA: XU100 icin veri gelmedi.")
else:
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"UYARI: eksik kolonlar: {missing}, mevcut kolonlar: {df.columns.tolist()}")
        # Volume endekste olmayabilir; varsa kullan, yoksa 0 ile doldur
        if "Volume" in missing:
            df["Volume"] = 0
            missing.remove("Volume")
    if not missing:
        df = df[needed].sort_values("Date").drop_duplicates(subset="Date", keep="last")
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Kaydedildi: {OUTPUT_PATH} ({len(df)} satir, "
              f"{df['Date'].min().date()} -> {df['Date'].max().date()})")
"""
test_borsapy.py - borsapy kurulumunu dogrular ve AKYHO'daki bedelli sermaye
artirimi (Agustos 2024) etrafinda adjusted vs raw fiyat farkini gosterir.
"""

import borsapy as bp
import pandas as pd

pd.set_option("display.max_rows", 30)
pd.set_option("display.width", 120)

symbol = "AKYHO"
ticker = bp.Ticker(symbol)

print(f"=== {symbol} - ADJUSTED (duzeltilmis) fiyat ===")
try:
    df_adj = ticker.history(start="2024-07-20", end="2024-08-20")
    print(df_adj[["Close", "Volume"]])
except Exception as e:
    print(f"HATA (adjusted): {e}")

print(f"\n=== {symbol} - RAW (ham) fiyat ===")
try:
    df_raw = ticker.history(start="2024-07-20", end="2024-08-20", adjust=False)
    print(df_raw[["Close", "Volume"]])
except Exception as e:
    print(f"HATA (raw): {e}")

print(f"\n=== {symbol} - Kurumsal islemler (splits/actions) ===")
try:
    print(ticker.splits)
except Exception as e:
    print(f"HATA (splits): {e}")

try:
    print(ticker.actions)
except Exception as e:
    print(f"HATA (actions): {e}")
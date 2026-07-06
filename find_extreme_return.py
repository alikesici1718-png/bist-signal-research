"""
find_extreme_return.py - 302437bps gibi imkansiz excess return'lerin
kaynagini bulur: hangi sembol, hangi tarih, ne kadar boşluk var.
"""
import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "data"
MARKET_PATH = "data_market/XU100.csv"
MIN_ROWS = 250

def load_symbol(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return None
    return df

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
symbols = {}
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    df = load_symbol(f)
    if df is not None:
        symbols[name] = df

all_dates = sorted(set().union(*[set(df["Date"]) for df in symbols.values()]))
price_cols = {name: df.set_index("Date")["Close"] for name, df in symbols.items()}
price_panel = pd.concat(price_cols, axis=1).reindex(all_dates)
ret_panel = price_panel.pct_change()

# En yuksek tekil gunluk getiriyi bul (bunlar excess'i sisiren olaylar)
flat = ret_panel.stack()
top = flat.sort_values(ascending=False).head(15)
print("=== EN YUKSEK 15 GUNLUK GETIRI (muhtemel veri hatasi) ===")
for (date, sym), val in top.items():
    print(f"{sym} {date.date()}: {val*100:.1f}%")

print()
print("=== Bu sembollerde veri bosluklarina bakalim ===")
for sym in set(s for d,s in top.index):
    df = symbols[sym]
    gaps = df["Date"].diff().dt.days
    big_gaps = gaps[gaps > 10]
    print(f"{sym}: {len(df)} satir, buyuk bosluk sayisi (>10 gun): {len(big_gaps)}")
    if len(big_gaps) > 0:
        for idx in big_gaps.index[:3]:
            print(f"    bosluk: {df.loc[idx-1,'Date'].date()} -> {df.loc[idx,'Date'].date()} ({int(gaps[idx])} gun)")
"""
find_bad_csv.py - data/ klasorundeki tum CSV'leri tarar, 'Date' kolonu
olmayan veya okunamayan dosyalari raporlar.
"""
import os
import pandas as pd

DATA_DIR = "data"

bad_files = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(DATA_DIR, fname)
    try:
        # Once sadece header'i oku, hizli olsun
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if "Date" not in header:
            bad_files.append((fname, header))
    except Exception as e:
        bad_files.append((fname, f"OKUMA HATASI: {e}"))

print(f"Toplam dosya sayisi taranan CSV: {len([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])}")
print(f"Sorunlu dosya sayisi: {len(bad_files)}")
for fname, info in bad_files:
    print(f"  {fname}: {info}")
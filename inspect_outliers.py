"""
inspect_outliers.py

Amac: diagnose_signals.py'nin urettigi outlier_events.csv'yi okuyup
manuel inceleme icin okunabilir bir ozet cikarir. Her outlier icin,
ilgili sembolun ham CSV'sinden o tarih civarindaki fiyat/hacim satirlarini
da gosterir -- boylece limit-down / split / temettu supheli olaylari
gozle tespit etmek kolaylasir.

Kullanim:
    python inspect_outliers.py
Cikti:
    Konsola tablo + her outlier icin +/-3 gunluk fiyat/hacim penceresi
    outlier_inspection.csv (detayli, Excel'de acilabilir)
"""

import os
import pandas as pd

DATA_DIR = "data"
OUTLIER_CSV = "outlier_events.csv"
OUTPUT_CSV = "outlier_inspection.csv"
WINDOW_DAYS = 3  # olay tarihinin etrafinda kac gun gosterilsin


def load_raw(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def window_around(df, event_date, days):
    df = df.set_index("Date")
    idx = df.index.get_indexer([event_date], method="nearest")[0]
    lo = max(0, idx - days)
    hi = min(len(df), idx + days + 1)
    return df.iloc[lo:hi].reset_index()


def main():
    if not os.path.exists(OUTLIER_CSV):
        print(f"HATA: {OUTLIER_CSV} bulunamadi. Once diagnose_signals.py calistir.")
        return

    outliers = pd.read_csv(OUTLIER_CSV, parse_dates=["date"])
    print(f"{len(outliers)} outlier olay okundu.\n")

    detail_rows = []

    for i, row in outliers.iterrows():
        symbol = row["symbol"]
        event_date = row["date"]
        excess = row["excess"]
        liquidity = row.get("liquidity", "?")

        print("=" * 70)
        print(f"#{i+1}  {symbol}  {event_date.date()}  excess={excess*10000:.1f}bps  liquidity={liquidity}")

        raw = load_raw(symbol)
        if raw is None:
            print(f"  UYARI: {symbol}.csv bulunamadi (belki silinmis/yeniden adlandirilmis).")
            continue

        win = window_around(raw, event_date, WINDOW_DAYS)
        if win.empty:
            print("  UYARI: pencere bos.")
            continue

        # gunluk % degisim ekle, gozle kontrol icin
        win["pct_change"] = win["Close"].pct_change() * 100

        display_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume", "pct_change"] if c in win.columns]
        print(win[display_cols].to_string(index=False))

        # basit heuristik bayraklar (kesin degil, sadece dikkat ceksin diye)
        flags = []
        if win["pct_change"].abs().max() > 15:
            flags.append("BUYUK_TEK_GUN_HAREKETI(>%15)")
        if "Volume" in win.columns and win["Volume"].min() == 0:
            flags.append("SIFIR_HACIM_GUNU_VAR")
        if flags:
            print(f"  BAYRAK: {', '.join(flags)}")

        for _, wr in win.iterrows():
            detail_rows.append({
                "outlier_rank": i + 1,
                "symbol": symbol,
                "event_date": event_date.date(),
                "excess_bps": excess * 10000,
                "liquidity": liquidity,
                "row_date": wr["Date"].date() if pd.notna(wr["Date"]) else None,
                "Open": wr.get("Open"),
                "High": wr.get("High"),
                "Low": wr.get("Low"),
                "Close": wr.get("Close"),
                "Volume": wr.get("Volume"),
                "pct_change": wr.get("pct_change"),
                "flags": ";".join(flags) if flags else "",
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 70)
    print(f"Detayli tablo kaydedildi: {OUTPUT_CSV}")
    print("Excel'de ac, 'flags' kolonuna gore filtrele -- BUYUK_TEK_GUN_HAREKETI"
          " veya SIFIR_HACIM_GUNU_VAR olanlari once incele.")


if __name__ == "__main__":
    main()
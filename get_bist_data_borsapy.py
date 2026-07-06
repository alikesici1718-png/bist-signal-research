"""
get_bist_data_borsapy.py

yfinance yerine borsapy kullanarak, config/symbols.txt icindeki tum
BIST sembolleri icin mumkun olan en eski tarihten bugune kadar
duzeltilmis (adjusted) OHLCV verisini indirir.

Cikti formati eskiyle ayni: data/<SYMBOL>.csv
Kolonlar: Date, Open, High, Low, Close, Volume

Kullanim:
    python get_bist_data_borsapy.py
"""

import os
import time
import logging
import pandas as pd
import borsapy as bp

SYMBOLS_FILE = "config/symbols.txt"
OUTPUT_DIR = "data"
LOG_FILE = "logs/get_bist_data_borsapy.log"

RATE_LIMIT_SLEEP = 0.3      # istekler arasi bekleme (saniye)
BACKOFF_SLEEP = 5.0         # art arda hata sonrasi ek bekleme
MAX_CONSECUTIVE_ERRORS = 3  # bu kadar art arda hatada backoff devreye girer

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def load_symbols(path):
    with open(path, "r", encoding="utf-8") as f:
        symbols = [line.strip() for line in f if line.strip()]
    return symbols


def fetch_symbol(symbol):
    """Tek bir sembol icin tum gecmisi ceker, standart kolon formatina cevirir."""
    ticker = bp.Ticker(symbol)
    df = ticker.history(period="max")  # varsayilan: adjusted (duzeltilmis)

    if df is None or df.empty:
        raise ValueError("Bos veri dondu")

    df = df.reset_index()

    # Tarih kolonunun adi 'Date' olmayabilir; standardize et
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    # Timezone bilgisini kaldir (naive datetime), sadece tarihi tut
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df = df[needed].sort_values("Date").drop_duplicates(subset="Date", keep="last")
    df = df.reset_index(drop=True)
    return df


def main():
    symbols = load_symbols(SYMBOLS_FILE)
    log.info(f"{len(symbols)} sembol yuklenecek (kaynak: borsapy, adjusted)")

    success, failed = [], []
    consecutive_errors = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_symbol(symbol)
            out_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            df.to_csv(out_path, index=False)
            success.append(symbol)
            consecutive_errors = 0
            log.info(f"[{i}/{len(symbols)}] {symbol}: OK ({len(df)} satir, "
                      f"{df['Date'].min().date()} -> {df['Date'].max().date()})")
        except Exception as e:
            failed.append((symbol, str(e)))
            consecutive_errors += 1
            log.warning(f"[{i}/{len(symbols)}] {symbol}: HATA - {e}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.warning(f"{MAX_CONSECUTIVE_ERRORS} art arda hata, "
                            f"{BACKOFF_SLEEP}s ek bekleme...")
                time.sleep(BACKOFF_SLEEP)
                consecutive_errors = 0

        time.sleep(RATE_LIMIT_SLEEP)

    log.info("=" * 50)
    log.info(f"TAMAMLANDI: {len(success)} basarili, {len(failed)} basarisiz")
    if failed:
        log.info("Basarisiz semboller:")
        for symbol, err in failed:
            log.info(f"  {symbol}: {err}")


if __name__ == "__main__":
    main()
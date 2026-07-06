"""
fetch_kap_financial_reports.py

BIST sembolleri icin KAP'tan (Kamuyu Aydinlatma Platformu) finansal rapor
bildirim tarihlerini ceker. pykap kutuphanesini kullanir.

Onceki versiyon hataliydi. Bu versiyonda:
  - dogru subject UUID (finansal rapor: 4028328c594bfdca01594c0af9aa0057)
  - dogru disclosure_type = "FR"
  - her istek arasinda rate-limit (KAP sunucusunu yormamak icin)
  - hata durumunda exponential backoff ile retry (varsayilan 3 deneme)
  - her 50 sembolde bir ara kayit (checkpoint) -> crash olursa bastan baslamaya gerek yok
  - kaldigi yerden devam edebilme (zaten islenmis sembolleri atlar)
  - basarisiz sembolleri ayri bir dosyada loglar

Kullanim:
    python fetch_kap_financial_reports.py
    python fetch_kap_financial_reports.py --symbols THYAO,AKBNK
    python fetch_kap_financial_reports.py --from-date 2018-01-01 --to-date 2026-07-06
    python fetch_kap_financial_reports.py --resume   (varsayilan davranis zaten resume'dur)
"""

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

try:
    import pykap
except ImportError:
    print("pykap kurulu degil. Once calistirin: pip install pykap --break-system-packages")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
FINANCIAL_REPORT_SUBJECT = "4028328c594bfdca01594c0af9aa0057"  # Finansal Rapor
DISCLOSURE_TYPE = "FR"

OUTPUT_DIR = Path("kap_data")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "kap_financial_report_dates.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "kap_fetch_checkpoint.json"
FAILED_LOG_PATH = OUTPUT_DIR / "kap_fetch_failed.csv"
LOG_PATH = OUTPUT_DIR / "kap_fetch.log"

CHECKPOINT_EVERY = 50          # kac sembolde bir ara kayit yapilsin
REQUEST_DELAY_SECONDS = 2.5    # her sembol istegi arasi bekleme (KAP'i yormamak icin)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5.0       # saniye; deneme basina 1x, 2x, 4x... katlanir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------
def load_checkpoint() -> set:
    """Daha once basariyla islenmis sembolleri dondurur."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        return set(data.get("done_symbols", []))
    return set()


def save_checkpoint(done_symbols: set):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"done_symbols": sorted(done_symbols), "updated_at": datetime.now().isoformat()}, f, indent=2)


def load_existing_results() -> pd.DataFrame:
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame(columns=["ticker", "publish_date", "year", "period", "title", "disclosure_index"])


def append_results(rows: list):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        df_new.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RESULTS_PATH, mode="w", header=True, index=False)


def log_failure(ticker: str, error: str):
    row = pd.DataFrame([{"ticker": ticker, "error": error, "timestamp": datetime.now().isoformat()}])
    if FAILED_LOG_PATH.exists():
        row.to_csv(FAILED_LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(FAILED_LOG_PATH, mode="w", header=True, index=False)


def fetch_one_symbol(ticker: str, from_date: date, to_date: date) -> list:
    """
    Tek bir sembol icin finansal rapor bildirim tarihlerini ceker.
    Tarih araligi 365 gunluk pencerelere bolunur; her pencere ayri istek.
    Basarisiz olursa MAX_RETRIES kadar exponential backoff ile tekrar dener.
    """
    from datetime import timedelta
    all_rows = []
    window_start = from_date
    while window_start < to_date:
        window_end = min(window_start + timedelta(days=365), to_date)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                comp = pykap.BISTCompany(ticker)
                disclosures = comp.get_historical_disclosure_list(
                    fromdate=window_start,
                    todate=window_end,
                    disclosure_type=DISCLOSURE_TYPE,
                    subject=FINANCIAL_REPORT_SUBJECT,
                )
                for d in disclosures:
                    all_rows.append({
                        "ticker": ticker,
                        "publish_date": d.get("publishDate") or d.get("publish_date"),
                        "year": d.get("year"),
                        "period": d.get("period"),
                        "title": d.get("title"),
                        "disclosure_index": d.get("disclosureIndex") or d.get("disclosure_index"),
                    })
                break
            except Exception as e:
                last_error = str(e)
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                log.warning(f"{ticker} [{window_start}-{window_end}]: deneme {attempt}/{MAX_RETRIES} basarisiz ({last_error}). {wait:.0f}sn bekleniyor...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"{ticker} [{window_start}-{window_end}] icin {MAX_RETRIES} deneme de basarisiz: {last_error}")

        time.sleep(REQUEST_DELAY_SECONDS)
        window_start = window_end

    return all_rows


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KAP finansal rapor bildirim tarihlerini cek")
    parser.add_argument("--symbols", type=str, default=None,
                         help="Virgulle ayrilmis sembol listesi (varsayilan: tum BIST sembolleri)")
    parser.add_argument("--from-date", type=str, default="2015-01-01")
    parser.add_argument("--to-date", type=str, default=date.today().isoformat())
    parser.add_argument("--no-resume", action="store_true",
                         help="Checkpoint'i yok say, tum sembolleri bastan cek")
    args = parser.parse_args()

    from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = sorted(pykap.bist_company_list())

    log.info(f"Toplam {len(symbols)} sembol islenecek ({from_date} -> {to_date})")

    done_symbols = set() if args.no_resume else load_checkpoint()
    if done_symbols:
        log.info(f"Checkpoint bulundu: {len(done_symbols)} sembol daha once islenmis, atlanacak")

    remaining = [s for s in symbols if s not in done_symbols]
    log.info(f"Islenecek kalan sembol sayisi: {len(remaining)}")

    processed_this_run = 0
    total_rows_this_run = 0
    failed_this_run = []

    for i, ticker in enumerate(remaining, start=1):
        try:
            rows = fetch_one_symbol(ticker, from_date, to_date)
            append_results(rows)
            total_rows_this_run += len(rows)
            done_symbols.add(ticker)
            processed_this_run += 1
            log.info(f"[{i}/{len(remaining)}] {ticker}: {len(rows)} bildirim bulundu")
        except Exception as e:
            log.error(f"[{i}/{len(remaining)}] {ticker}: KALICI HATA -> {e}")
            log_failure(ticker, str(e))
            failed_this_run.append(ticker)
            # basarisiz sembolu done_symbols'a EKLEMIYORUZ ki bir sonraki
            # calistirmada tekrar denensin

        # rate limit
        time.sleep(REQUEST_DELAY_SECONDS)

        # periyodik checkpoint
        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(done_symbols)
            log.info(f"--- Checkpoint kaydedildi: {len(done_symbols)}/{len(symbols)} sembol tamamlandi ---")

    # son checkpoint
    save_checkpoint(done_symbols)

    log.info("=" * 60)
    log.info(f"BITTI. Bu calistirmada islenen: {processed_this_run}, "
             f"toplam cekilen satir: {total_rows_this_run}, "
             f"basarisiz: {len(failed_this_run)}")
    if failed_this_run:
        log.info(f"Basarisiz semboller: {failed_this_run}")
        log.info(f"Detaylar icin: {FAILED_LOG_PATH}")
    log.info(f"Sonuclar: {RESULTS_PATH}")
    log.info("Tekrar calistirirsan (--no-resume vermeden) kaldigi yerden / "
             "basarisiz sembollerden devam eder.")


if __name__ == "__main__":
    main()
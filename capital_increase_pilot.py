"""
capital_increase_pilot.py

Pilot: 25 likit BIST sembolü için KAP'tan Sermaye Artırımı/Azaltımı bildirimlerini çeker,
BEDELSIZ / BEDELLI / TAHSISLI / BELIRSIZ olarak sınıflandırır ve raporlar.

Tarih aralığı: 2023-01-01 → 2026-07-06 (1 yıllık pencereler)
Çıktı: kap_data/pilot_capital_increase.csv
"""

import re
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    import pykap
except ImportError:
    print("pykap kurulu degil. Once calistirin: pip install pykap")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
TICKERS = [
    "THYAO", "AKBNK", "GARAN", "ISCTR", "YKBNK",
    "SAHOL", "KCHOL", "SISE",  "EREGL", "BIMAS",
    "TUPRS", "ASELS", "FROTO", "TOASO", "TCELL",
    "PGSUS", "ARCLK", "VAKBN", "HALKB", "PETKM",
    "KOZAL", "ENKAI", "MGROS", "ULKER", "TAVHL",
]

CAPITAL_INCREASE_SUBJECT = "4028328d5988e2630159d5fd68661ff4"  # Sermaye Artırımı/Azaltımı

FROM_DATE = date(2023, 1, 1)
TO_DATE   = date(2026, 7, 6)

OUTPUT_DIR  = Path("kap_data")
OUTPUT_PATH = OUTPUT_DIR / "pilot_capital_increase.csv"

REQUEST_DELAY_SECONDS = 8.0
MAX_RETRIES           = 3
RETRY_BACKOFF_BASE    = 15.0

KAP_API_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://www.kap.org.tr/",
    "Origin":  "https://www.kap.org.tr",
}


def build_company_id_map(tickers: list[str]) -> dict[str, str]:
    """Her ticker için pykap üzerinden gerçek KAP company_id'sini çeker."""
    id_map = {}
    for ticker in tickers:
        try:
            comp = pykap.BISTCompany(ticker)
            id_map[ticker] = comp.company_id
        except Exception as e:
            print(f"  [UYARI] {ticker} için company_id alınamadı: {e}")
    return id_map


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
def normalize_tr(text: str) -> str:
    """Türkçe karakterleri ASCII'ye düşürüp küçük harfe çevirir."""
    if not text:
        return ""
    # Türkçe özgün dönüşümler
    tr_map = str.maketrans("ığüşöçİĞÜŞÖÇ", "igusocIGUSOC")
    text = text.translate(tr_map)
    # Kalan unicode aksan vs. düşür
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def classify(title: str) -> str:
    n = normalize_tr(title)
    if "bedelsiz" in n:
        return "BEDELSIZ"
    if "bedelli" in n:
        return "BEDELLI"
    if "tahsisli" in n:
        return "TAHSISLI"
    return "BELIRSIZ"


def fetch_window(company_id: str, ticker: str, window_start: date, window_end: date) -> list[dict]:
    """Tek sembol + tek pencere için KAP byCriteria POST isteği atar."""
    payload = {
        "fromDate":                window_start.isoformat(),
        "toDate":                  window_end.isoformat(),
        "subjectList":             [CAPITAL_INCREASE_SUBJECT],
        "mkkMemberOidList":        [company_id],
        "inactiveMkkMemberOidList": [],
        "bdkMemberOidList":        [],
        "fromSrc":                 False,
        "disclosureIndexList":     [],
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(KAP_API_URL, json=payload, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ("data", "result", "items", "disclosures"):
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    return []
                return []
            elif resp.status_code == 429:
                wait = 60.0
                print(f"  [429] Rate limit. {wait:.0f}sn bekleniyor...")
                time.sleep(wait)
                last_err = "HTTP 429"
                continue
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)

        wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
        print(f"  [{ticker}] Deneme {attempt}/{MAX_RETRIES} basarisiz ({last_err}). {wait:.0f}sn bekleniyor...")
        time.sleep(wait)

    raise RuntimeError(f"{ticker} [{window_start}->{window_end}]: {MAX_RETRIES} deneme de basarisiz. Son hata: {last_err}")


def fetch_ticker(ticker: str, company_id: str) -> list[dict]:
    """Tüm 1 yıllık pencereler için verileri toplar."""
    rows = []
    window_start = FROM_DATE
    while window_start < TO_DATE:
        window_end = min(window_start + timedelta(days=365), TO_DATE)
        label = f"{window_start} -> {window_end}"
        print(f"  Pencere: {label}")

        try:
            records = fetch_window(company_id, ticker, window_start, window_end)
        except RuntimeError as e:
            print(f"  HATA: {e}")
            window_start = window_end
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        for rec in records:
            # Alan adları API'ye göre değişebilir; en yaygın isimler deneniyor
            title = (
                rec.get("kapTitle")
                or rec.get("title")
                or rec.get("subject")
                or rec.get("summary")
                or ""
            )
            publish_date = (
                rec.get("publishDate")
                or rec.get("publish_date")
                or rec.get("publishedAt")
                or ""
            )
            disc_index = (
                rec.get("disclosureIndex")
                or rec.get("disclosure_index")
                or rec.get("id")
                or ""
            )
            rows.append({
                "ticker":           ticker,
                "publish_date":     publish_date,
                "disclosure_index": disc_index,
                "kategori":         classify(title),
                "ham_title":        title,
            })

        print(f"    → {len(records)} kayıt bulundu")
        time.sleep(REQUEST_DELAY_SECONDS)
        window_start = window_end

    return rows


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    errors   = []

    print(f"=== Sermaye Artirimi Pilot ===")
    print(f"Semboller: {len(TICKERS)} adet")
    print(f"Tarih araligi: {FROM_DATE} -> {TO_DATE}")
    print(f"Cikti: {OUTPUT_PATH}\n")

    print("Company ID haritasi olusturuluyor...")
    id_map = build_company_id_map(TICKERS)
    print(f"{len(id_map)}/{len(TICKERS)} sembol icin company_id bulundu.\n")

    for i, ticker in enumerate(TICKERS, 1):
        company_id = id_map.get(ticker)
        if not company_id:
            print(f"[{i:2d}/{len(TICKERS)}] {ticker} ATLANDI (company_id bulunamadi)\n")
            errors.append(f"{ticker}: company_id bulunamadi")
            continue
        print(f"[{i:2d}/{len(TICKERS)}] {ticker} isleniyor... (id: {company_id})")
        try:
            rows = fetch_ticker(ticker, company_id)
            all_rows.extend(rows)
            print(f"  Toplam {len(rows)} kayıt eklendi.\n")
        except Exception as e:
            msg = f"{ticker}: {e}"
            errors.append(msg)
            print(f"  !! Sembol atlandı: {msg}\n")

    # CSV kaydet
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print(f"\nCSV kaydedildi: {OUTPUT_PATH} ({len(df)} satır)")
    else:
        df = pd.DataFrame(columns=["ticker", "publish_date", "disclosure_index", "kategori", "ham_title"])
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        print("\nHiç kayıt bulunamadı, boş CSV kaydedildi.")

    # ---------------------------------------------------------------------------
    # Rapor
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RAPOR")
    print("=" * 60)
    print(f"Toplam çekilen kayıt: {len(all_rows)}")

    if all_rows:
        df_report = pd.DataFrame(all_rows)

        cat_counts = df_report["kategori"].value_counts()
        print("\nKategori dağılımı:")
        for cat in ["BEDELSIZ", "BEDELLI", "TAHSISLI", "BELIRSIZ"]:
            print(f"  {cat:12s}: {cat_counts.get(cat, 0)}")

        print()
        for cat in ["BEDELSIZ", "BEDELLI", "TAHSISLI", "BELIRSIZ"]:
            subset = df_report[df_report["kategori"] == cat]
            if subset.empty:
                print(f"--- {cat}: 0 kayıt ---\n")
                continue
            samples = subset["ham_title"].dropna().unique()[:3]
            print(f"--- {cat} ({len(subset)} kayıt) — 3 örnek başlık ---")
            for j, t in enumerate(samples, 1):
                print(f"  {j}. {t}")
            print()

    if errors:
        print("Hatalar:")
        for e in errors:
            print(f"  - {e}")

    print("=" * 60)
    print("Tamamlandı.")


if __name__ == "__main__":
    main()

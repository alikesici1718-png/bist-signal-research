# ON-TAAHHUT: Yonetici/buyuk ortak (icerideki) pay ALIM bildirimi sonrasi
# 20-60 gunde fiyat pozitif yonde hareket etmesi bekleniyor (akademik literatur:
# Sazak/Aydin BIST insider trading calismasi, 2015-2020, icerdekiler piyasayi
# etkin zamanliyor). SATIM bildirimi icin tersi ya da notr beklenir.
# Bu tek seferlik bir pilottur.

import time
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# KONFIGURASYON
# ---------------------------------------------------------------------------
SYMBOLS = [
    "ADESE", "AGHOL", "ALARK", "ALKIM", "ANELE",
    "ATAGY", "BAGFS", "BRISA", "BRYAT", "BUCIM",
    "CCOLA", "CEMTS", "DOAS",  "ECILC", "EGEEN",
    "GOODY", "GUBRF", "HEKTS", "KARSN", "KLMSN",
    "KONYA", "KORDS", "MAVI",  "OTKAR", "PARSN",
]

FROM_DATE   = date(2024, 1, 1)
TO_DATE     = date(2026, 7, 8)
WINDOW_DAYS = 180
RATE_LIMIT  = 8.0  # saniye, pencereler arasi

PAY_ALIM_SATIM_UUID = "8aca490d50286f620150287614ae005c"
API_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"

OUTPUT_DIR  = Path("kap_data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pilot_insider_trading_v2.csv"

# ---------------------------------------------------------------------------
# SESSION
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.kap.org.tr",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
    })
    s.get("https://www.kap.org.tr/tr/bildirim-sorgu", timeout=15)
    return s


# ---------------------------------------------------------------------------
# VERİ ÇEKME
# ---------------------------------------------------------------------------
def fetch_window(session: requests.Session, from_d: date, to_d: date) -> list:
    """Tek bir 180-gunluk pencere icin TUM Pay Alim Satim bildirimlerini ceker."""
    payload = {
        "fromDate": str(from_d),
        "toDate":   str(to_d),
        "disclosureClass": "ODA",
        "subjectList": [PAY_ALIM_SATIM_UUID],
        "mkkMemberOidList": [],
        "inactiveMkkMemberOidList": [],
        "bdkMemberOidList": [],
        "fromSrc": False,
        "disclosureIndexList": [],
    }
    r = session.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        raise RuntimeError("API hata dondurdu: " + str(data))
    return data


def fetch_all(session: requests.Session) -> list:
    """Tum tarih araligini 180-gunluk pencerelere bolup ceker."""
    all_records = []
    cur = FROM_DATE
    window_no = 0
    while cur < TO_DATE:
        end = min(cur + timedelta(days=WINDOW_DAYS), TO_DATE)
        window_no += 1
        print(f"  Pencere {window_no}: {cur} -> {end} ...", end=" ", flush=True)
        try:
            records = fetch_window(session, cur, end)
            print(f"{len(records)} kayit")
            all_records.extend(records)
        except Exception as e:
            print(f"HATA: {e}")
        cur = end + timedelta(days=1)
        if cur < TO_DATE:
            time.sleep(RATE_LIMIT)
    return all_records


# ---------------------------------------------------------------------------
# FİLTRELEME
# ---------------------------------------------------------------------------
def filter_symbols(records: list) -> list:
    """stockCodes alaninda SYMBOLS listesindeki sembolleri ara."""
    symbol_set = set(SYMBOLS)
    filtered = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        stock_codes = str(rec.get("stockCodes") or "")
        # stockCodes bazen virgülle ayrilmis olabilir ("THYAO, AKBNK")
        codes = {c.strip() for c in stock_codes.split(",")}
        match = codes & symbol_set
        if match:
            for sym in match:
                filtered.append({**rec, "_matched_ticker": sym})
    return filtered


# ---------------------------------------------------------------------------
# SINIFLANDIRMA
# ---------------------------------------------------------------------------
def classify(rec: dict) -> str:
    """
    Ham baslikta ALIM mi SATIM mi oldugunu belirle.
    Oncelik sirasi: summary > subject > kapTitle

    Kural:
    - "Pay Alım Satım Bildirimi" = konu basliginin kendisi, icerik bilgisi yok → BELIRSIZ
    - "satın alım", "piyasadan alım", "pay alım işlemleri" → ALIM
    - "satım", "satış" icerenler (ama "satın alım" değil) → SATIM
    - Sadece "alım" veya "alım işlemleri" → ALIM
    """
    # summary = gercek bildiri basligi (ISCTR: "Bankamizin...satin alinmasi...")
    # kapTitle = sirket adi (siniflama icin kullanisli degil)
    title = str(rec.get("summary") or rec.get("kapTitle") or rec.get("subject") or "").lower()

    # Normalize: Türkçe karakterleri ASCII'ye cevir (karsilastirma icin)
    tr_map = str.maketrans("çğıöşü", "cgiosu")
    t = title.translate(tr_map)

    # Hem "alim" hem "satim" iceren generik konu basliklarini BELIRSIZ say
    # Ornek: "pay alim satim bildirimi" (konu basligi, icerik degil)
    if "alim satim" in t or "alim-satim" in t:
        return "BELIRSIZ"

    # ALIM kaliplari: "satin alim", "piyasadan alim", "pay alim islemleri",
    # "hisse alim", "pay alimi", "alimina iliskin" vs.
    alim_patterns = ["satin al", "piyasadan al", "alim islemi", "alimi hakkinda",
                     "alima iliskin", "alinmasi hakkinda", "alinmasi hakk",
                     "pay alim", "hisse alim", "pay satin"]
    for pat in alim_patterns:
        if pat in t:
            return "ALIM"

    # SATIM kaliplari
    satim_patterns = ["satim islemi", "satimi hakkinda", "satima iliskin",
                      "satilmasi hakkinda", "pay satim", "hisse satim",
                      "satis islemi", "satis hakkinda", "satisina iliskin",
                      "elden cikarma"]
    for pat in satim_patterns:
        if pat in t:
            return "SATIM"

    # Kalan: sadece "alim" veya sadece "satim" geciyor mu?
    has_alim  = "alim" in t or "alimi" in t
    has_satim = "satim" in t or "satisi" in t or "satis" in t

    if has_alim and not has_satim:
        return "ALIM"
    if has_satim and not has_alim:
        return "SATIM"
    return "BELIRSIZ"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("KAP Insider Trading Pilot — Pay Alim Satim Bildirimleri")
    print(f"Semboller : {len(SYMBOLS)} adet")
    print(f"Pencere   : {FROM_DATE} -> {TO_DATE} ({WINDOW_DAYS}-gunluk alt-pencereler)")
    print(f"Konu UUID : {PAY_ALIM_SATIM_UUID}")
    print("=" * 65)

    session = make_session()

    print("\n[1] Veri cekiliyor (tum Pay Alim Satim bildirimleri)...")
    all_records = fetch_all(session)
    print(f"\nToplam ham kayit (tum semboller): {len(all_records)}")

    print("\n[2] 25 sembol icin filtreleniyor...")
    matched = filter_symbols(all_records)
    print(f"Eslesen kayit sayisi: {len(matched)}")

    if not matched:
        print("\nUYARI: Hic kayit bulunamadi. Semboller veya tarih araligi kontrol edilmeli.")
        return

    print("\n[3] ALIM/SATIM/BELIRSIZ siniflandirmasi yapiliyor...")
    rows = []
    for rec in matched:
        kategori = classify(rec)
        rows.append({
            "ticker":           rec["_matched_ticker"],
            "publish_date":     rec.get("publishDate", ""),
            "disclosure_index": rec.get("disclosureIndex", ""),
            "kategori":         kategori,
            "ham_title":        str(rec.get("summary") or rec.get("kapTitle") or rec.get("subject") or ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "publish_date"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"CSV kaydedildi: {OUTPUT_PATH}  ({len(df)} satir)")

    # ---------------------------------------------------------------------------
    # RAPOR
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RAPOR")
    print("=" * 65)
    print(f"Toplam kayit (25 sembol): {len(df)}")
    print()

    dist = df["kategori"].value_counts()
    print("Kategori dagilimi:")
    for kat, cnt in dist.items():
        print(f"  {kat:<12} : {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    print()
    for kat in ["ALIM", "SATIM", "BELIRSIZ"]:
        subset = df[df["kategori"] == kat]["ham_title"].unique()
        ornekler = subset[:3]
        print(f"[{kat}] ornek basliklar:")
        if len(ornekler) == 0:
            print("  (bu kategoride kayit yok)")
        for t in ornekler:
            print(f"  - {t[:100]}")
        print()

    print("Sembol bazinda kayit sayisi:")
    sym_dist = df.groupby("ticker")["kategori"].value_counts().unstack(fill_value=0)
    for col in ["ALIM", "SATIM", "BELIRSIZ"]:
        if col not in sym_dist.columns:
            sym_dist[col] = 0
    sym_dist = sym_dist[["ALIM", "SATIM", "BELIRSIZ"]]
    sym_dist["TOPLAM"] = sym_dist.sum(axis=1)
    sym_dist = sym_dist.sort_values("TOPLAM", ascending=False)
    print(sym_dist.to_string())

    print("\n" + "=" * 65)
    print("Pilot tamamlandi.")
    print("=" * 65)


if __name__ == "__main__":
    main()

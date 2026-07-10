# ON-TAAHHUT: BIST 30/50/100 endeksine DAHIL edilen hisselerin, efektif
# tarihten once (pasif fon rebalancing beklentisiyle) XU100'e gore POZITIF
# excess return gostermesi bekleniyor. CIKARILAN hisseler icin NEGATIF
# bekleniyor. Etkinin efektif tarihten onceki birkac gunde yogunlasmasi
# bekleniyor. Bu tek seferlik bir pilottur.

"""
index_inclusion_pilot.py — ADIM 1-3: PDF toplama ve parse.
Hedef endeksler: BIST 30, BIST 100
Cikti: index_changes.csv
"""

import re
import io
import glob
import os
import numpy as np
import requests
import pandas as pd
import pdfplumber
from scipy import stats

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20
TARGET_INDICES = {"BIST 30", "BIST 100"}

# -----------------------------------------------------------------------
# URL KAYNAKLARI
# Kaynak 1: fintables.com pattern — 2025_2_Donemsel_Degisiklikler.pdf
# Kaynak 2: KAP bildirim ekleri (kap.org.tr API)
# Kaynak 3: borsaistanbul.com doğrudan linkleri (Q4 2025 duyurusu)
# -----------------------------------------------------------------------

# Fintables URL pattern'i (2023 Q1 - 2026 Q2)
FINTABLES_BASE = "https://storage.fintables.com/media/uploads/kap-attachments/{year}_{q}_Donemsel_Degisiklikler.pdf"
FINTABLES_CANDIDATES = [
    (2022, 2), (2022, 3), (2022, 4),
    (2023, 1), (2023, 2), (2023, 3), (2023, 4),
    (2024, 1), (2024, 2), (2024, 3), (2024, 4),
    (2025, 1), (2025, 2), (2025, 3), (2025, 4),
    (2026, 1), (2026, 2), (2026, 3),
]

# KAP bildirim numaralari (pay endeksleri donemsel degisiklik bildirimleri)
KAP_BILDIRIMLERI = {
    "Q4-2025": "1621908",
    "Q1-2026": "1528220",
    "Q3-2025": "1299052",   # aslinda Q3-2024 olabilir, kontrol edilecek
    "Q3-2024": "1299052",
    "Q4-2024": "1336462",
    "Q1-2025": "1367507",
    "Q2-2025": "1409989",
}

def try_download(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200 and len(r.content) > 5000:
            ct = r.headers.get("content-type", "")
            # PDF magic veya content-type kontrolu (bazi sunucular Java serialize ediyor)
            if "pdf" in ct.lower() or r.content[:4] == b'%PDF' or len(r.content) > 10000:
                return r.content
    except Exception:
        pass
    return None

def kap_pdf_url(bildirim_id):
    """KAP bildirim sayfasindan PDF ek linkini bul."""
    api_url = f"https://www.kap.org.tr/tr/api/disclosureDetail/{bildirim_id}"
    try:
        r = requests.get(api_url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        # Ekleri ara
        attachments = data.get("attachmentList", []) or data.get("ekler", [])
        for att in attachments:
            name = att.get("fileName", "") or att.get("ad", "")
            url = att.get("url", "") or att.get("dosyaUrl", "")
            if url and ("degisik" in name.lower() or "donem" in name.lower() or ".pdf" in name.lower()):
                if not url.startswith("http"):
                    url = "https://www.kap.org.tr" + url
                return url
        # Tum ekleri dene
        for att in attachments:
            url = att.get("url", "") or att.get("dosyaUrl", "")
            if url and ".pdf" in url.lower():
                if not url.startswith("http"):
                    url = "https://www.kap.org.tr" + url
                return url
    except Exception:
        pass
    return None

SKIP_WORDS = {
    "ALINACAK", "CIKARILACAK", "YEDEK", "PAYLAR", "ENDEKSI", "ENDEKSLERI",
    "BIST", "THE", "FOR", "AND", "INDEX", "LIKIT", "BANKA", "SURDURULEBILIRLIK",
    "PAY", "KATILIM", "TEKNOLOJI", "HOLDING", "SANAYI", "ENERJI", "SAVUNMA",
    "SPORIF", "SPORTIF", "TRAKTOR", "GMYO", "URETIM", "MERKEZI",
}

def parse_pdf(pdf_bytes, source_label):
    """
    PDF'ten BIST 30 / BIST 100 giris/cikis verisi cikar.

    Format: Her satirda 3 sutun yan yana:
      'N TICKER_IN  NAME_IN  N TICKER_OUT  NAME_OUT  N TICKER_RES  NAME_RES'
    Her satirdaki N+TICKER esleri sirasiyla IN, OUT, RESERVE'e karsilik gelir.
    Sadece rakamla baslayan satirlar veri satiridir; devam satirlari atlanir.
    """
    results = []
    errors = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            lines = []
            for p in pdf.pages:
                txt = p.extract_text() or ""
                lines.extend(txt.split("\n"))
    except Exception as e:
        return [], [f"{source_label}: PDF acilamadi ({e})"]

    # Efektif tarih
    full_text = "\n".join(lines)
    date_match = re.search(r'(\d{2}[.\-/]\d{2}[.\-/]\d{4})', full_text)
    if date_match:
        try:
            effective_date = pd.to_datetime(date_match.group(1), dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            effective_date = date_match.group(1)
    else:
        effective_date = "UNKNOWN"
        errors.append(f"{source_label}: efektif tarih bulunamadi")

    # Endeks bolgesi takip et
    current_index = None
    in_data_block = False  # "ALINACAK PAYLAR..." basligi goruldu mu
    col_map = ["IN", "OUT", "RESERVE"]  # N+TICKER'in satirdaki sira pozisyonu

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Endeks basligi: "BIST 30 ENDEKSI" vb.
        m_idx = re.match(r'^(BIST\s+(\d+))\s+ENDEKS', line, re.IGNORECASE)
        if m_idx:
            num = m_idx.group(2)
            current_index = f"BIST {num}"
            in_data_block = False
            continue

        # Sutun basligi: "ALINACAK PAYLAR CIKARILACAK PAYLAR YEDEK PAYLAR"
        if re.search(r'ALINACAK', line, re.IGNORECASE) and current_index in TARGET_INDICES:
            in_data_block = True
            continue

        if not in_data_block or current_index not in TARGET_INDICES:
            continue

        # Devam satiri: rakamla baslamayan satir => atla
        if not re.match(r'^\d', line):
            continue

        # Her satirdaki tum N+TICKER esleri bul (siraya gore IN/OUT/RESERVE)
        # Pattern: bir veya iki rakam, bosluk, 3-6 buyuk harf ticker
        hits = re.findall(r'\b\d{1,2}\s+([A-Z]{3,6})\b', line)
        for pos, ticker in enumerate(hits):
            if ticker in SKIP_WORDS:
                continue
            if pos >= len(col_map):
                break
            change_type = col_map[pos]
            if change_type == "RESERVE":
                continue  # Yedekleri kaydetme
            results.append({
                "ticker": ticker,
                "index_name": current_index,
                "change_type": change_type,
                "effective_date": effective_date,
                "source": source_label,
            })

    if not any(r["change_type"] == "OUT" for r in results):
        errors.append(f"{source_label}: OUT kaydi bulunamadi (tarih={effective_date})")

    return results, errors


# -----------------------------------------------------------------------
# ANA AKIS
# -----------------------------------------------------------------------
all_records = []
all_errors = []
processed_pdfs = 0
skipped = []

print("PDF toplaniyor...\n")

# 1. Fintables pattern
seen_dates = set()
for year, q in FINTABLES_CANDIDATES:
    url = FINTABLES_BASE.format(year=year, q=q)
    label = f"fintables-{year}Q{q}"
    data = try_download(url)
    if data:
        print(f"  OK  {label} ({len(data)//1024} KB)")
        records, errs = parse_pdf(data, label)
        # Ayni efektif tarihi iki kaynaktan alma
        if records:
            ed = records[0]["effective_date"]
            if ed in seen_dates:
                print(f"       -> Atlandi (zaten var: {ed})")
                continue
            seen_dates.add(ed)
        all_records.extend(records)
        all_errors.extend(errs)
        processed_pdfs += 1
    else:
        skipped.append(label)

# 2. KAP bildirimleri — pykap ile ek bul (fintables'tan alınamayanlar icin)
# pykap API dogrudan desteklemediginden, bu blok su an pasif
# (KAP JSON API 404 donuyor, kap.org.tr JS-rendered)

# -----------------------------------------------------------------------
# KAYDET
# -----------------------------------------------------------------------
df = pd.DataFrame(all_records)
if not df.empty:
    df = df[df["change_type"].isin(["IN", "OUT"])].drop_duplicates(
        subset=["ticker", "index_name", "change_type", "effective_date"]
    ).sort_values(["effective_date", "index_name", "change_type", "ticker"])
    df.to_csv("index_changes.csv", index=False, encoding="utf-8")

# -----------------------------------------------------------------------
# RAPOR
# -----------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"OZET")
print(f"{'='*60}")
print(f"Islenen PDF sayisi   : {processed_pdfs}")
print(f"Atlanan URL sayisi   : {len(skipped)}")
if all_errors:
    print(f"\nParse hatalari ({len(all_errors)}):")
    for e in all_errors:
        print(f"  - {e}")

if not df.empty:
    print(f"\nToplam olay (IN+OUT, tekillestirilmis):")
    for idx in sorted(TARGET_INDICES):
        sub = df[df["index_name"] == idx]
        ins  = (sub["change_type"] == "IN").sum()
        outs = (sub["change_type"] == "OUT").sum()
        print(f"  {idx:10s}: {ins:3d} giris, {outs:3d} cikis")

    print(f"\nKapsanan donemler:")
    for d in sorted(df["effective_date"].unique()):
        src = df[df["effective_date"] == d]["source"].iloc[0]
        n = len(df[df["effective_date"] == d])
        print(f"  {d}  ({n} olay)  [{src}]")

    print(f"\nindex_changes.csv — ilk 15 satir:")
    print(df.head(15).to_string(index=False))
else:
    print("\nHic kayit toplanamadi.")

# =======================================================================
# BACKTEST
# =======================================================================
print(f"\n{'='*60}")
print("BACKTEST — BIST 100, efektif tarih oncesi 10 gun penceresi")
print(f"{'='*60}")

WINDOW_PRE  = 10   # efektif tarihten onceki islem gunu sayisi
WINDOW_POST =  5   # sonraki (bilgi icin, test disinda)
COST_LOW    = 20.0   # bps, Midas+DUSUK round-trip
COST_HIGH   = 150.0  # bps, AtaYatirim+YUKSEK round-trip
PLACEBO_SEED = 42
DATA_DIR    = "data"
XU100_PATH  = "data_market/XU100.csv"
EXCLUDE_SYM = {"USDTRY", "USDTRY=X", "ISKUR"}

# --- Fiyat paneli yukle ---
xu100_df = pd.read_csv(XU100_PATH, parse_dates=["Date"])
xu100_close = xu100_df.sort_values("Date").set_index("Date")["Close"]
xu100_ret = xu100_close.pct_change()

price_cache = {}
def get_close(sym):
    if sym in price_cache:
        return price_cache[sym]
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_csv(path, parse_dates=["Date"])
        s = d.sort_values("Date").set_index("Date")["Close"]
        s = s[s.notna()]
        price_cache[sym] = s
        return s
    except Exception:
        return None

def cum_excess(sym, anchor_date, pre, post):
    """
    anchor_date: efektif tarih (pd.Timestamp).
    Donus: (pre_cum_bps, post_cum_bps) — anchor oncesi/sonrasi kumülatif excess.
    """
    s = get_close(sym)
    if s is None:
        return np.nan, np.nan
    # XU100 ile ortak islem gunleri
    common = s.index.intersection(xu100_ret.index)
    if len(common) < pre + post + 2:
        return np.nan, np.nan
    # anchor_date'e en yakin gecmis islem gununu bul
    before = common[common < anchor_date]
    if len(before) < pre:
        return np.nan, np.nan
    after = common[common >= anchor_date]
    if len(after) < 1:
        return np.nan, np.nan

    # Pre pencere: anchor'dan ONCEKI pre gun
    pre_days  = before[-pre:]
    post_days = after[:post]

    stock_pre  = s.pct_change().reindex(pre_days)
    xu_pre     = xu100_ret.reindex(pre_days)
    exc_pre    = (stock_pre - xu_pre).dropna()
    if len(exc_pre) < pre // 2:
        return np.nan, np.nan
    pre_cum    = (1 + exc_pre).prod() - 1

    stock_post = s.pct_change().reindex(post_days)
    xu_post    = xu100_ret.reindex(post_days)
    exc_post   = (stock_post - xu_post).dropna()
    post_cum   = (1 + exc_post).prod() - 1 if len(exc_post) >= 1 else np.nan

    return pre_cum * 10000, post_cum * 10000  # bps

# --- Olaylari isle: sadece BIST 100 ---
bt_df = df[df["index_name"] == "BIST 100"].copy()
bt_df["effective_date"] = pd.to_datetime(bt_df["effective_date"])
bt_df = bt_df[~bt_df["ticker"].isin(EXCLUDE_SYM)].reset_index(drop=True)

rows = []
for _, row in bt_df.iterrows():
    pre_bps, post_bps = cum_excess(row["ticker"], row["effective_date"],
                                    WINDOW_PRE, WINDOW_POST)
    rows.append({
        "ticker": row["ticker"],
        "change_type": row["change_type"],
        "effective_date": row["effective_date"],
        "pre_bps": pre_bps,
        "post_bps": post_bps,
    })

bt = pd.DataFrame(rows).dropna(subset=["pre_bps"])
print(f"Kullanilabilir olay: {len(bt)} / {len(bt_df)} (data bulunan)")

def bh_fdr(p_values):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    for i in range(n-2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i+1]])
    return q.clip(max=1.0)

def group_stats(series, label):
    s = series.dropna()
    t, p = stats.ttest_1samp(s, 0)
    return {"Grup": label, "n": len(s), "Ort_bps": s.mean(), "t_stat": t, "p_value": p}

in_pre  = bt[bt["change_type"] == "IN"]["pre_bps"]
out_pre = bt[bt["change_type"] == "OUT"]["pre_bps"]

# Tablo 1: IN ve OUT vs 0
t1_rows = [group_stats(in_pre, "IN (giris)"), group_stats(out_pre, "OUT (cikis)")]
t1 = pd.DataFrame(t1_rows)
t1["q_value_BH"] = bh_fdr(t1["p_value"].values)
t1["Net_LOW_bps"]  = t1["Ort_bps"] - COST_LOW
t1["Net_HIGH_bps"] = t1["Ort_bps"] - COST_HIGH

# Tablo 2: IN vs OUT
t2_t, t2_p = stats.ttest_ind(in_pre.dropna(), out_pre.dropna(), equal_var=False)
t2 = pd.DataFrame([{"Karsilastirma": "IN vs OUT", "t_stat": t2_t, "p_value": t2_p}])

# Tablo 3: Placebo
rng = np.random.default_rng(PLACEBO_SEED)
all_dates = bt["effective_date"].unique()
placebo_rows = []
for _, row in bt.iterrows():
    fake_date = pd.Timestamp(rng.choice(all_dates))
    pre_bps, _ = cum_excess(row["ticker"], fake_date, WINDOW_PRE, WINDOW_POST)
    placebo_rows.append({"change_type": row["change_type"], "pre_bps": pre_bps})
pb = pd.DataFrame(placebo_rows).dropna(subset=["pre_bps"])

pb_in  = pb[pb["change_type"] == "IN"]["pre_bps"]
pb_out = pb[pb["change_type"] == "OUT"]["pre_bps"]
t3_rows = [group_stats(pb_in, "IN placebo"), group_stats(pb_out, "OUT placebo")]
t3 = pd.DataFrame(t3_rows)
t3["q_value_BH"] = bh_fdr(t3["p_value"].values)

# --- RAPOR ---
print(f"\nTABLO 1 — IN ve OUT grubu: 10-gun oncesi kumülatif excess return (t vs 0)")
print(f"  Maliyet: LOW={COST_LOW}bps, HIGH={COST_HIGH}bps round-trip")
print(t1[["Grup","n","Ort_bps","t_stat","p_value","q_value_BH","Net_LOW_bps","Net_HIGH_bps"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print(f"\nTABLO 2 — IN vs OUT (Welch t-test)")
print(t2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print(f"\nTABLO 3 — PLACEBO (rastgele tarih, seed=42)")
print(t3[["Grup","n","Ort_bps","t_stat","p_value","q_value_BH"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

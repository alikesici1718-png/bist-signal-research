"""
pead_earnings_surprise_pilot.py

Pilot: 25 likit BIST sembolü için earnings surprise → PEAD excess return backtest.
Veri kaynagi: kap_data/kap_financial_report_dates.csv (disclosure_index bilgileri)
HTML parse: dogrudan https://www.kap.org.tr/tr/Bildirim/{disc_ind}
(pykap'in byCriteria API'si 500 dondurdugu icin disclosure listesi CSV'den alinir)
"""

import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
TICKERS = [
    "THYAO", "AKBNK", "GARAN", "ISCTR", "YKBNK",
    "SAHOL", "KCHOL", "SISE",  "EREGL", "BIMAS",
    "TUPRS", "ASELS", "FROTO", "TOASO", "TCELL",
    "PGSUS", "ARCLK", "VAKBN", "HALKB", "PETKM",
    "KOZAL", "ENKAI", "MGROS", "ULKER", "TAVHL",
]

REQUEST_DELAY_SECONDS = 8
MAX_PERIODS_PER_TICKER = 12   # son 12 donem (YoY icin 3 yil yeterli)
KAP_CSV   = Path("kap_data/kap_financial_report_dates.csv")
DATA_DIR  = Path("data")
XU100_CSV = Path("data_market/XU100.csv")
PILOT_CSV = Path("kap_data/pilot_earnings.csv")
HORIZONS  = [5, 10, 20]
SEED      = 42

SCENARIOS = {
    "Midas+DUSUK":       {"commission_bps": 1.0,  "spread_frac": 0.10},
    "AtaYatirim+YUKSEK": {"commission_bps": 38.0, "spread_frac": 0.20},
}

NET_KAR_PATTERNS = [
    "net dönem", "net kar", "dönem net", "net profit", "profit for the period",
    "net income", "dönem kârı", "dönem zararı", "net dönem kârı", "net dönem zararı",
    "net donem", "donem kari", "donem zarari",
]

TR_CLASS  = re.compile(r".*_role_.*data-input-row.*presentation-enabled")
LBL_CLASS = "gwt-Label multi-language-content content-tr"
VAL_CLASS = re.compile(r"taxonomy-context-value.*")

# ---------------------------------------------------------------------------
# HTML parse
# ---------------------------------------------------------------------------

def parse_disclosure_page(disc_ind: int) -> dict:
    url = f"https://www.kap.org.tr/tr/Bildirim/{disc_ind}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html5lib")
    results = {}
    for row in soup.find_all("tr", {"class": TR_CLASS}):
        lbl_el = row.find(True, {"class": LBL_CLASS})
        val_el = row.find(True, {"class": VAL_CLASS})
        if lbl_el and val_el:
            lbl = lbl_el.get_text().strip()
            val_txt = val_el.get_text().strip()
            try:
                results[lbl] = float(val_txt.replace(".", "").replace(",", "."))
            except ValueError:
                results[lbl] = val_txt
    return results


def find_net_kar(results: dict) -> tuple:
    if not results:
        return None, None
    for key in results:
        key_lower = key.lower()
        for pat in NET_KAR_PATTERNS:
            if pat in key_lower:
                val = results[key]
                try:
                    return float(val), key
                except (TypeError, ValueError):
                    return None, key
    return None, None


# ---------------------------------------------------------------------------
# Fiyat panelleri
# ---------------------------------------------------------------------------

def load_price_panels():
    files = glob.glob(str(DATA_DIR / "*.csv"))
    opens, highs, lows = {}, {}, {}
    for f in files:
        t = Path(f).stem.upper()
        try:
            df = pd.read_csv(f, parse_dates=["Date"], index_col="Date")
        except Exception:
            try:
                df = pd.read_csv(f, parse_dates=[0], index_col=0)
                df.index.name = "Date"
            except Exception:
                continue
        df.sort_index(inplace=True)
        if "Open" in df.columns: opens[t] = df["Open"]
        if "High" in df.columns: highs[t] = df["High"]
        if "Low"  in df.columns: lows[t]  = df["Low"]
    return pd.DataFrame(opens), pd.DataFrame(highs), pd.DataFrame(lows)


def load_xu100():
    df = pd.read_csv(XU100_CSV, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df["Open"]


def compute_hl_baseline(high_panel, low_panel):
    hl_ratio = (high_panel - low_panel) / ((high_panel + low_panel) / 2)
    return hl_ratio.rolling(60, min_periods=20).median().shift(1)


# ---------------------------------------------------------------------------
# Sürpriz hesaplama
# ---------------------------------------------------------------------------

def compute_surprises(earnings_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, grp in earnings_df.groupby("ticker"):
        grp = grp.dropna(subset=["net_kar", "year", "period"]).copy()
        grp["year"]   = grp["year"].astype(float)
        grp["period"] = grp["period"].astype(float)
        grp = grp.sort_values(["year", "period"])
        for _, row in grp.iterrows():
            yr = row["year"]; q = row["period"]
            prev = grp[(grp["year"] == yr - 1) & (grp["period"] == q)]
            if prev.empty:
                continue
            prev_kar = prev.iloc[0]["net_kar"]
            if prev_kar == 0 or pd.isna(prev_kar):
                continue
            surprise = (row["net_kar"] - prev_kar) / abs(prev_kar)
            rows.append({
                "ticker":    ticker,
                "disc_ind":  int(row["disc_ind"]),
                "period_str": f"{int(yr)}-Q{int(q)}",
                "net_kar":   row["net_kar"],
                "prev_kar":  prev_kar,
                "surprise":  surprise,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def build_event_returns(event_df, kap_df, open_panel, hl_baseline, xu100_open, horizon):
    trading_days = open_panel.index
    rows = []
    for _, ev in event_df.iterrows():
        ticker   = ev["ticker"]
        disc_ind = int(ev["disc_ind"])
        if ticker not in open_panel.columns:
            continue
        match = kap_df[kap_df["disclosure_index"] == disc_ind]
        if match.empty:
            continue
        event_date = match.iloc[0]["publish_date"].normalize()
        future_days = trading_days[trading_days > event_date]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        entry_pos  = trading_days.get_loc(entry_date)
        exit_pos   = entry_pos + horizon
        if exit_pos >= len(trading_days):
            continue
        exit_date = trading_days[exit_pos]
        ep = open_panel.at[entry_date, ticker]
        xp = open_panel.at[exit_date, ticker]
        if pd.isna(ep) or pd.isna(xp) or ep <= 0:
            continue
        xu_e = xu100_open.get(entry_date, np.nan)
        xu_x = xu100_open.get(exit_date, np.nan)
        if pd.isna(xu_e) or pd.isna(xu_x) or xu_e <= 0:
            continue
        excess = (xp / ep - 1) - (xu_x / xu_e - 1)
        hl_val = hl_baseline.at[entry_date, ticker] if ticker in hl_baseline.columns else np.nan
        rows.append({"ticker": ticker, "excess_return": excess, "hl_baseline": hl_val})
    return pd.DataFrame(rows)


def run_backtest(event_df, kap_df, open_panel, hl_baseline, xu100_open):
    all_results = []
    for horizon in HORIZONS:
        df = build_event_returns(event_df, kap_df, open_panel, hl_baseline, xu100_open, horizon)
        if df.empty:
            continue
        excess_bps = df["excess_return"].values * 10000
        t_stat, p_value = ttest_1samp(excess_bps, 0)
        row = {
            "horizon": horizon, "n": len(excess_bps),
            "brut_excess_bps": excess_bps.mean(),
            "t_stat": t_stat, "p_value": p_value,
        }
        for scen_name, scen in SCENARIOS.items():
            spread_bps = df["hl_baseline"].fillna(df["hl_baseline"].median()) * scen["spread_frac"] * 10000 * 2
            net = excess_bps - (spread_bps.values + scen["commission_bps"] * 2)
            row[f"net_{scen_name}_bps"] = net.mean()
        all_results.append(row)
    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return result_df
    valid = result_df["p_value"].notna()
    p_vals = result_df.loc[valid, "p_value"].values
    if len(p_vals) > 0:
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        result_df.loc[valid, "q_value"] = q_vals
    else:
        result_df["q_value"] = np.nan
    return result_df


# ---------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------

def build_placebo_events(tickers, open_panel, n_per_ticker: dict, rng):
    trading_days = open_panel.index
    max_horizon  = max(HORIZONS)
    valid_days   = trading_days[: len(trading_days) - max_horizon - 1]
    rows = []
    for ticker in tickers:
        if ticker not in open_panel.columns:
            continue
        n = min(n_per_ticker.get(ticker, 5), len(valid_days))
        chosen = rng.choice(valid_days, size=n, replace=False)
        for d in chosen:
            rows.append({"ticker": ticker, "event_date": pd.Timestamp(d)})
    return pd.DataFrame(rows)


def build_placebo_returns(placebo_df, open_panel, hl_baseline, xu100_open, horizon):
    trading_days = open_panel.index
    rows = []
    for _, ev in placebo_df.iterrows():
        ticker     = ev["ticker"]
        event_date = ev["event_date"]
        future_days = trading_days[trading_days > event_date]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        entry_pos  = trading_days.get_loc(entry_date)
        exit_pos   = entry_pos + horizon
        if exit_pos >= len(trading_days):
            continue
        exit_date = trading_days[exit_pos]
        ep = open_panel.at[entry_date, ticker]
        xp = open_panel.at[exit_date, ticker]
        if pd.isna(ep) or pd.isna(xp) or ep <= 0:
            continue
        xu_e = xu100_open.get(entry_date, np.nan)
        xu_x = xu100_open.get(exit_date, np.nan)
        if pd.isna(xu_e) or pd.isna(xu_x) or xu_e <= 0:
            continue
        excess = (xp / ep - 1) - (xu_x / xu_e - 1)
        hl_val = hl_baseline.at[entry_date, ticker] if ticker in hl_baseline.columns else np.nan
        rows.append({"ticker": ticker, "excess_return": excess, "hl_baseline": hl_val})
    return pd.DataFrame(rows)


def run_placebo(placebo_df, open_panel, hl_baseline, xu100_open):
    all_results = []
    for horizon in HORIZONS:
        df = build_placebo_returns(placebo_df, open_panel, hl_baseline, xu100_open, horizon)
        if df.empty:
            continue
        excess_bps = df["excess_return"].values * 10000
        t_stat, p_value = ttest_1samp(excess_bps, 0)
        row = {
            "horizon": horizon, "n": len(excess_bps),
            "brut_excess_bps": excess_bps.mean(),
            "t_stat": t_stat, "p_value": p_value,
        }
        all_results.append(row)
    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return result_df
    valid = result_df["p_value"].notna()
    p_vals = result_df.loc[valid, "p_value"].values
    if len(p_vals) > 0:
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        result_df.loc[valid, "q_value"] = q_vals
    else:
        result_df["q_value"] = np.nan
    return result_df


# ---------------------------------------------------------------------------
# Yazdırma
# ---------------------------------------------------------------------------

def print_table(result_df, title):
    print(f"\n{'='*105}")
    print(title)
    print(f"{'='*105}")
    if result_df.empty:
        print("  Veri yok.")
        return
    hdr = f"{'Horizon':>8} {'N':>5} {'BrutExcess':>11} {'t-stat':>8} {'p-value':>10} {'q-value':>10}"
    for s in SCENARIOS:
        hdr += f"  {'Net '+s:>26}"
    print(hdr)
    print("-" * 105)
    for _, r in result_df.iterrows():
        line = (f"{int(r['horizon']):>8} {int(r['n']):>5} {r['brut_excess_bps']:>11.1f} "
                f"{r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['q_value']:>10.4f}")
        for s in SCENARIOS:
            v = r.get(f"net_{s}_bps", float("nan"))
            line += f"  {v:>26.1f}"
        print(line)
    print("=" * 105)


def print_placebo_table(result_df):
    print(f"\n{'='*70}")
    print("PLACEBO (seed=42, ayni 25 sembol, rastgele tarih)")
    print(f"{'='*70}")
    if result_df.empty:
        print("  Veri yok.")
        return
    print(f"{'Horizon':>8} {'N':>6} {'BrutExcess':>11} {'t-stat':>8} {'p-value':>10} {'q-value':>10}")
    print("-" * 70)
    for _, r in result_df.iterrows():
        print(f"{int(r['horizon']):>8} {int(r['n']):>6} {r['brut_excess_bps']:>11.1f} "
              f"{r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['q_value']:>10.4f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fiyat panelleri yukleniyor...")
    open_panel, high_panel, low_panel = load_price_panels()
    hl_baseline = compute_hl_baseline(high_panel, low_panel)
    xu100_open  = load_xu100()

    print("KAP bildirimleri yukleniyor...")
    kap_df = pd.read_csv(KAP_CSV)
    kap_df["publish_date"] = pd.to_datetime(kap_df["publish_date"], dayfirst=True, errors="coerce")
    kap_df = kap_df.dropna(subset=["publish_date", "disclosure_index"])
    kap_df["disclosure_index"] = kap_df["disclosure_index"].astype(int)

    # Pilot: 25 ticker, her biri icin son MAX_PERIODS_PER_TICKER donem
    pilot_kap = kap_df[kap_df["ticker"].isin(TICKERS)].copy()
    pilot_kap = pilot_kap.sort_values(["ticker", "publish_date"], ascending=[True, False])
    pilot_kap = pilot_kap.groupby("ticker").head(MAX_PERIODS_PER_TICKER).reset_index(drop=True)

    mevcut = pilot_kap["ticker"].nunique()
    print(f"Pilot ticker: {mevcut}/25 (KOZAL CSV'de yok)")
    print(f"Cekiyor: {len(pilot_kap)} bildirim x {REQUEST_DELAY_SECONDS}s = ~{len(pilot_kap)*REQUEST_DELAY_SECONDS//60}dk\n")

    # --- Finansal veri çekme ---
    all_rows = []
    label_log = {}   # ticker -> (disc_ind, ilk 20 etiket)
    errors = []

    for i, (_, rec) in enumerate(pilot_kap.iterrows(), 1):
        ticker   = rec["ticker"]
        disc_ind = int(rec["disclosure_index"])
        year     = rec.get("year", None)
        period   = rec.get("period", None)

        print(f"[{i:>3}/{len(pilot_kap)}] {ticker} {year} Q{period}  disc={disc_ind}", end="  ")

        try:
            time.sleep(REQUEST_DELAY_SECONDS)
            results = parse_disclosure_page(disc_ind)
            net_kar, label_found = find_net_kar(results)

            if ticker not in label_log and results:
                label_log[ticker] = (disc_ind, list(results.keys())[:20])

            print(f"etiket={label_found!r:45s}  net_kar={net_kar}")
            all_rows.append({
                "ticker":   ticker,
                "disc_ind": disc_ind,
                "year":     year,
                "period":   period,
                "net_kar":  net_kar,
                "label":    label_found,
            })

        except Exception as e:
            err_msg = str(e)
            print(f"HATA: {err_msg[:80]}")
            errors.append({"ticker": ticker, "disc_ind": disc_ind, "hata": err_msg})
            all_rows.append({
                "ticker": ticker, "disc_ind": disc_ind,
                "year": year, "period": period,
                "net_kar": None, "label": None,
            })

    earnings_df = pd.DataFrame(all_rows)
    earnings_df.to_csv(PILOT_CSV, index=False)

    found     = earnings_df.dropna(subset=["net_kar"])
    not_found = earnings_df[earnings_df["net_kar"].isna()]

    print(f"\npilot_earnings.csv kaydedildi: {len(earnings_df)} satir")
    print(f"Net kar bulundu    : {len(found)} ({found['ticker'].nunique()} sembol)")
    print(f"Net kar bulunamadi : {len(not_found)}")
    if errors:
        print(f"HTTP/parse hata    : {len(errors)}")
    if not_found[not_found["label"].isna()].shape[0] > 0:
        # Etiket bulunamayan satirlarda ilk bulunan etiket ornegi logla
        sample = not_found[not_found["label"].isna()].head(3)
        print(f"Etiket bulunamayan ornek disc_ind: {list(sample['disc_ind'])}")

    # Etiket ozeti
    print("\n--- Bulunan net kar etiketleri (ticker basi ilk donem) ---")
    for tk in sorted(label_log):
        disc, lbls = label_log[tk]
        net_lbl = [l for l in lbls if any(p in l.lower() for p in NET_KAR_PATTERNS)]
        print(f"  {tk}: {net_lbl[:3]}")

    # --- Sürpriz hesaplama ---
    print("\nSurpriz hesaplaniyor (YoY net kar)...")
    surprise_df = compute_surprises(found)
    print(f"Surpriz hesaplanan event sayisi: {len(surprise_df)}")

    if surprise_df.empty:
        print("\nSurpriz verisi yok — backtest yapilemiyor.")
        print("(Muhtemel sebep: her ticker icin MAX_PERIODS_PER_TICKER=12 limit, YoY icin hem "
              "baz donem hem karsilastirma donemi gerekiyor; veri penceresi dar olabilir.)")
        return

    q30 = surprise_df["surprise"].quantile(0.30)
    q70 = surprise_df["surprise"].quantile(0.70)
    top_df    = surprise_df[surprise_df["surprise"] >= q70].copy()
    bottom_df = surprise_df[surprise_df["surprise"] <= q30].copy()
    print(f"Top %%30 (>= {q70:.2f}): {len(top_df)} event")
    print(f"Bottom %%30 (<= {q30:.2f}): {len(bottom_df)} event")

    # --- Backtestler ---
    print("\nTop surpriz backtest calistiriliyor...")
    top_result = run_backtest(top_df, kap_df, open_panel, hl_baseline, xu100_open)

    print("Bottom surpriz backtest calistiriliyor...")
    bot_result = run_backtest(bottom_df, kap_df, open_panel, hl_baseline, xu100_open)

    # --- Placebo ---
    print("Placebo calistiriliyor (seed=42)...")
    n_per = earnings_df.groupby("ticker").size().to_dict()
    rng   = np.random.default_rng(SEED)
    placebo_df  = build_placebo_events(TICKERS, open_panel, n_per, rng)
    placebo_res = run_placebo(placebo_df, open_panel, hl_baseline, xu100_open)

    # --- Rapor ---
    print("\n\n" + "=" * 105)
    print("PILOT EARNINGS SURPRISE BACKTEST RAPORU")
    print("=" * 105)
    print(f"Sembol sayisi       : {mevcut}/25  (KOZAL CSV'de mevcut degil)")
    print(f"Cekilen donem raporu: {len(earnings_df)}")
    print(f"Net kar bulundu     : {len(found)} satir, {found['ticker'].nunique()} sembol")
    print(f"Net kar bulunamadi  : {len(not_found)} satir")
    print(f"HTTP/parse hatasi   : {len(errors)}")
    print(f"Surpriz hesaplanan  : {len(surprise_df)} event")
    print(f"Top %%30 threshold  : surprise >= {q70:.4f}")
    print(f"Bottom %%30 threshold: surprise <= {q30:.4f}")

    print_table(top_result, "TOP %30 POZITIF SURPRIZ — Excess Return (XU100'e gore)")
    print_table(bot_result, "BOTTOM %30 NEGATIF SURPRIZ — Excess Return (XU100'e gore)")
    print_placebo_table(placebo_res)


if __name__ == "__main__":
    main()

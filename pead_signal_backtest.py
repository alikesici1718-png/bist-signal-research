"""
pead_signal_backtest.py

Post-Earnings-Announcement-Drift (PEAD) backtest.

Sinyal: KAP'tan cekilen finansal rapor bildirimi (kap_data/kap_financial_report_dates.csv)
Giris:  publish_date'den sonraki ilk islem gununun Acilis fiyati (t+1 Open)
Cikis:  giris gununun horizon islem gunü sonrasinin acilis fiyati

Maliyetler: check_net_returns.py ile ayni HL_baseline metodolojisi, 2 senaryo:
  - Senaryo A: Midas (1 bps komisyon) + DUSUK spread (HL_baseline x 0.10)
  - Senaryo B: AtaYatirim (38 bps komisyon) + YUKSEK spread (HL_baseline x 0.20)

NOT: config/symbols.txt statik bir listedir, delist olmus sirketler evrenden eksik
olabilir (survivorship bias, onceki denetimde tespit edildi).
"""

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
KAP_CSV = Path("kap_data/kap_financial_report_dates.csv")
DATA_DIR = Path("data")
HORIZONS = [5, 10, 20]

SCENARIOS = {
    "Midas+DUSUK":     {"commission_bps": 1.0,  "spread_frac": 0.10},
    "AtaYatirim+YUKSEK": {"commission_bps": 38.0, "spread_frac": 0.20},
}


# ---------------------------------------------------------------------------
# 1. Fiyat panellerini yukle (data/ klasoru, ayni pattern check_net_returns.py)
# ---------------------------------------------------------------------------
def load_price_panels():
    files = glob.glob(str(DATA_DIR / "*.csv"))
    opens, closes, highs, lows = {}, {}, {}, {}

    for f in files:
        ticker = Path(f).stem.upper()
        try:
            df = pd.read_csv(f, parse_dates=["Date"], index_col="Date")
        except Exception:
            try:
                df = pd.read_csv(f, parse_dates=[0], index_col=0)
                df.index.name = "Date"
            except Exception:
                continue
        df.sort_index(inplace=True)
        if "Open" in df.columns:
            opens[ticker] = df["Open"]
        if "Close" in df.columns:
            closes[ticker] = df["Close"]
        if "High" in df.columns:
            highs[ticker] = df["High"]
        if "Low" in df.columns:
            lows[ticker] = df["Low"]

    open_panel  = pd.DataFrame(opens)
    close_panel = pd.DataFrame(closes)
    high_panel  = pd.DataFrame(highs)
    low_panel   = pd.DataFrame(lows)

    # Ortak tarih indeksi
    idx = open_panel.index
    open_panel  = open_panel.reindex(idx)
    close_panel = close_panel.reindex(idx)
    high_panel  = high_panel.reindex(idx)
    low_panel   = low_panel.reindex(idx)

    return open_panel, close_panel, high_panel, low_panel


# ---------------------------------------------------------------------------
# 2. HL_baseline spread tahmini (check_net_returns.py ile ayni)
# ---------------------------------------------------------------------------
def compute_hl_baseline(high_panel, low_panel):
    hl_ratio = (high_panel - low_panel) / ((high_panel + low_panel) / 2)
    hl_baseline = hl_ratio.rolling(60, min_periods=20).median().shift(1)
    return hl_baseline


# ---------------------------------------------------------------------------
# 3. KAP bildirim tarihlerini yukle ve parse et
# ---------------------------------------------------------------------------
def load_kap_events():
    df = pd.read_csv(KAP_CSV)
    # publish_date formati: "DD.MM.YYYY HH:MM:SS"
    df["publish_date"] = pd.to_datetime(df["publish_date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["event_date"] = df["publish_date"].dt.normalize()  # gun duzeyine indir
    return df


# ---------------------------------------------------------------------------
# 4. Her olay icin giris/cikis tarihlerini bul
# ---------------------------------------------------------------------------
def build_event_returns(events_df, open_panel, hl_baseline, horizon):
    trading_days = open_panel.index
    rows = []

    for _, ev in events_df.iterrows():
        ticker = ev["ticker"]
        if ticker not in open_panel.columns:
            continue
        event_date = ev["event_date"]

        # Giris: event_date'den SONRAKI ilk islem gunu
        future_days = trading_days[trading_days > event_date]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]

        # Cikis: giris gununun horizon islem gunü sonrasi (t+horizon+1 Open)
        entry_pos = trading_days.get_loc(entry_date)
        exit_pos = entry_pos + horizon
        if exit_pos >= len(trading_days):
            continue
        exit_date = trading_days[exit_pos]

        entry_price = open_panel.at[entry_date, ticker]
        exit_price  = open_panel.at[exit_date, ticker]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # HL_baseline -> spread proxy (round-trip bps)
        hl_val = hl_baseline.at[entry_date, ticker] if ticker in hl_baseline.columns else np.nan

        gross_return = exit_price / entry_price - 1.0

        rows.append({
            "ticker": ticker,
            "event_date": event_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "gross_return": gross_return,
            "hl_baseline": hl_val,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. BH-FDR ile istatistik hesapla
# ---------------------------------------------------------------------------
def run_stats(events_df, open_panel, hl_baseline):
    all_results = []

    for horizon in HORIZONS:
        df = build_event_returns(events_df, open_panel, hl_baseline, horizon)
        if df.empty:
            continue

        gross_bps = df["gross_return"].values * 10000

        # t-test (olay bazinda, sembol-gun seviyesinde)
        t_stat, p_value = ttest_1samp(gross_bps, 0)
        n = len(gross_bps)
        gross_mean = gross_bps.mean()
        win_rate = (gross_bps > 0).mean()

        row = {
            "horizon": horizon,
            "n": n,
            "brut_bps": gross_mean,
            "t_stat": t_stat,
            "p_value": p_value,
            "win_rate": win_rate,
        }

        # Maliyet senaryolari
        for scen_name, scen in SCENARIOS.items():
            spread_bps = df["hl_baseline"].fillna(df["hl_baseline"].median()) * scen["spread_frac"] * 10000 * 2
            total_cost_bps = spread_bps + scen["commission_bps"] * 2
            net_bps = gross_bps - total_cost_bps.values
            row[f"net_{scen_name}_bps"] = net_bps.mean()

        all_results.append(row)

    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return result_df

    # BH-FDR duzeltmesi
    valid = result_df["p_value"].notna()
    p_vals = result_df.loc[valid, "p_value"].values
    if len(p_vals) > 0:
        _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
        result_df.loc[valid, "q_value"] = q_vals
    else:
        result_df["q_value"] = np.nan

    return result_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Fiyat panelleri yukleniyor...")
    open_panel, close_panel, high_panel, low_panel = load_price_panels()
    print(f"  {open_panel.shape[1]} sembol, {len(open_panel)} islem gunu")

    print("HL_baseline hesaplaniyor...")
    hl_baseline = compute_hl_baseline(high_panel, low_panel)

    print("KAP bildirimleri yukleniyor...")
    events_df = load_kap_events()
    print(f"  {len(events_df)} bildirim, {events_df['ticker'].nunique()} sembol")

    print("PEAD backtest calistiriliyor...\n")
    result_df = run_stats(events_df, open_panel, hl_baseline)

    if result_df.empty:
        print("Sonuc yok.")
        return

    # Cikti tablosu
    scen_cols = [f"net_{s}_bps" for s in SCENARIOS]
    col_order = ["horizon", "n", "brut_bps", "t_stat", "p_value", "q_value"] + scen_cols + ["win_rate"]
    result_df = result_df[col_order]

    print("=" * 90)
    print("PEAD BACKTEST SONUCLARI")
    print("=" * 90)
    print(f"{'Horizon':>8} {'N':>6} {'Brut(bps)':>10} {'t-stat':>8} {'p-value':>10} {'q-value':>10}", end="")
    for s in SCENARIOS:
        print(f" {'Net '+s+' bps':>22}", end="")
    print(f" {'WinRate':>8}")
    print("-" * 90)

    for _, r in result_df.iterrows():
        print(f"{int(r['horizon']):>8} {int(r['n']):>6} {r['brut_bps']:>10.1f} {r['t_stat']:>8.2f} {r['p_value']:>10.4f} {r['q_value']:>10.4f}", end="")
        for s in SCENARIOS:
            val = r[f"net_{s}_bps"]
            print(f" {val:>22.1f}", end="")
        print(f" {r['win_rate']:>8.1%}")

    print("=" * 90)
    print()
    print("NOT: config/symbols.txt statik bir listedir, delist olmus sirketler evrenden")
    print("eksik olabilir (survivorship bias, onceki denetimde tespit edildi).")


if __name__ == "__main__":
    main()

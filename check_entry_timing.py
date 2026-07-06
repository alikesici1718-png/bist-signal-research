"""
check_entry_timing.py

Amac: extreme_down_reversal sinyalinin GERCEKTEN trade edilebilir olup
olmadigini test eder.

Sorun: Sinyal t=0 gunun KAPANIS fiyatina gore tanimli (cross-sectional en
kotu %5). Bu sinyali "gordugun" an piyasa zaten kapanmis. En erken
pozisyon acabilecegin an t+1 GUNUNUN ACILISI'dir, t=0 kapanisi degil.

Bu script sunlari olcer:
  1. t=0 kapanis -> t+1 acilis arasindaki "gap" (gece/hafta sonu hareketi).
     Eger bu gap zaten cok negatifse, sinyali gordugunde fiyat cogunlukla
     dusmus olur -- excess return'un onemli bir kismini kacirirsin.
  2. Iki farkli giris senaryosu ile forward excess return'u yeniden hesaplar:
     A) "Kagit uzerinde" (t=0 KAPANIS'tan girildigini varsayarak) -- bu
        onceki scriptlerin yaptigi, gercekci degil.
     B) "Gercekci" (t+1 ACILIS'tan girildigini varsayarak) -- gercekte
        yapabilecegin.
  3. Ikisi arasindaki farkin ne kadari "kacirilmis" excess return oldugunu
     gosterir.

Not: Bu script Open kolonunu kullanir, dolayisiyla CSV'lerde Open kolonu
olmasi gerekir (get_bist_data.py ciktisinda var).

Kullanim:
    python check_entry_timing.py
Cikti:
    entry_timing_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
FWD_HORIZONS = [1, 3, 5]
EXCLUDE_NAMES = {"USDTRY", "USDTRY=X"}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_TXT = os.path.join(OUTPUT_DIR, "entry_timing_report.txt")


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        if not {"Date", "Open", "Close"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = {}
    skipped_no_open = 0
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name in EXCLUDE_NAMES:
            continue
        df = load_symbol(f)
        if df is not None:
            symbols[name] = df
        else:
            skipped_no_open += 1

    print(f"Loaded {len(symbols)} symbols (Open kolonu olmayan/kisa {skipped_no_open} dosya atlandi)")

    all_dates = pd.to_datetime(
        sorted(set().union(*[df["Date"].unique() for df in symbols.values()]))
    )

    close_series = {}
    open_series = {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        close_series[name] = s["Close"].reindex(all_dates)
        open_series[name] = s["Open"].reindex(all_dates)

    close_panel = pd.concat(close_series, axis=1)
    open_panel = pd.concat(open_series, axis=1)
    close_panel.index.name = None
    open_panel.index.name = None

    ret_panel = close_panel.pct_change()  # close-to-close, oran (0.05 = %5)
    market_ret = ret_panel.mean(axis=1, skipna=True).fillna(0)

    # gap: t=0 kapanis -> t+1 acilis arasindaki degisim
    # gap[t+1] = Open[t+1] / Close[t] - 1
    gap_panel = open_panel / close_panel.shift(1) - 1

    # --- extreme_down events (t=0 kapanisina gore tanimli, ayni onceki tanim)
    events = []
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        for name in losers.index:
            events.append((name, d))

    df_events = pd.DataFrame(events, columns=["symbol", "date"])
    print(f"Total extreme_down events: {len(df_events)}")

    report = []
    report.append(f"Total extreme_down_reversal events: {len(df_events)}")
    report.append("")

    # --- Gap analizi: t=0 kapanistan t+1 acilisa ne kadar hareket var?
    gap_values = []
    for name, d in zip(df_events["symbol"], df_events["date"]):
        if name not in gap_panel.columns:
            continue
        idx = ret_panel.index.get_indexer([d])[0]
        if idx < 0 or idx + 1 >= len(ret_panel.index):
            continue
        next_date = ret_panel.index[idx + 1]
        g = gap_panel.loc[next_date, name]
        if pd.notna(g):
            gap_values.append(g)

    gap_series = pd.Series(gap_values)
    report.append("=== Gap analizi: t=0 kapanis -> t+1 acilis ===")
    report.append(f"  Ortalama gap: {gap_series.mean()*10000:.1f} bps (n={len(gap_series)})")
    report.append(f"  Medyan gap  : {gap_series.median()*10000:.1f} bps")
    report.append(f"  Negatif gap (%<0) oran: {(gap_series < 0).mean()*100:.1f}%")
    report.append("(Negatifse: sinyali gordugunde fiyat zaten dusmus oluyor demektir --")
    report.append(" acilista girsen bile bir kisim hareketi kacirmis olursun.)")
    report.append("")

    # --- Iki senaryo ile forward excess return
    def forward_excess_close_entry(events_df, horizon):
        """Senaryo A: t=0 KAPANIS'tan giris varsayimi (kagit uzerinde, gercekci degil)."""
        rows = []
        for name, d in zip(events_df["symbol"], events_df["date"]):
            if name not in ret_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx < 0 or idx + horizon >= len(ret_panel.index):
                continue
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            stock_series = ret_panel.loc[fwd_dates, name]
            if stock_series.isna().any():
                continue
            stock_fwd = (1 + stock_series).prod() - 1
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1
            rows.append(stock_fwd - mkt_fwd)
        return pd.Series(rows)

    def forward_excess_open_entry(events_df, horizon):
        """Senaryo B: t+1 ACILIS'tan giris varsayimi (gercekci, gercekten trade edilebilir)."""
        rows = []
        for name, d in zip(events_df["symbol"], events_df["date"]):
            if name not in ret_panel.columns or name not in open_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx < 0 or idx + horizon >= len(ret_panel.index):
                continue
            entry_date = ret_panel.index[idx + 1]
            exit_idx = idx + horizon
            exit_date = ret_panel.index[exit_idx]

            entry_price = open_panel.loc[entry_date, name]
            exit_price = close_panel.loc[exit_date, name]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue
            stock_fwd = exit_price / entry_price - 1

            # market icin ayni donemi (entry_date dahil, exit_date dahil) kapanis-kapanis kullan
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1

            rows.append(stock_fwd - mkt_fwd)
        return pd.Series(rows)

    report.append("=== Senaryo A (kagit uzerinde, t=0 kapanistan giris -- ONCEKI SCRIPTLERIN YAPTIGI) ===")
    for horizon in FWD_HORIZONS:
        s = forward_excess_close_entry(df_events, horizon)
        report.append(f"  Horizon {horizon}d: {s.mean()*10000:.1f} bps (n={len(s)})")
    report.append("")

    report.append("=== Senaryo B (gercekci, t+1 acilistan giris) ===")
    for horizon in FWD_HORIZONS:
        s = forward_excess_open_entry(df_events, horizon)
        report.append(f"  Horizon {horizon}d: {s.mean()*10000:.1f} bps (n={len(s)})")
    report.append("")

    report.append("YORUM: Senaryo B, Senaryo A'dan onemli olcude daha dusuk (0'a daha yakin)")
    report.append("ise, 'edge'in buyuk kismi zaten senin sinyali gordugunde olmus olan")
    report.append("gap hareketinden geliyor demektir -- yani KACIRIYORSUN, trade edilebilir")
    report.append("degil. Eger B, A'ya yakin kaliyorsa, edge'in cogu t+1'den itibaren hala")
    report.append("orada demektir -- bu durumda gercekten trade edilebilir olma ihtimali var.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nRapor kaydedildi: {REPORT_TXT}")


if __name__ == "__main__":
    main()
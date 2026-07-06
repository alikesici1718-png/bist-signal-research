"""
check_limit_moves.py

Amac: extreme_down_reversal ve volume_spike_2x sinyallerinin, gercek bir
fiyatlanma hareketi mi yoksa BIST limit-down (taban) mekanigi mi
yakaladigini test eder.

Mantik: BIST'te gunluk hareket bandi ~%10 (eski donem) veya ~%20 (daha
sonraki genisletilmis bant) ile sinirlidir. Eger bir "extreme_down" olayi
gercekte limit-down ise, o gunun |pct_change| degeri banda cok yakin
(ornegin -18.5 ile -20.0 arasi) olacaktir. Bu script:

  1. TUM extreme_down_reversal olaylari icin o gunun gercek pct_change'ini
     hesaplar (sinyal tanimindaki cross-sectional en kotu %5 degil,
     ham fiyat degisimi).
  2. Bu degerlerin ne kadari "limit bandina yakin" (<= -18%) dusuyor.
  3. Ayni analizi event sonrasi ardisik gunler icin de yapar (kac gun
     ust uste limit'e yakin hareket var).
  4. Limit-yakini olaylari filtreleyip, KALANLARLA (yani gercek limit
     mekanizmasi olmayan olaylarla) forward excess return'u yeniden
     hesaplar -- eger sinyal bu filtreden sonra ciddi olcude zayiflarsa,
     "edge" in cogu limit-down artefaktidir.

Kullanim:
    python check_limit_moves.py
Cikti:
    limit_check_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
LIMIT_THRESHOLD = -18.0   # bu esigin altindaki gunluk % degisim "limit-yakini" sayilir
FWD_HORIZONS = [1, 3, 5]
EXCLUDE_NAMES = {"USDTRY", "USDTRY=X"}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_TXT = os.path.join(OUTPUT_DIR, "limit_check_report.txt")


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        if not {"Date", "Close", "Volume"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if name in EXCLUDE_NAMES:
            continue
        df = load_symbol(f)
        if df is not None:
            symbols[name] = df

    print(f"Loaded {len(symbols)} symbols")

    all_dates = pd.to_datetime(
        sorted(set().union(*[df["Date"].unique() for df in symbols.values()]))
    )
    close_series = {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        close_series[name] = s["Close"].reindex(all_dates)
    price_panel = pd.concat(close_series, axis=1)
    price_panel.index.name = None
    ret_panel = price_panel.pct_change() * 100  # yuzde olarak
    market_ret = (ret_panel / 100).mean(axis=1, skipna=True).fillna(0)

    # extreme_down events (ayni tanim: cross-sectional en kotu %5)
    events_extreme = []
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        for name in losers.index:
            events_extreme.append((name, d, day_rets[name]))

    df_events = pd.DataFrame(events_extreme, columns=["symbol", "date", "pct_change"])
    print(f"Total extreme_down events: {len(df_events)}")

    near_limit = df_events[df_events["pct_change"] <= LIMIT_THRESHOLD]
    if len(df_events) > 0:
        pct_near_limit = len(near_limit) / len(df_events) * 100
    else:
        pct_near_limit = 0.0

    report = []
    report.append(f"Total extreme_down_reversal events: {len(df_events)}")
    report.append(f"Events with day-of pct_change <= {LIMIT_THRESHOLD}% (limit-yakini): "
                   f"{len(near_limit)} ({pct_near_limit:.1f}%)")
    report.append("")

    # Ardisik gun limit zinciri kontrolu: olay tarihinden sonraki N gun
    # icinde kac tanesi de limit-yakini?
    def consecutive_limit_days(symbol, event_date, max_days=5):
        if symbol not in ret_panel.columns:
            return 0
        idx = ret_panel.index.get_indexer([event_date])[0]
        if idx < 0:
            return 0
        col = ret_panel.columns.get_loc(symbol)
        count = 0
        for k in range(1, max_days + 1):
            if idx + k >= len(ret_panel.index):
                break
            r = ret_panel.iloc[idx + k, col]
            if pd.notna(r) and r <= LIMIT_THRESHOLD:
                count += 1
            else:
                break
        return count

    near_limit = near_limit.copy()
    near_limit["consecutive_limit_days_after"] = near_limit.apply(
        lambda row: consecutive_limit_days(row["symbol"], row["date"]), axis=1
    )
    avg_chain = near_limit["consecutive_limit_days_after"].mean()
    pct_with_chain = (near_limit["consecutive_limit_days_after"] > 0).mean() * 100

    report.append(f"Limit-yakini olaylardan sonra ortalama ardisik limit-gunu sayisi: {avg_chain:.2f}")
    report.append(f"Limit-yakini olaylarin %{pct_with_chain:.1f}'inde olay sonrasi da en az 1 limit-yakini gun var")
    report.append("(Bu yuksekse: sinyal 'taban yapan hisse tabana devam eder' seklinde limit-down zincirini")
    report.append(" yakaliyor demektir -- trade edilebilirligi supheli, cunku bu gunlerde satis/kisa pozisyon")
    report.append(" pratik degil.)")
    report.append("")

    # --- Forward excess return: TUMU vs SADECE limit-disi olaylar
    non_limit_events = df_events[df_events["pct_change"] > LIMIT_THRESHOLD]
    report.append(f"Limit-disi (gercek fiyatlanma) extreme_down olaylari: {len(non_limit_events)} "
                   f"({100 - pct_near_limit:.1f}%)")
    report.append("")

    def forward_excess(events_df, horizon):
        rows = []
        if len(events_df) == 0:
            return pd.Series(rows)
        for name, d in zip(events_df["symbol"], events_df["date"]):
            if name not in ret_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx < 0 or idx + horizon >= len(ret_panel.index):
                continue
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            # NaN gunler (islem durmasi vb.) 0 varsayilmiyor -- gercek NaN
            # olarak birakiliyor, cunku 0 varsaymak limit-down gunlerinin
            # etkisini gizler (tam da test ettigimiz sey). Eksik veri olan
            # olay bu horizon icin atlanir.
            stock_series = ret_panel.loc[fwd_dates, name] / 100
            if stock_series.isna().any():
                continue
            stock_fwd = (1 + stock_series).prod() - 1
            mkt_fwd = (1 + market_ret.loc[fwd_dates].fillna(0)).prod() - 1
            rows.append(stock_fwd - mkt_fwd)
        return pd.Series(rows)

    report.append("=== Forward excess return karsilastirmasi ===")
    for horizon in FWD_HORIZONS:
        all_excess = forward_excess(df_events, horizon)
        limit_excess = forward_excess(near_limit, horizon)
        nonlimit_excess = forward_excess(non_limit_events, horizon)
        report.append(f"--- Horizon {horizon}d ---")
        report.append(f"  TUM olaylar        : {all_excess.mean()*10000:.1f} bps (n={len(all_excess)})")
        report.append(f"  Limit-yakini olaylar: {limit_excess.mean()*10000:.1f} bps (n={len(limit_excess)})")
        report.append(f"  Limit-disi olaylar  : {nonlimit_excess.mean()*10000:.1f} bps (n={len(nonlimit_excess)})")
        report.append("")

    report.append("YORUM: Eger 'Limit-disi olaylar' bps degeri 'Limit-yakini olaylar'a gore")
    report.append("cok daha kucukse (0'a yakinsa), sinyalin gorunen 'edge'inin buyuk kismi")
    report.append("trade edilemeyen limit-down zincirinden geliyor demektir -- gercek, trade")
    report.append("edilebilir bir edge degil.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nRapor kaydedildi: {REPORT_TXT}")


if __name__ == "__main__":
    main()
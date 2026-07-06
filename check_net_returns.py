# ❌ ARŞİVLENDİ: extreme_down_reversal karsiz cikti (3 bagimsiz maliyet yontemiyle dogrulandi, tum senaryolarda net negatif). Detay: check_net_returns.py, capacity_backtest.py, check_net_returns_cs.py sonuclarina bakiniz.
"""
check_net_returns.py

extreme_down_reversal sinyalinin BRUT excess return'unu, iki ayri maliyet
bileseniyle duzelterek NET getiriye cevirir:

  1) KOMISYON+VERGI (sabit, GERCEK): aracı kurum komisyon tarifeleri + BSMV
     + Borsa payi + Takas payi. Bunlar tahmin degil, resmi/yayinlanmis
     oranlar (round-trip, yani alis+satis toplami).
  2) SPREAD (tahmini, HL_RATIO tabanli): gercek bid-ask spread verisi
     olmadigi icin, olay-oncesi "sakin" donem HL_RATIO'sunun bir kesri
     olarak proxy'leniyor. Onceki versiyonda %25/%50 kullanilmisti, bu
     literaturde (Corwin-Schultz benzeri yontemler) kullanilan orandan
     yuksekti; burada %10/%20 kullaniliyor.

Onemli: Bu iki bilesen FARKLI SEYLERI olcer ve TOPLANIR, biri digerinin
yerine gecmez. Komisyon = araci kuruma/borsaya odenen ucret. Spread =
piyasaya girip cikarken fiyatin kendisinin aleyhine calismasi (alis
fiyati > satis fiyati farki). HL_RATIO oynakligi olcer, spread'i degil -
bu yuzden spread_fraction ile kucultulerek kullaniliyor, ama hala kesin
degil.

Ayni temiz altyapiyi kullanir (diagnose_signals.py ile tutarli):
  - ISKUR gibi veri hatali sembolleri otomatik eler (>%100 tek gunluk getiri)
  - Gercek XU100 endeksini benchmark olarak kullanir
  - IPO sonrasi ilk N islem gununu (taze halka arz oynakligi) haric tutar
  - HL_RATIO'yu olay gunun DEGIL, oncesindeki 60 gunluk sakin donem
    medyanindan olcer (bias'i onlemek icin)

Kullanim:
    python check_net_returns.py
Cikti:
    net_returns_report.txt
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "data"
MARKET_INDEX_PATH = "data_market/XU100.csv"
MIN_ROWS = 250
EXTREME_DOWN_PCTL = 0.05
FWD_HORIZONS = [1, 3, 5]

EXCLUDE_NAMES = {"USDTRY", "USDTRY=X", "ISKUR"}
MAX_PLAUSIBLE_DAILY_RETURN = 1.0
IPO_SEASONING_DAYS = 60
BASELINE_WINDOW = 60

# --- KOMISYON+VERGI senaryolari (SABIT, gercek/resmi oranlara dayali) ---
# Round-trip (alis+satis) toplam: araci kurum komisyonu + %5 BSMV + Borsa
# payi (~0.25bps) + Takas payi (~0.04bps), tek yon x2.
# Kaynak: kullanicinin arastirdigi guncel (2026) araci kurum tarifeleri.
# Bunlar TAHMIN DEGIL, yayinlanmis oranlardan hesaplanan sabit rakamlar --
# ama hangi araci kurumun kullanilacagina bagli oldugu icin senaryo olarak
# tutuluyor.
COMMISSION_SCENARIOS_BPS = {
    "Midas_kampanyali": 1.0,
    "YapiKredi_kademeli": 22.0,
    "AtaYatirim": 38.0,
    "HalkYatirim_standart": 64.0,
}

# --- SPREAD senaryolari (TAHMINI, HL_RATIO baseline'inin bir kesri) ---
# Onceki versiyon %25/%50 kullaniyordu (range'in cok buyuk bir kismini
# spread sayiyordu). Corwin-Schultz ve benzeri literatur high-low range'in
# spread'e oranini genelde %10-20 araliginda tahmin eder. Burada da o
# araliga cekiliyor. Hala KESIN DEGIL - gercek order book verisi yok.
SPREAD_SCENARIOS = {
    "DUSUK_spread": 0.10,
    "YUKSEK_spread": 0.20,
}


def load_market_index(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    return df.set_index("Date")["Close"].pct_change()


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        needed = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if not needed.issubset(df.columns):
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

    print(f"Loaded {len(symbols)} symbols with >= {MIN_ROWS} rows")

    all_dates = sorted(set().union(*[set(df["Date"]) for df in symbols.values()]))
    price_cols, open_cols, vol_cols, hl_cols = {}, {}, {}, {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        price_cols[name] = s["Close"]
        open_cols[name] = s["Open"]
        vol_cols[name] = s["Volume"]
        hl_cols[name] = (s["High"] - s["Low"]) / s["Close"]

    price_panel = pd.concat(price_cols, axis=1, sort=False).reindex(all_dates)
    open_panel = pd.concat(open_cols, axis=1, sort=False).reindex(all_dates)
    vol_panel = pd.concat(vol_cols, axis=1, sort=False).reindex(all_dates)
    hl_panel = pd.concat(hl_cols, axis=1, sort=False).reindex(all_dates)

    # HL_RATIO'yu olay penceresinde DEGIL, olay ONCESI "sakin" bir baseline
    # penceresinde olcuyoruz (bkz. modul docstring'i).
    hl_baseline_panel = hl_panel.rolling(BASELINE_WINDOW, min_periods=20).median().shift(1)

    ret_panel = price_panel.pct_change()

    # ISKUR-tipi veri hatasi guvenlik filtresi (diagnose_signals.py ile ayni)
    max_abs_ret = ret_panel.abs().max(axis=0)
    suspect_symbols = max_abs_ret[max_abs_ret > MAX_PLAUSIBLE_DAILY_RETURN].index.tolist()
    if suspect_symbols:
        print(f"UYARI: {len(suspect_symbols)} sembol veri hatasi supheli, cikariliyor: {suspect_symbols}")
        price_panel = price_panel.drop(columns=suspect_symbols)
        open_panel = open_panel.drop(columns=suspect_symbols)
        vol_panel = vol_panel.drop(columns=suspect_symbols)
        hl_panel = hl_panel.drop(columns=suspect_symbols)
        ret_panel = ret_panel.drop(columns=suspect_symbols)
        for s in suspect_symbols:
            symbols.pop(s, None)

    xu100_ret = load_market_index(MARKET_INDEX_PATH)
    if xu100_ret is not None:
        market_ret = xu100_ret.reindex(all_dates)
        market_source = f"XU100 gercek endeks ({MARKET_INDEX_PATH})"
    else:
        market_ret = ret_panel.mean(axis=1, skipna=True)
        market_source = "ESIT-AGIRLIKLI PROXY (XU100 bulunamadi!)"

    dollar_vol = (price_panel * vol_panel)
    liquidity = dollar_vol.median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(liquidity, 3, labels=["illiquid", "mid", "liquid"])

    # IPO seasoning filtresi (diagnose_signals.py ile ayni mantik)
    seasoning_cutoff = {}
    for name in ret_panel.columns:
        first_valid = ret_panel[name].first_valid_index()
        if first_valid is None:
            continue
        pos = ret_panel.index.get_loc(first_valid)
        cutoff_pos = min(pos + IPO_SEASONING_DAYS, len(ret_panel.index) - 1)
        seasoning_cutoff[name] = ret_panel.index[cutoff_pos]

    def in_seasoning_period(name, date):
        cutoff = seasoning_cutoff.get(name)
        return cutoff is not None and date <= cutoff

    events_extreme = []
    for d in ret_panel.index:
        day_rets = ret_panel.loc[d].dropna()
        if len(day_rets) < 20:
            continue
        cutoff = day_rets.quantile(EXTREME_DOWN_PCTL)
        losers = day_rets[day_rets <= cutoff]
        for name in losers.index:
            if in_seasoning_period(name, d):
                continue
            events_extreme.append((name, d))

    print(f"extreme_down_reversal events (IPO/veri-hatasi filtreli): {len(events_extreme)}")

    def compute_event_results(events, horizon):
        rows = []
        for name, d in events:
            if name not in ret_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx == -1 or idx + horizon + 1 >= len(ret_panel.index):
                continue
            entry_date = ret_panel.index[idx + 1]
            exit_date = ret_panel.index[idx + horizon + 1]
            stock_fwd = open_panel.loc[exit_date, name] / open_panel.loc[entry_date, name] - 1
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1
            excess = stock_fwd - mkt_fwd
            hl_baseline = hl_baseline_panel.loc[ret_panel.index[idx], name]
            liq_bucket = tertiles.get(name, np.nan)
            rows.append((name, d, excess, hl_baseline, liq_bucket))
        return pd.DataFrame(rows, columns=["symbol", "date", "excess", "hl_avg", "liquidity"]).dropna(subset=["hl_avg"])

    report = []
    report.append(f"Symbols: {len(symbols)} | Market benchmark: {market_source}")
    report.append(f"extreme_down_reversal events: {len(events_extreme)}")
    report.append("")
    report.append("Maliyet modeli: NET = BRUT(short) - KOMISYON(sabit,gercek) - SPREAD(tahmini,HL bazli)")
    report.append("Komisyon senaryolari (round-trip, bps): " +
                   ", ".join(f"{k}={v}" for k, v in COMMISSION_SCENARIOS_BPS.items()))
    report.append("Spread kesir senaryolari (HL_baseline'in yuzdesi, round-trip icin x2): " +
                   ", ".join(f"{k}={v*100:.0f}%" for k, v in SPREAD_SCENARIOS.items()))
    report.append("")

    for horizon in FWD_HORIZONS:
        df_h = compute_event_results(events_extreme, horizon)
        if df_h.empty:
            report.append(f"--- Horizon {horizon}d: veri yok ---")
            continue

        report.append(f"=== Horizon {horizon}d (n={len(df_h)}) ===")
        gross_bps = df_h["excess"].mean() * 10000
        strategy_return_all = -df_h["excess"] * 10000  # short yonlu brut, bps
        report.append(f"BRUT excess (short yon): {-gross_bps:.1f} bps "
                       f"(ort. brut(short)={strategy_return_all.mean():.1f} bps)")
        report.append("  (not: extreme_down sonrasi excess negatif demek dusus devam ediyor demek -- "
                       "'reversal' stratejisi burada LONG degil, momentum/continuation'a dayanir)")

        for spread_name, spread_frac in SPREAD_SCENARIOS.items():
            spread_bps = df_h["hl_avg"] * spread_frac * 10000 * 2  # round-trip
            report.append(f"  -- Spread senaryosu: {spread_name} ({spread_frac*100:.0f}% of HL, "
                           f"ort={spread_bps.mean():.1f} bps) --")

            for comm_name, comm_bps in COMMISSION_SCENARIOS_BPS.items():
                total_cost_bps = spread_bps + comm_bps
                net_return = strategy_return_all - total_cost_bps
                win_rate = (net_return > 0).mean() * 100

                report.append(f"      [{comm_name:20s}] komisyon={comm_bps:.1f}bps + "
                               f"spread={spread_bps.mean():.1f}bps = "
                               f"toplam={total_cost_bps.mean():.1f}bps | "
                               f"NET={net_return.mean():.1f} bps | kazanma={win_rate:.1f}%")

                bucket_strs = []
                for bucket in ["illiquid", "mid", "liquid"]:
                    mask = df_h["liquidity"] == bucket
                    if mask.sum() == 0:
                        continue
                    net_b = net_return[mask]
                    bucket_strs.append(f"{bucket}=NET:{net_b.mean():.0f}bps/win:{100*(net_b>0).mean():.0f}%/n{mask.sum()}")
                report.append(f"          {' | '.join(bucket_strs)}")
        report.append("")

    report.append("YORUM: Bu bir SHORT stratejisi testi -- extreme_down sonrasi dususun")
    report.append("devam ettigini varsayip short pozisyon acmayi simule eder (Edge Factory'nin")
    report.append("comprehensive_scan.py bulgusuyla tutarli: excess surekli negatif = momentum,")
    report.append("reversal degil).")
    report.append("")
    report.append("Komisyon bileseni artik SABIT ve GERCEK (yayinlanmis araci kurum tarifeleri).")
    report.append("Spread bileseni HALA TAHMINI (HL_RATIO'nun bir kesri) - gercek bid-ask spread")
    report.append("verisi yok. Kesin karar icin gercek order book / Level 2 verisi gerekir.")

    with open("net_returns_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))


if __name__ == "__main__":
    main()
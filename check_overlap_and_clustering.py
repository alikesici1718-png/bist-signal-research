# NOT: Bu dosya check_net_returns_cs.py ile aynidir, yinelenen dosya
"""
check_net_returns_cs.py

Amac: check_net_returns.py'deki KEYFI COST_BPS varsayimlarini (illiquid=150,
mid=60, liquid=25bps) GERCEK VERIDEN TAHMIN EDILEN spread ile degistirmek.

Yontem: Corwin & Schultz (2012, Journal of Finance) high-low spread
estimator. Mantik: gunluk high-low orani hem volatiliteyi hem spread'i
yansitir. Volatilite zamanla sqrt(t) ile buyur, spread buyumez -- bu
farkli olceklenmeyi kullanarak 1-gunluk ve 2-gunluk high-low oranlarindan
spread'i ayristirir.

Formul (Corwin-Schultz 2012):
    beta  = E[ (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2 ]   (ardisik 2 gun ortalamasi)
    gamma = (ln(H_2/L_2))^2    (2-gunluk max-high/min-low araligi)
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3-2*sqrt(2)))
    S     = 2*(exp(alpha) - 1) / (1 + exp(alpha))
    S     = max(S, 0)   (negatif spread anlamsiz, 0'a kirpilir)

Bu, sembol+ay bazinda ortalama spread tahmini uretir (literaturde aylik
agregasyon yaygin, gunluk tahminler gurultulu olabilir). Sonra bu spread
tahmini COST_BPS yerine kullanilarak net kar yeniden hesaplanir.

Kullanim:
    python check_net_returns_cs.py
Cikti:
    net_returns_cs_report.txt
    cs_spread_by_symbol.csv (sembol bazinda ortalama tahmini spread, referans icin)
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

K = 3 - 2 * np.sqrt(2)  # Corwin-Schultz sabiti

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_TXT = os.path.join(OUTPUT_DIR, "net_returns_cs_report.txt")
SPREAD_CSV = os.path.join(OUTPUT_DIR, "cs_spread_by_symbol.csv")


def load_symbol(path):
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if len(df) < MIN_ROWS:
            return None
        if not {"Date", "Open", "High", "Low", "Close"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def corwin_schultz_spread(high, low):
    """
    high, low: pandas Series (Date index), tek bir sembol icin.
    Doner: gunluk tahmini spread serisi (oran olarak, 0.01 = %1).
    Her deger, o gunu iceren 2-gunluk pencereden hesaplanan spread tahminidir.
    """
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)

    # tek gunluk log(H/L)^2
    hl1 = (np.log(high / low)) ** 2

    # iki gunluk: pencere icindeki max(High) / min(Low)
    high2 = high.rolling(2).max()
    low2 = low.rolling(2).min()
    hl2 = (np.log(high2 / low2)) ** 2

    beta = hl1 + hl1.shift(1)  # ardisik 2 gunun tek-gunluk terimlerinin toplami
    gamma = hl2

    sqrt_beta = np.sqrt(beta.clip(lower=0))
    sqrt_gamma = np.sqrt(gamma.clip(lower=0))

    alpha = (np.sqrt(2) - 1) * sqrt_beta / K - sqrt_gamma / K

    with np.errstate(over="ignore", invalid="ignore"):
        exp_alpha = np.exp(alpha)
        spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)

    spread = spread.clip(lower=0, upper=0.5)  # %50 ustu anlamsiz, veri hatasi sayilir
    return spread


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

    print(f"Loaded {len(symbols)} symbols (High/Low kolonu olan)")

    all_dates = pd.to_datetime(
        sorted(set().union(*[df["Date"].unique() for df in symbols.values()]))
    )

    close_series, open_series, vol_series, spread_series = {}, {}, {}, {}
    for name, df in symbols.items():
        s = df.set_index("Date")
        close_series[name] = s["Close"].reindex(all_dates)
        open_series[name] = s["Open"].reindex(all_dates)
        vol_series[name] = s["Volume"].reindex(all_dates) if "Volume" in df.columns else pd.Series(np.nan, index=all_dates)
        cs_spread = corwin_schultz_spread(s["High"], s["Low"])
        spread_series[name] = cs_spread.reindex(all_dates)

    close_panel = pd.concat(close_series, axis=1)
    open_panel = pd.concat(open_series, axis=1)
    vol_panel = pd.concat(vol_series, axis=1)
    spread_panel = pd.concat(spread_series, axis=1)
    for p in (close_panel, open_panel, vol_panel, spread_panel):
        p.index.name = None

    ret_panel = close_panel.pct_change()
    market_ret = ret_panel.mean(axis=1, skipna=True).fillna(0)

    # sembol bazinda ortalama tahmini spread (referans/gorsel kontrol icin)
    avg_spread_by_symbol = spread_panel.median(axis=0, skipna=True).dropna().sort_values(ascending=False)
    avg_spread_by_symbol.to_frame("median_spread_estimate").to_csv(SPREAD_CSV)
    print(f"Sembol bazinda medyan spread tahmini: {SPREAD_CSV}'e kaydedildi")
    print(f"Ornek (en yuksek 5 spread): \n{avg_spread_by_symbol.head(5)}")
    print(f"Ornek (en dusuk 5 spread): \n{avg_spread_by_symbol.tail(5)}")

    dollar_vol = close_panel * vol_panel
    liquidity = dollar_vol.median(axis=0, skipna=True).dropna()
    tertiles = pd.qcut(liquidity, 3, labels=["illiquid", "mid", "liquid"]) if len(liquidity) >= 3 else pd.Series(dtype="object")

    def get_bucket(symbol):
        if symbol in tertiles.index and pd.notna(tertiles.get(symbol)):
            return tertiles[symbol]
        return "mid"

    # extreme_down events
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
    print(f"\nTotal extreme_down events: {len(df_events)}")

    report = []
    report.append(f"Total extreme_down_reversal events: {len(df_events)}")
    report.append("")
    report.append("=== Corwin-Schultz spread tahmini ile genel istatistik ===")
    report.append(f"Universe medyan spread tahmini (round-trip icin x2): "
                   f"{avg_spread_by_symbol.median()*10000:.1f} bps (tek yon), "
                   f"{avg_spread_by_symbol.median()*20000:.1f} bps (round-trip)")
    report.append("")
    for bucket in ["illiquid", "mid", "liquid"]:
        syms_in_bucket = [s for s in avg_spread_by_symbol.index if get_bucket(s) == bucket]
        if not syms_in_bucket:
            continue
        med = avg_spread_by_symbol.loc[syms_in_bucket].median()
        report.append(f"  {bucket:8s}: medyan tek-yon spread={med*10000:.1f} bps, "
                       f"round-trip={med*20000:.1f} bps, n_sembol={len(syms_in_bucket)}")
    report.append("")
    report.append("(NOT: Bu tahmini, senin onceki sabit varsayimlarinla (illiquid=150,")
    report.append(" mid=60, liquid=25bps round-trip) karsilastir. Corwin-Schultz genelde")
    report.append(" sadece SPREAD'i yakalar, market impact/slippage'i yakalamaz -- yani")
    report.append(" gercek toplam maliyet CS tahmininden biraz yuksek olabilir, ozellikle")
    report.append(" buyuk pozisyon boyutlarinda.)")
    report.append("")

    def forward_net_profit(events_df, horizon):
        rows = []
        for name, d in zip(events_df["symbol"], events_df["date"]):
            if name not in ret_panel.columns or name not in open_panel.columns:
                continue
            idx = ret_panel.index.get_indexer([d])[0]
            if idx < 0 or idx + horizon >= len(ret_panel.index):
                continue
            entry_date = ret_panel.index[idx + 1]
            exit_date = ret_panel.index[idx + horizon]
            entry_price = open_panel.loc[entry_date, name]
            exit_price = close_panel.loc[exit_date, name]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue
            stock_fwd = exit_price / entry_price - 1
            fwd_dates = ret_panel.index[idx + 1: idx + 1 + horizon]
            mkt_fwd = (1 + market_ret.loc[fwd_dates]).prod() - 1
            gross_excess = stock_fwd - mkt_fwd

            bucket = get_bucket(name)
            if bucket == "illiquid":
                continue  # onceki testte illiquid zaten elendi

            # o olay tarihindeki (veya en yakin onceki) CS spread tahminini kullan
            sp = spread_panel.loc[:d, name].dropna()
            if sp.empty:
                continue
            spread_estimate = sp.iloc[-1]
            round_trip_cost = spread_estimate  # spread orani zaten round-trip'e yakin
            # (Corwin-Schultz S degeri, alis-satis farkinin fiyata oranidir --
            # bir kere alip bir kere satarken toplamda bu kadar odenir, yani
            # dogrudan round-trip maliyet olarak kullanilabilir)

            net_profit = -gross_excess - round_trip_cost
            rows.append({"symbol": name, "date": d, "gross": gross_excess,
                         "net": net_profit, "liquidity": bucket, "cost": round_trip_cost})
        return pd.DataFrame(rows)

    report.append("=== Net kar (Corwin-Schultz spread tahmini ile, mid+liquid) ===")
    for horizon in FWD_HORIZONS:
        df_res = forward_net_profit(df_events, horizon)
        if df_res.empty:
            report.append(f"--- Horizon {horizon}d: veri yok ---")
            continue
        report.append(f"--- Horizon {horizon}d (n={len(df_res)}) ---")
        report.append(f"  Ortalama tahmini maliyet: {df_res['cost'].mean()*10000:.1f} bps")
        report.append(f"  BRUT excess (long yon)   : {df_res['gross'].mean()*10000:.1f} bps")
        report.append(f"  NET KAR (kisa, CS maliyet): {df_res['net'].mean()*10000:.1f} bps")
        report.append(f"  NET KAR pozitif oran      : {(df_res['net']>0).mean()*100:.1f}%")
        for bucket in ["mid", "liquid"]:
            sub = df_res[df_res["liquidity"] == bucket]
            if len(sub) == 0:
                continue
            report.append(f"    {bucket:8s}: net_kar={sub['net'].mean()*10000:.1f} bps, "
                           f"maliyet={sub['cost'].mean()*10000:.1f} bps, n={len(sub)}, "
                           f"kazanma={100*(sub['net']>0).mean():.1f}%")
        report.append("")

    report.append("YORUM: Bu sonucu check_net_returns.py'deki sabit-varsayim sonucuyla")
    report.append("karsilastir. Eger CS-tabanli net kar da pozitif kaliyorsa, edge'in")
    report.append("varligina olan guven artar (iki farkli maliyet yontemi ayni yone isaret")
    report.append("ediyor). Eger CS spread'i sabit varsayimlardan cok daha yuksek cikip")
    report.append("net kari negatife ceviriyorsa, once daha guvenilir sabit varsayimlar")
    report.append("YANLIS DUSUK'tu demektir -- gercek maliyet daha yuksek.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nRapor kaydedildi: {REPORT_TXT}")


if __name__ == "__main__":
    main()
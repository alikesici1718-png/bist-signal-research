"""
check_data_quality_borsapy.py

data/ klasorundeki CSV'lerden rastgele 20-30 sembol secip su kontrolleri yapar:
  1. Tarihler eksiksiz mi (buyuk bosluk var mi - haftasonu/tatil disi)
  2. OHLC mantikli mi (Low <= Open/Close <= High, Low <= High)
  3. Volume negatif degil mi
  4. Ayni tarih iki kez var mi (duplicate)
  5. Buyuk kurumsal islem (borsapy .actions) etrafinda fiyat serisi
     mantikli (asiri sicrama yok) mi

Kullanim:
    python check_data_quality_borsapy.py
Cikti:
    quality_report_borsapy.txt
"""

import os
import glob
import random
import pandas as pd
import borsapy as bp

DATA_DIR = "data"
SAMPLE_SIZE = 25
MAX_GAP_DAYS = 10  # bu kadar gunden uzun bosluklari raporla (uzun tatil/OHAL vb olabilir, sadece bilgi amacli)
BIG_MOVE_THRESHOLD = 0.30  # tek gunde %30+ hareket varsa isaretle

random.seed(42)


def check_symbol(symbol, path):
    issues = []
    df = pd.read_csv(path, parse_dates=["Date"])

    if df.empty:
        return [f"BOS DOSYA"]

    # 1. Duplicate tarih kontrolu
    dup_count = df["Date"].duplicated().sum()
    if dup_count > 0:
        issues.append(f"DUPLICATE TARIH: {dup_count} adet")

    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Tarih bosluklari (is gunu bazinda kaba kontrol)
    gaps = df["Date"].diff().dt.days
    big_gaps = gaps[gaps > MAX_GAP_DAYS]
    if len(big_gaps) > 0:
        issues.append(f"BUYUK BOSLUK: {len(big_gaps)} adet ({MAX_GAP_DAYS}+ gun), "
                       f"en buyugu {int(big_gaps.max())} gun")

    # 3. OHLC mantik kontrolu
    bad_ohlc = df[
        (df["Low"] > df["High"]) |
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"])
    ]
    if len(bad_ohlc) > 0:
        issues.append(f"BOZUK OHLC: {len(bad_ohlc)} satir "
                       f"(ornek tarih: {bad_ohlc.iloc[0]['Date'].date()})")

    # 4. Negatif volume
    neg_vol = (df["Volume"] < 0).sum()
    if neg_vol > 0:
        issues.append(f"NEGATIF VOLUME: {neg_vol} satir")

    # 5. Tek gunde asiri hareket (kaba anomali taramasi)
    ret = df["Close"].pct_change().abs()
    big_moves = df[ret > BIG_MOVE_THRESHOLD]
    if len(big_moves) > 0:
        dates_str = ", ".join(str(d.date()) for d in big_moves["Date"].head(5))
        issues.append(f"ASIRI GUNLUK HAREKET (>{int(BIG_MOVE_THRESHOLD*100)}%): "
                       f"{len(big_moves)} adet, ilk birkaci: {dates_str}")

    return issues


def check_against_actions(symbol, issues_found):
    """Eger asiri hareket bulunduysa, borsapy'nin kurumsal islem kaydiyla eslesip
    eslesmedigine bakar - eslesirse bu beklenen bir olay demektir, sinyal degil."""
    try:
        ticker = bp.Ticker(symbol)
        actions = ticker.actions
        if actions is None or actions.empty:
            return "kurumsal islem kaydi yok/bos"
        action_dates = set(pd.to_datetime(actions.index).date)
        return f"kurumsal islem tarihleri: {sorted(action_dates)}"
    except Exception as e:
        return f"actions cekilemedi: {e}"


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print(f"HATA: {DATA_DIR} klasorunde CSV bulunamadi")
        return

    sample = random.sample(files, min(SAMPLE_SIZE, len(files)))
    report_lines = [f"Kalite kontrolu: {len(sample)} sembol (toplam {len(files)} icinden rastgele secildi)\n"]

    clean_count = 0
    for path in sample:
        symbol = os.path.splitext(os.path.basename(path))[0]
        issues = check_symbol(symbol, path)

        if not issues:
            clean_count += 1
            report_lines.append(f"{symbol}: TEMIZ")
        else:
            report_lines.append(f"{symbol}: {len(issues)} sorun")
            for issue in issues:
                report_lines.append(f"    - {issue}")
                if "ASIRI GUNLUK HAREKET" in issue:
                    action_info = check_against_actions(symbol, issues)
                    report_lines.append(f"      -> {action_info}")

    report_lines.append(f"\nOZET: {clean_count}/{len(sample)} sembol tamamen temiz")

    report = "\n".join(report_lines)
    print(report)
    with open("quality_report_borsapy.txt", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
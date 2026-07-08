"""
btc_wallet_clustering_pilot.py

Common-input-ownership heuristic pilot:
Bilinen bir BTC adresiyle ayni islemde input olarak gorunen adresler
buyuk olasilikla ayni cüzdana aittir.

Veri kaynagi: blockchain.info public API (api-key gerektirmez)

Guncelleme: hot wallet adaylarini sirali dene, INPUT tx'i en fazla
olani kullan. 200 tx tarama.
"""

import time
import urllib.request
import urllib.error
import json

RAWADDR_URL = "https://blockchain.info/rawaddr/{addr}?limit={limit}"
TX_LIMIT = 200
REQUEST_DELAY = 1.2  # blockchain.info rate-limit icin

# Adaylar: (adres, aciklama)
# 1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s -> Binance hot wallet (yuksek tx hacmi, 2-yonlu)
# 3FrSzikNqBgikWgTHixywhXcx57q6H6rHC -> gorevde belirtilen 1. aday
# 3KnZmJohDM8tmmkwUax9JHXpaQPK28Ja8s -> gorevde belirtilen 2. aday
CANDIDATES = [
    ("1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s", "Binance Hot Wallet (bilinen, yuksek hacim)"),
    ("3FrSzikNqBgikWgTHixywhXcx57q6H6rHC", "Binance transfer zinciri aday 1"),
    ("3KnZmJohDM8tmmkwUax9JHXpaQPK28Ja8s", "Binance transfer zinciri aday 2"),
]

KNOWN_LABELS = {
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "Binance Cold Wallet (bilinen)",
    "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s": "Binance Hot Wallet (bilinen)",
    "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb": "Binance (bilinen)",
    "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h": "Binance (bilinen)",
    "3FrSzikNqBgikWgTHixywhXcx57q6H6rHC": "Binance transfer zinciri aday 1",
    "3KnZmJohDM8tmmkwUax9JHXpaQPK28Ja8s": "Binance transfer zinciri aday 2",
}


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-wallet-clustering-pilot/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def count_input_txs(txs, addr):
    count = 0
    for tx in txs:
        input_addrs = {i.get("prev_out", {}).get("addr") for i in tx.get("inputs", [])}
        if addr in input_addrs:
            count += 1
    return count


def run_clustering(known_address, label, txs):
    discovered = {}
    input_tx_count = 0
    for tx in txs:
        txid = tx.get("hash", "?")
        input_addrs = set()
        for inp in tx.get("inputs", []):
            a = inp.get("prev_out", {}).get("addr")
            if a:
                input_addrs.add(a)
        if known_address not in input_addrs:
            continue
        input_tx_count += 1
        for addr in input_addrs - {known_address}:
            discovered.setdefault(addr, []).append(txid)
    return input_tx_count, discovered


def main():
    print("=" * 65)
    print("BTC Wallet Clustering Pilot — Hot Wallet Taramasi")
    print(f"Taranacak tx limit : {TX_LIMIT}")
    print("=" * 65)

    # Her adayi dene, INPUT tx sayisina gore sec
    best_addr = None
    best_label = None
    best_txs = None
    best_input_count = -1

    for addr, desc in CANDIDATES:
        url = RAWADDR_URL.format(addr=addr, limit=TX_LIMIT)
        print(f"\n[Aday] {addr}  ({desc})")
        print(f"  Cekiliyor: {url}")
        try:
            data = fetch_json(url)
        except urllib.error.HTTPError as e:
            print(f"  HTTP HATA: {e.code} {e.reason} — bu aday atlanıyor")
            time.sleep(REQUEST_DELAY)
            continue
        except urllib.error.URLError as e:
            print(f"  URL HATA: {e.reason} — bu aday atlaniyor")
            time.sleep(REQUEST_DELAY)
            continue

        txs = data.get("txs", [])
        n_total = data.get("n_tx", "?")
        n_input = count_input_txs(txs, addr)
        print(f"  Toplam tx (adres gecmisi): {n_total} | Cekilen: {len(txs)} | INPUT olarak: {n_input}")

        if n_input > best_input_count:
            best_input_count = n_input
            best_addr = addr
            best_label = desc
            best_txs = txs

        time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 65)
    print(f"SECILEN ADRES: {best_addr}")
    print(f"ACIKLAMA     : {best_label}")
    print(f"INPUT TX     : {best_input_count} / {len(best_txs)} incelenen")
    print("=" * 65)

    if best_input_count == 0:
        print("\nSonuc: Hic INPUT tx bulunamadi.")
        print("Bu adresler son 200 islemde yalnizca output (para alma) islemi yapmis.")
        print("Common-input heuristic uygulanamaz — soğuk cüzdan davranışı.")
        return

    # Clustering
    input_tx_count, discovered = run_clustering(best_addr, best_label, best_txs)

    print(f"\n[CLUSTERING SONUCU]")
    print(f"  INPUT oldugu tx sayisi   : {input_tx_count}")
    print(f"  Kesfedilen yeni adres    : {len(discovered)}")

    if not discovered:
        print("\n  Bu INPUT tx'lerin hepsinde tek girdi varmis (co-input yok).")
        print("  Heuristic calisti ama kume genislemedi.")
        return

    print(f"\n  {'Adres':<45} {'Kac tx':<8} {'Etiket'}")
    print("  " + "-" * 80)
    for addr, txids in sorted(discovered.items(), key=lambda x: -len(x[1])):
        lbl = KNOWN_LABELS.get(addr, "--- (etiketsiz, potansiyel yeni kesif)")
        print(f"  {addr:<45} {len(txids):<8} {lbl}")

    print(f"\n[ORNEK DETAYLAR — ilk 5 adres]")
    for addr in list(discovered.keys())[:5]:
        lbl = KNOWN_LABELS.get(addr)
        status = f"BILINEN ETIKET: {lbl}" if lbl else "Yeni kesif (etiketsiz)"
        txids_short = [t[:14] + "..." for t in discovered[addr][:3]]
        print(f"\n  Adres : {addr}")
        print(f"  Durum : {status}")
        print(f"  Tx    : {', '.join(txids_short)}")

    print("\n" + "=" * 65)
    print("Pilot tamamlandi. (1-hop, 200-tx tarama)")
    print("=" * 65)


if __name__ == "__main__":
    main()

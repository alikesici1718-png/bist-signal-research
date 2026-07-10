"""Debug script: fire 7 sequential pykap disclosure requests for THYAO
to check for rate limiting / intermittent failures. No analysis output.
"""
import pykap
import time
import traceback

for i in range(1, 8):
    print(f"=== İstek {i} ===")
    try:
        comp = pykap.BISTCompany('THYAO')
        r = comp.get_historical_disclosure_list(
            fromdate="2023-01-01",
            todate="2024-01-01",
            disclosure_type="FR",
            subject="4028328c594bfdca01594c0af9aa0057"
        )
        print(f"Basarili: {len(r)} sonuc")
    except Exception as e:
        print(f"HATA: {e}")
    time.sleep(1.0)

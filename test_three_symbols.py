import pykap
import traceback

for ticker in ['THYAO', 'AKBNK', 'ASELS']:
    print(f"=== {ticker} ===")
    try:
        comp = pykap.BISTCompany(ticker)
        print(f"company_id: {comp.company_id}")
        r = comp.get_historical_disclosure_list(
            fromdate="2020-01-01",
            todate="2026-07-06",
            disclosure_type="FR",
            subject="4028328c594bfdca01594c0af9aa0057"
        )
        print(f"Basarili: {len(r)} sonuc")
    except Exception as e:
        print(f"HATA: {e}")
        traceback.print_exc()
    print()

import pykap
import traceback

print("=== ADIM 1: get_bist_companies'ten THYAO satırı ===")
df = pykap.get_bist_companies()
row = df[df['ticker'] == 'THYAO']
print(row)
print()

print("=== ADIM 2: BISTCompany('THYAO') init ===")
try:
    comp = pykap.BISTCompany('THYAO')
    print(f"comp.company_id = {comp.company_id}")
    print(f"comp.ticker = {comp.ticker}")
    print(f"comp.name = {comp.name}")
except Exception as e:
    print("INIT HATASI:")
    traceback.print_exc()
print()

print("=== ADIM 3: get_historical_disclosure_list çağrısı, manuel try/except ile tam traceback ===")
try:
    result = comp.get_historical_disclosure_list(
        fromdate="2023-01-01",
        todate="2024-01-01",
        disclosure_type="FR",
        subject="4028328c594bfdca01594c0af9aa0057"
    )
    print(f"Basarili, {len(result)} sonuc")
except Exception as e:
    print("CAGRI HATASI - TAM TRACEBACK:")
    traceback.print_exc()

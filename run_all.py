"""Single entry point for the BIST signal research analyses.

Usage:
    python run_all.py --list                          # show available analyses
    python run_all.py --analysis liquidity_premium    # run one analysis
    python run_all.py --all                           # run every analysis in order

Each analysis is executed from its own script via runpy (identical to
``python <script>.py``), so there is no duplicated logic here and results
are byte-for-byte the same as running the scripts directly. Descriptions
shown by --list are extracted from each script's module docstring, which
is the single source of truth.

Data-acquisition scripts (get_*, fetch_*) and one-off KAP debug scripts
are intentionally NOT registered here — they need network access / API
keys and are documented in the README instead.
"""
import argparse
import ast
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name -> script, in a sensible execution order (diagnostics after scans).
ANALYSES = {
    # Data quality
    "data_quality":            "check_data_quality.py",
    "limit_moves":             "check_limit_moves.py",
    # Signal scans
    "comprehensive_scan":      "comprehensive_scan.py",
    "liquidity_filtered_scan": "liquidity_filtered_scan.py",
    "illiquid_segment_scan":   "illiquid_segment_scan.py",
    "vol_compression_v2":      "vol_compression_breakout_v2.py",
    "laggard_catchup":         "laggard_catchup_scan.py",
    "breakout_52w":            "test_52w_breakout_v2.py",
    "breakout_multi_lookback": "test_multi_lookback_breakout.py",
    "cointegration_scan":      "cointegration_scan.py",
    "dispersion_basket":       "dispersion_basket_backtest.py",
    # Net returns / costs / capacity
    "net_returns":             "check_net_returns.py",
    "net_returns_cs":          "check_net_returns_cs.py",
    "net_returns_proxy":       "check_net_returns_simple_proxy.py",
    "capacity":                "capacity_backtest.py",
    "entry_timing_audit":      "check_entry_timing.py",
    # Diagnostics
    "diagnose_signals":        "diagnose_signals.py",
    "signal_overlap":          "signal_overlap_check.py",
    "event_concentration":     "event_concentration_check.py",
    "event_freq_drift":        "event_frequency_and_post_drift.py",
    "overlap_clustering":      "check_overlap_and_clustering.py",
    # Walk-forward
    "walkforward":             "walkforward_multi_signal.py",
    # Regime / macro
    "regime_backtest":         "regime_backtest.py",
    "regime_regression":       "extreme_down_regime_regression.py",
    "usdtry_analysis":         "usdtry_bist_analysis.py",
    "fx_shock_dates":          "fx_shock_dates_check.py",
    # PEAD / earnings
    "pead_backtest":           "pead_signal_backtest.py",
    "pead_placebo":            "pead_placebo_test.py",
    "pead_excess":             "pead_signal_excess.py",
    "pead_placebo_excess":     "pead_placebo_excess.py",
    "pead_surprise_pilot":     "pead_earnings_surprise_pilot.py",
    # Pilots / newer hypotheses
    "index_inclusion":         "index_inclusion_pilot.py",
    "post_ipo_neglect":        "post_ipo_neglect_test.py",
    "liquidity_premium":       "liquidity_premium_test.py",
    "info_diffusion":          "information_diffusion_speed_test.py",
    "capital_increase":        "capital_increase_pilot.py",
    "insider_trading":         "kap_insider_trading_pilot.py",
    "btc_wallet_clustering":   "btc_wallet_clustering_pilot.py",
}


def one_line_description(script: str) -> str:
    """First line of the script's module docstring, or a placeholder."""
    try:
        source = (ROOT / script).read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(source))
        for line in (doc or "").splitlines():
            line = line.strip()
            # skip blank lines and lines that just repeat the filename
            if line and not line.lower().endswith(".py"):
                return line
        # fall back to the first leading comment line
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("# ").strip()
            if stripped and not stripped.startswith("#"):
                break
    except (OSError, SyntaxError):
        pass
    return "(no description)"


def run_analysis(name: str) -> bool:
    """Execute one registered analysis exactly as `python <script>.py` would."""
    script = ROOT / ANALYSES[name]
    print(f"\n{'=' * 70}\nRunning {name}  ({script.name})\n{'=' * 70}")
    start = time.time()
    try:
        runpy.run_path(str(script), run_name="__main__")
        print(f"-- {name} finished in {time.time() - start:.1f}s")
        return True
    except SystemExit as e:
        ok = e.code in (None, 0)
        print(f"-- {name} exited with code {e.code} in {time.time() - start:.1f}s")
        return ok
    except Exception as e:  # keep going on --all; report at the end
        print(f"-- {name} FAILED after {time.time() - start:.1f}s: {type(e).__name__}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BIST signal research analyses.",
        epilog="Example: python run_all.py --analysis liquidity_premium",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="list available analyses with one-line descriptions")
    group.add_argument("--analysis", metavar="NAME", choices=sorted(ANALYSES),
                       help="run a single analysis by name (see --list)")
    group.add_argument("--all", action="store_true",
                       help="run every registered analysis in order")
    args = parser.parse_args()

    if args.list:
        width = max(map(len, ANALYSES))
        out = sys.stdout
        for name, script in ANALYSES.items():
            line = f"{name:<{width}}  {one_line_description(script)}"
            # legacy Windows console codepages can't render every character
            out.write(line.encode(out.encoding or "utf-8", errors="replace")
                          .decode(out.encoding or "utf-8", errors="replace") + "\n")
        return 0

    if args.analysis:
        return 0 if run_analysis(args.analysis) else 1

    results = {name: run_analysis(name) for name in ANALYSES}
    failed = [name for name, ok in results.items() if not ok]
    print(f"\n{'=' * 70}\nDone: {len(results) - len(failed)}/{len(results)} analyses succeeded.")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

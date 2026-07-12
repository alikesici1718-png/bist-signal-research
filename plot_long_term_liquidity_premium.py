"""
Two-panel chart: long-term liquidity premium (Q1 vs Q4 medyan excess return, 1m-12m)
with placebo comparison inset.
Values sourced from long_term_diffusion_cleaned.py output.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- Verified values from long_term_diffusion_cleaned.py ---
windows = ["1m", "3m", "6m", "12m"]
q1_medians = [-47.60, -52.14, 157.87, 1080.62]
q4_medians = [-126.36, -342.04, -603.72, -1151.66]

# Cost scenarios applied to 12m Q1 median
q1_12m_gross = 1080.62
net_midas    = q1_12m_gross - 82.5
net_ata      = q1_12m_gross - 227.0

# Placebo values (from placebo run: random ticker split)
placebo_q1 = -165.7
placebo_q4 = -342.7

x = np.arange(len(windows))
width = 0.35

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[2.8, 1.4, 1.4], wspace=0.38)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ---- Panel 1: Q1 vs Q4 across horizons ----
bars1 = ax1.bar(x - width/2, q1_medians, width, label="Q1 (Low Liquidity)", color="#2e75b6", edgecolor="white")
bars2 = ax1.bar(x + width/2, q4_medians, width, label="Q4 (High Liquidity)", color="#c0392b", edgecolor="white")
ax1.axhline(0, color="black", linewidth=1.0)
ax1.set_xticks(x); ax1.set_xticklabels(windows, fontsize=11)
ax1.set_ylabel("Medyan Excess Return (bps)", fontsize=10)
ax1.set_title("Long-Term Liquidity Premium\nQ1 vs Q4 Medyan (XU100-Adjusted)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
for bar, val in zip(list(bars1) + list(bars2), q1_medians + q4_medians):
    offset = 20 if val >= 0 else -40
    va = "bottom" if val >= 0 else "top"
    ax1.text(bar.get_x() + bar.get_width()/2, val + offset,
             f"{val:+.0f}", ha="center", va=va, fontsize=8, fontweight="bold")
ax1.text(0.5, -0.16,
         "BH-FDR düzeltmeli: 1m q=0.012, 3m/6m/12m q<0.001   n≈4,000–4,300 olay/kartil",
         ha="center", va="top", transform=ax1.transAxes, fontsize=8, color="#555555", style="italic")

# ---- Panel 2: Net return after costs (12m Q1 only) ----
labels2 = ["Gross\n(Medyan)", "Net:\nMidas\n+82.5 bps", "Net:\nAtaYatırım\n+227 bps"]
vals2   = [q1_12m_gross, net_midas, net_ata]
colors2 = ["#2e75b6", "#27ae60", "#27ae60"]
bars3 = ax2.bar(labels2, vals2, color=colors2, width=0.5, edgecolor="white")
ax2.axhline(0, color="black", linewidth=1.0)
ax2.set_title("Q1 12m — Net After Costs", fontsize=10, fontweight="bold")
ax2.set_ylabel("bps", fontsize=10)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
for bar, val in zip(bars3, vals2):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 15,
             f"+{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_ylim(0, 1300)

# ---- Panel 3: Placebo comparison (12m) ----
placebo_labels = ["Real Q1", "Real Q4", "Placebo\npQ1", "Placebo\npQ4"]
placebo_vals   = [q1_12m_gross, -1151.66, placebo_q1, placebo_q4]
placebo_colors = ["#2e75b6", "#c0392b", "#7fb0d5", "#e07b7b"]
bars4 = ax3.bar(placebo_labels, placebo_vals, color=placebo_colors, width=0.5, edgecolor="white")
ax3.axhline(0, color="black", linewidth=1.0)
ax3.set_title("Placebo Kontrolü (12m)\nRastgele Bölme vs Gerçek", fontsize=10, fontweight="bold")
ax3.set_ylabel("Medyan Excess (bps)", fontsize=10)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
for bar, val in zip(bars4, placebo_vals):
    offset = 20 if val >= 0 else -40
    va = "bottom" if val >= 0 else "top"
    ax3.text(bar.get_x() + bar.get_width()/2, val + offset,
             f"{val:+.0f}", ha="center", va=va, fontsize=8.5, fontweight="bold")
ax3.text(0.5, -0.16,
         "Gerçek: p<0.001   Placebo: p=0.269",
         ha="center", va="top", transform=ax3.transAxes, fontsize=8, color="#555555", style="italic")

fig.suptitle("BIST Long-Term Liquidity Premium: Real Signal, Survives Costs, Likely Risk Premium",
             fontsize=12, fontweight="bold", y=1.02)
fig.text(0.99, -0.04,
         "Source: long_term_diffusion_cleaned.py | KAP events n=17,205 | Mann-Whitney U, BH-FDR corrected",
         ha="right", va="bottom", fontsize=7.5, color="gray")

plt.tight_layout()
out = "visualizations/long_term_liquidity_premium.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Kaydedildi: {out}")

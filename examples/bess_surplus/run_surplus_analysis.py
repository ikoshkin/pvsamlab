"""
run_surplus_analysis.py

Headless runner for bess_surplus_optimization.ipynb:
  1. Simulates all 75 cases (25 sizes × 3 charging modes) in parallel
  2. Saves results CSV
  3. Generates 3 static PNGs:
       surplus_irr_heatmap.png
       surplus_revenue_waterfall.png
       surplus_price_dispatch.png
"""
from __future__ import annotations
import sys, pathlib, warnings
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid", font_scale=0.9)

# ── Config ────────────────────────────────────────────────────────────────────
SITE_LAT        = 33.0278
SITE_LON        = -100.0814
MET_YEAR        = "2017"
PV_TARGET_KWAC  = 300_000
PV_DCAC         = 1.35
POI_LIMIT_KW    = 300_000
LOAD_KW         = 200_000

POWER_MW_LIST    = [50, 75, 100, 125, 150]
DURATION_HR_LIST = [1,  2,   4,   6,   8]

CHARGING_MODES = {
    "solar_only":   {"can_gridcharge": 0, "can_solar_charge": 1, "label": "Solar charge only"},
    "grid_only":    {"can_gridcharge": 1, "can_solar_charge": 0, "label": "Grid charge only"},
    "unrestricted": {"can_gridcharge": 1, "can_solar_charge": 1, "label": "Solar + grid charge"},
}

CAPACITY_PAYMENT_KW_YR = 80.0
ANCILLARY_KW_YR        = 15.0
BATT_CAPEX_KWH         = 250.0
BATT_CAPEX_KW          = 150.0
OPEX_KWH_YR            = 8.0
DISCOUNT_RATE          = 8.0
NUM_WORKERS            = 4

BATT_KWARGS = dict(
    chemistry="LFP", capex_per_kwh=BATT_CAPEX_KWH, capex_per_kw=BATT_CAPEX_KW,
    opex_per_kwh_year=OPEX_KWH_YR, soc_min=10.0, soc_max=95.0,
    calendar_degradation=2.0,
)
FINANCIAL_KWARGS = dict(
    analysis_period=25, discount_rate=DISCOUNT_RATE,
    degradation_rate=0.5, ppa_rate=40.0, ppa_escalation=1.0,
    inflation_rate=2.5, debt_fraction=70.0, loan_rate=5.0, loan_term=18,
    federal_tax_rate=21.0, state_tax_rate=0.0, itc_rate=30.0,
)

PRICE_CSV   = _HERE.parent.parent / "pvsamlab" / "data" / "DAMPriceExample.csv"
RESULTS_CSV = _HERE / "outputs" / "bess_surplus_optimization_results.csv"

DT_IDX = pd.date_range("2017-01-01", periods=8760, freq="h")


# ── Load prices ───────────────────────────────────────────────────────────────
def load_prices() -> list:
    df = pd.read_csv(PRICE_CSV, header=None)
    return df.iloc[:, 0].tolist()[:8760]


# ── Build task list ───────────────────────────────────────────────────────────
def build_tasks(hourly_prices: list) -> list:
    tasks = []
    for mode, mc in CHARGING_MODES.items():
        for power_mw in POWER_MW_LIST:
            for dur in DURATION_HR_LIST:
                tasks.append({
                    "power_kw":    power_mw * 1000,
                    "duration_hr": dur,
                    "mode":        mode,
                    "hourly_prices": hourly_prices,
                    "poi_limit_kw":  POI_LIMIT_KW,
                    "load_kw":       LOAD_KW,
                    "can_gridcharge":  mc["can_gridcharge"],
                    "can_solar_charge": mc["can_solar_charge"],
                    "lat": SITE_LAT, "lon": SITE_LON,
                    "pv_target_kwac": PV_TARGET_KWAC,
                    "pv_dcac": PV_DCAC,
                    "met_year": MET_YEAR,
                    "batt_kwargs": BATT_KWARGS,
                    "capacity_payment_kw_yr": CAPACITY_PAYMENT_KW_YR,
                    "ancillary_kw_yr": ANCILLARY_KW_YR,
                    "financial_kwargs": FINANCIAL_KWARGS,
                })
    return tasks


# ── Run sweep ─────────────────────────────────────────────────────────────────
def run_sweep(tasks: list) -> tuple[pd.DataFrame, dict]:
    from bess_surplus_worker import run_surplus_case

    rows       = []
    hourly_data = {}
    errors     = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(run_surplus_case, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Surplus cases"):
            r = fut.result()
            if r["status"] == "error":
                errors.append(r)
                print(f"\n  ERROR {r['power_mw']}MW/{r['duration_hr']}hr "
                      f"{r['charging_mode']}: {r.get('error_msg','')}")
                continue
            key = (r["power_mw"], r["duration_hr"], r["charging_mode"])
            hourly_data[key] = {
                "gen_mw":    np.array(r.pop("_gen_mw")),
                "batt_mw":   np.array(r.pop("_batt_mw")),
                "soc_pct":   np.array(r.pop("_soc_pct")),
                "export_mw": np.array(r.pop("_export_mw")),
            }
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"\nCompleted {len(rows)}/75, errors: {len(errors)}")
    return df, hourly_data


# ── Chart 1: IRR heatmap 4-panel ──────────────────────────────────────────────
def chart1_irr_heatmap(df: pd.DataFrame) -> None:
    modes = list(CHARGING_MODES.keys())
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("IRR Heatmap by BESS Size and Charging Mode\n"
                 f"300 MWac PV | {POI_LIMIT_KW/1000:.0f} MW POI | "
                 f"DAM merchant prices + ${CAPACITY_PAYMENT_KW_YR}/kW-yr capacity",
                 fontsize=10)

    irr_matrices = {}
    vmin = df["irr_pct"].min()
    vmax = df["irr_pct"].max()

    for col, mode in enumerate(modes):
        ax  = axes[col]
        sub = df[df["charging_mode"] == mode]
        mat = sub.pivot(index="duration_hr", columns="power_mw", values="irr_pct")
        mat = mat.loc[sorted(mat.index, reverse=True)]   # 8hr at top
        irr_matrices[mode] = mat

        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{int(c)}" for c in mat.columns], fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{int(d)} hr" for d in mat.index], fontsize=8)
        ax.set_xlabel("Power (MW)", fontsize=8)
        ax.set_title(CHARGING_MODES[mode]["label"], fontsize=9)
        if col == 0:
            ax.set_ylabel("Duration (hr)", fontsize=8)
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=7, color="black" if vmin + (vmax-vmin)*0.3 < v < vmin + (vmax-vmin)*0.8 else "white")

    # Delta panel: unrestricted − solar_only
    ax = axes[3]
    delta = irr_matrices["unrestricted"].values - irr_matrices["solar_only"].values
    lim   = max(abs(delta.min()), abs(delta.max()), 0.1)
    im2   = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(irr_matrices["unrestricted"].columns)))
    ax.set_xticklabels([f"{int(c)}" for c in irr_matrices["unrestricted"].columns], fontsize=8)
    ax.set_yticks(range(len(irr_matrices["unrestricted"].index)))
    ax.set_yticklabels([f"{int(d)} hr" for d in irr_matrices["unrestricted"].index], fontsize=8)
    ax.set_xlabel("Power (MW)", fontsize=8)
    ax.set_title("IRR uplift: Unrestricted − Solar only (pp)", fontsize=9)
    fig.colorbar(im2, ax=ax, shrink=0.75, label="IRR diff (pp)")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta[i,j]:+.1f}", ha="center", va="center",
                    fontsize=7)

    fig.colorbar(im, ax=axes[:3].tolist(), shrink=0.55, label="IRR (%)", pad=0.01)
    plt.tight_layout()
    out = _HERE / "outputs" / "surplus_irr_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Chart 2: Revenue waterfall — best IRR case per mode ──────────────────────
def chart2_revenue_waterfall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    modes  = list(CHARGING_MODES.keys())
    colors = {"energy_revenue_usd": "#27AE60", "capacity_revenue_usd": "#3498DB",
              "ancillary_revenue_usd": "#9B59B6", "charging_cost_usd": "#E74C3C"}
    labels = {"energy_revenue_usd": "Energy revenue",
              "capacity_revenue_usd": "Capacity revenue",
              "ancillary_revenue_usd": "Ancillary revenue",
              "charging_cost_usd": "Charging cost"}

    x       = np.arange(len(modes))
    w       = 0.18
    offsets = {"energy_revenue_usd": -1.5, "capacity_revenue_usd": -0.5,
               "ancillary_revenue_usd": 0.5, "charging_cost_usd": 1.5}

    for col_name, off in offsets.items():
        vals = []
        for mode in modes:
            row = df[df["charging_mode"] == mode].nlargest(1, "irr_pct").iloc[0]
            vals.append(-row[col_name] / 1e6 if col_name == "charging_cost_usd"
                        else row[col_name] / 1e6)
        bars = ax.bar(x + off * w, vals, w, color=colors[col_name],
                      alpha=0.85, label=labels[col_name])
        for bar, v in zip(bars, vals):
            if abs(v) > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.1 if v >= 0 else -0.3),
                        f"${abs(v):.1f}M", ha="center", va="bottom", fontsize=6.5)

    # Net revenue line
    for i, mode in enumerate(modes):
        row     = df[df["charging_mode"] == mode].nlargest(1, "irr_pct").iloc[0]
        net_m   = row["net_annual_revenue_usd"] / 1e6
        pw, dur = int(row["power_mw"]), int(row["duration_hr"])
        ax.hlines(net_m, i - 2*w, i + 2*w, colors="black", linewidth=2)
        ax.text(i, net_m + 0.15, f"Net ${net_m:.1f}M\n{pw}MW/{dur}hr",
                ha="center", fontsize=7, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CHARGING_MODES[m]["label"] for m in modes])
    ax.set_ylabel("Annual Revenue ($M)")
    ax.set_title("Revenue Waterfall — Best IRR Case per Charging Mode\n"
                 "(black line = net annual revenue)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = _HERE / "outputs" / "surplus_revenue_waterfall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Chart 3: Price vs dispatch scatter (best case overall) ────────────────────
def chart3_price_dispatch(df: pd.DataFrame, hourly_data: dict,
                           hourly_prices: list) -> None:
    best = df.nlargest(1, "irr_pct").iloc[0]
    key  = (best["power_mw"], best["duration_hr"], best["charging_mode"])
    ts   = hourly_data[key]

    hod  = np.tile(np.arange(24), 365)
    px   = np.array(hourly_prices)
    py   = ts["batt_mw"]

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(px, py, c=hod, cmap="twilight", alpha=0.25, s=4)
    fig.colorbar(sc, ax=ax, label="Hour of day")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Spot price ($/MWh)")
    ax.set_ylabel("Battery power (MW)  [+ = discharge / − = charge]")
    ax.set_title(
        f"Price vs Dispatch Scatter — Best case: "
        f"{int(best['power_mw'])} MW / {int(best['duration_hr'])} hr  "
        f"({CHARGING_MODES[best['charging_mode']]['label']})\n"
        f"IRR: {best['irr_pct']:.1f}%  |  "
        f"Annual discharge: {best['batt_annual_discharge_energy_kwh']/1e6:.1f} TWh"
    )
    plt.tight_layout()
    out = _HERE / "outputs" / "surplus_price_dispatch.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hourly_prices = load_prices()
    print(f"Price range: ${min(hourly_prices):.2f} – ${max(hourly_prices):.2f}/MWh  "
          f"avg: ${sum(hourly_prices)/len(hourly_prices):.2f}/MWh")

    tasks = build_tasks(hourly_prices)
    print(f"Running {len(tasks)} cases ({NUM_WORKERS} workers)…")

    df, hourly_data = run_sweep(tasks)

    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved → {RESULTS_CSV}")

    if len(df) > 0:
        chart1_irr_heatmap(df)
        chart2_revenue_waterfall(df, )
        chart3_price_dispatch(df, hourly_data, hourly_prices)
        print("\nAll 3 PNGs generated.")

        # Summary
        best = df.nlargest(3, "irr_pct")[["power_mw","duration_hr","charging_mode",
                                           "irr_pct","npv_usd","lcos_usd_per_kwh"]]
        print("\nTop 3 cases by IRR:")
        print(best.to_string(index=False))

"""
run_sizing_study.py

Standalone script to execute the 25-case PV+BESS sizing sweep, save the CSV,
and regenerate the heatmap and summary table PNGs.
"""
import sys, time, warnings, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pvsamlab import System, TrackingMode
from pv_bess_sizing_worker import run_bess_case

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', font_scale=0.9)

# ---- Config ----
SITE_LAT = 33.0278; SITE_LON = -100.0814; MET_YEAR = '2017'
PV_TARGET_KWAC = 300_000; PV_DCAC = 1.35
POWER_MW_LIST    = [50, 75, 100, 125, 150]
DURATION_HR_LIST = [1, 2, 4, 6, 8]
LOAD_KW = 200_000
PPA_RATE = 45.0; PRICE_CSV = None
CAPACITY_PAYMENT_PER_KW_YEAR = 80.0; ANCILLARY_PER_KW_YEAR = 15.0
FIN_KWARGS = dict(
    analysis_period=25, pv_capex_per_kwdc=700.0, opex_per_kwac_year=15.0,
    ppa_rate=PPA_RATE, ppa_escalation=1.0, discount_rate=8.0, inflation_rate=2.5,
    debt_fraction=0.0, loan_rate=5.0, loan_term=18, federal_tax_rate=21.0,
    itc_rate=0.0, depreciation_schedule='MACRS5',
)
BATT_KWARGS = dict(
    chemistry='LFP', capex_per_kwh=250.0, capex_per_kw=150.0,
    opex_per_kwh_year=8.0, soc_min=10.0, soc_max=95.0, calendar_degradation=2.0,
)
NUM_WORKERS = 8
_HERE = pathlib.Path(__file__).parent
RESULTS_CSV = _HERE / 'outputs' / 'pv_bess_sizing_study_results.csv'
HEATMAP_PNG = _HERE / 'outputs' / 'pv_bess_sizing_heatmap.png'
TABLE_PNG   = _HERE / 'outputs' / 'pv_bess_sizing_table.png'


def _pivot(df, metric):
    return df.pivot(index='duration_hr', columns='power_mw', values=metric).sort_index(ascending=False)

def _best_rc(piv, maximize=True):
    arr = piv.values
    idx = arr.argmax() if maximize else arr.argmin()
    return np.unravel_index(idx, arr.shape)

def _red_border(ax, row, col):
    ax.add_patch(patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
        fill=False, edgecolor='red', linewidth=2.5, linestyle='--'))


def main():
    # ---- Cache PV plant ----
    print('Building fixed PV plant (300 MWac)...')
    pv_plant = System(lat=SITE_LAT, lon=SITE_LON, target_kwac=PV_TARGET_KWAC,
                      target_dcac=PV_DCAC, met_year=MET_YEAR,
                      tracking_mode=TrackingMode.SAT)
    pv_results = pv_plant.run()
    print(f'Annual energy: {pv_results["annual_energy"]/1e6:.1f} GWh/yr')

    # ---- Build tasks ----
    price_series = [PPA_RATE] * 8760
    revenue_kwargs = dict(
        energy_arbitrage_prices=price_series,
        capacity_payment_per_kw_year=CAPACITY_PAYMENT_PER_KW_YEAR,
        ancillary_services_per_kw_year=ANCILLARY_PER_KW_YEAR,
    )
    tasks = [
        {
            'power_kw': power_mw * 1000, 'duration_hr': duration_hr,
            'lat': SITE_LAT, 'lon': SITE_LON, 'met_year': MET_YEAR,
            'pv_target_kwac': PV_TARGET_KWAC, 'pv_dcac': PV_DCAC,
            'financial_kwargs': FIN_KWARGS, 'batt_kwargs': BATT_KWARGS,
            'revenue_kwargs': revenue_kwargs,
            'load_profile': [LOAD_KW] * 8760,
        }
        for power_mw in POWER_MW_LIST for duration_hr in DURATION_HR_LIST
    ]

    print(f'\nRunning {len(tasks)} cases with {NUM_WORKERS} workers...')
    t_start = time.time()
    ok_results = []; err_results = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_bess_case, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc='BESS cases'):
            res = future.result()
            (ok_results if res['status'] == 'ok' else err_results).append(res)

    elapsed = time.time() - t_start
    print(f'\nDone: {len(ok_results)} OK, {len(err_results)} failed in {elapsed:.1f}s')
    if err_results:
        for e in err_results:
            print(f'  {e["power_mw"]:.0f} MW / {e["duration_hr"]}h: {e.get("error", "?")}')

    # ---- Save CSV ----
    df = pd.DataFrame(ok_results).sort_values(['power_mw', 'duration_hr']).reset_index(drop=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nSaved CSV -> {RESULTS_CSV}')

    # Sanity check
    print('\nDischarge sample (should be > 0):')
    cols = ['power_mw', 'duration_hr', 'batt_annual_discharge_energy_kwh', 'lcos_usd_per_kwh']
    print(df[cols].head(5).to_string(index=False))

    # ---- Heatmaps ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'PV+BESS Parametric Sizing Study - Scurry County TX\n'
        f'300 MWac PV | self_consumption dispatch (200 MW flat load) | '
        f'${PPA_RATE:.0f}/MWh + ${CAPACITY_PAYMENT_PER_KW_YEAR:.0f} cap '
        f'+ ${ANCILLARY_PER_KW_YEAR:.0f} anc ($/kW-yr)', fontsize=11,
    )

    piv = _pivot(df, 'irr_pct'); r, c = _best_rc(piv, True)
    ax = axes[0, 0]
    sns.heatmap(piv, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn', center=8, linewidths=0.5, cbar_kws={'label': 'IRR (%)'})
    _red_border(ax, r, c); ax.set_title('IRR (%)  [diverging, center 8% | red = best]')
    ax.set_xlabel('Power (MW)'); ax.set_ylabel('Duration (hr)')

    piv = _pivot(df, 'npv_usd') / 1e6; r, c = _best_rc(piv, True)
    ax = axes[0, 1]
    sns.heatmap(piv, ax=ax, annot=True, fmt='.0f', cmap='RdYlGn', center=0, linewidths=0.5, cbar_kws={'label': 'NPV ($M)'})
    _red_border(ax, r, c); ax.set_title('NPV ($M)  [diverging, center $0 | red = best]')
    ax.set_xlabel('Power (MW)'); ax.set_ylabel('Duration (hr)')

    piv = _pivot(df, 'lcos_usd_per_kwh'); r, c = _best_rc(piv, False)
    ax = axes[1, 0]
    sns.heatmap(piv, ax=ax, annot=True, fmt='.3f', cmap='YlOrRd_r', linewidths=0.5, cbar_kws={'label': 'LCOS ($/kWh)'})
    _red_border(ax, r, c); ax.set_title('LCOS ($/kWh)  [lower is better | red = min]')
    ax.set_xlabel('Power (MW)'); ax.set_ylabel('Duration (hr)')

    piv = _pivot(df, 'batt_annual_discharge_energy_kwh') / 1000; r, c = _best_rc(piv, True)
    ax = axes[1, 1]
    sns.heatmap(piv, ax=ax, annot=True, fmt='.0f', cmap='Blues', linewidths=0.5, cbar_kws={'label': 'Discharge (MWh/yr)'})
    _red_border(ax, r, c); ax.set_title('BESS Annual Discharge (MWh/yr)  [red = max]')
    ax.set_xlabel('Power (MW)'); ax.set_ylabel('Duration (hr)')

    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved heatmap -> {HEATMAP_PNG}')

    # ---- Summary table ----
    df_ranked = df.sort_values('irr_pct', ascending=False).reset_index(drop=True).assign(rank=lambda x: x.index + 1)
    _HDRS = ['Rank', 'Power MW', 'Dur hr', 'Energy MWh', 'IRR %', 'NPV $M', 'LCOE c/kWh', 'LCOS $/kWh', 'Discharge MWh/yr']

    def _fmt(row):
        return [f"{int(row['rank'])}", f"{row['power_mw']:.0f}", f"{row['duration_hr']:.0f}",
                f"{row['energy_mwh']:.0f}", f"{row['irr_pct']:.2f}", f"{row['npv_usd']/1e6:.1f}",
                f"{row['lcoe_real_cents_per_kwh']:.2f}", f"{row['lcos_usd_per_kwh']:.3f}",
                f"{row['batt_annual_discharge_energy_kwh']/1000:.0f}"]

    cell_text = [_fmt(row) for _, row in df_ranked.iterrows()]
    n_rows = len(df_ranked)
    fig, ax = plt.subplots(figsize=(15, max(4.0, 0.45 * n_rows + 1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=_HDRS, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(_HDRS))))
    for j in range(len(_HDRS)):
        cell = tbl[0, j]; cell.set_facecolor('#2c5f8a'); cell.set_text_props(color='white', fontweight='bold')
    for i in range(min(3, n_rows)):
        for j in range(len(_HDRS)): tbl[i + 1, j].set_facecolor('#90ee90')
    for i in range(3, n_rows):
        bg = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(_HDRS)): tbl[i + 1, j].set_facecolor(bg)
    ax.set_title('PV+BESS Sizing Study - 25 Cases Ranked by IRR  (top 3 in green)',
                 fontsize=11, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(TABLE_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved table  -> {TABLE_PNG}')


if __name__ == '__main__':
    main()

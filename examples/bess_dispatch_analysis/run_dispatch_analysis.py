"""
run_dispatch_analysis.py

Standalone script that runs all 25 PV+BESS cases and generates the 6 PNG
outputs for pv_bess_dispatch_analysis.ipynb.
"""
import sys, pathlib, warnings
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pvsamlab import System, TrackingMode, Battery, BessDispatch, PvBessSystem

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', font_scale=0.9)

# ── Config ────────────────────────────────────────────────────────────────────
SITE_LAT       = 33.0278
SITE_LON       = -100.0814
MET_YEAR       = '2017'
PV_TARGET_KWAC = 300_000
PV_DCAC        = 1.35
LOAD_KW        = 200_000

POWER_MW_LIST    = [50, 75, 100, 125, 150]
DURATION_HR_LIST = [1,  2,   4,   6,   8]

CASE_A = (50,  2)
CASE_B = (100, 4)
CASE_C = (150, 8)
CASE_LABELS = {
    CASE_A: f'{CASE_A[0]} MW / {CASE_A[1]} hr  (best IRR)',
    CASE_B: f'{CASE_B[0]} MW / {CASE_B[1]} hr  (balanced)',
    CASE_C: f'{CASE_C[0]} MW / {CASE_C[1]} hr  (max storage)',
}
BATT_KWARGS = dict(
    chemistry='LFP', capex_per_kwh=250.0, capex_per_kw=150.0,
    opex_per_kwh_year=8.0, soc_min=10.0, soc_max=95.0, calendar_degradation=2.0,
)

DT_IDX        = pd.date_range('2017-01-01', periods=8760, freq='h')
SUMMER_SLICE  = slice(4344, 4344 + 7 * 24)
WINTER_SLICE  = slice(0, 7 * 24)
MONTH_START_DAYS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTH_ABBR    = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']


# ── Simulate 25 cases ─────────────────────────────────────────────────────────
def simulate_all():
    ts_data = {}
    disp = BessDispatch(strategy='self_consumption', can_gridcharge=[0] * 6)
    cases = [(p, d) for p in POWER_MW_LIST for d in DURATION_HR_LIST]

    for power_mw, duration_hr in tqdm(cases, desc='Simulating cases'):
        energy_kwh = power_mw * 1000 * duration_hr
        batt = Battery(
            energy_kwh=float(energy_kwh),
            power_kw=float(power_mw * 1000),
            **BATT_KWARGS,
        )
        plant = PvBessSystem(
            lat=SITE_LAT, lon=SITE_LON,
            target_kwac=PV_TARGET_KWAC, target_dcac=PV_DCAC,
            met_year=MET_YEAR, tracking_mode=TrackingMode.SAT,
            battery=batt, dispatch=disp,
            load_profile=[LOAD_KW] * 8760,
        )
        plant.run()
        out = plant.model.Outputs
        ts_data[(power_mw, duration_hr)] = {
            'hourly_gen_mw':    np.array(list(out.gen))        / 1000.0,
            'hourly_batt_mw':   np.array(list(out.batt_power)) / 1000.0,
            'hourly_soc_pct':   np.array(list(out.batt_SOC)),
            'hourly_export_mw': np.array(list(out.grid_power)) / 1000.0,
            'energy_kwh':       energy_kwh,
        }
    print(f'Simulated {len(ts_data)} cases.')
    return ts_data


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _plot_dispatch_panel(ax, ts, sl, panel_title, show_legend=False):
    gen_w  = ts['hourly_gen_mw'][sl]
    batt_w = ts['hourly_batt_mw'][sl]
    soc_w  = ts['hourly_soc_pct'][sl]
    exp_w  = ts['hourly_export_mw'][sl]
    dt_w   = DT_IDX[sl]
    hours  = np.arange(len(dt_w))
    pv_ac  = gen_w - np.maximum(batt_w, 0)
    b_dis  = np.maximum(batt_w, 0)
    b_chg  = np.minimum(batt_w, 0)
    ax.fill_between(hours, 0, pv_ac,              color='#FFD700', alpha=0.85, label='PV gen')
    ax.fill_between(hours, pv_ac, pv_ac + b_dis,  color='#27AE60', alpha=0.75, label='BESS discharge')
    ax.fill_between(hours, b_chg, 0,              color='#E74C3C', alpha=0.70, label='BESS charge')
    ax.plot(hours, exp_w, color='#2C3E50', linewidth=1.5, label='Grid export', zorder=5)
    ax.axhline(300, color='black', linestyle='--', linewidth=0.8, alpha=0.35, label='300 MW nameplate')
    ax2 = ax.twinx()
    ax2.plot(hours, soc_w, color='grey', linestyle='--', linewidth=1.0, alpha=0.6)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel('SOC (%)', color='grey', fontsize=7)
    ax2.tick_params(axis='y', labelcolor='grey', labelsize=7)
    day_ticks = np.arange(0, len(hours), 24)
    ax.set_xticks(day_ticks)
    ax.set_xticklabels([dt_w[i].strftime('%b %d') for i in day_ticks], fontsize=7)
    ax.set_title(panel_title, fontsize=8)
    ax.set_ylabel('Power (MW)', fontsize=8)
    if show_legend:
        ax.legend(loc='upper right', fontsize=6, ncol=3)


# ── Chart 1: weekly dispatch ──────────────────────────────────────────────────
def chart1_weekly(ts_data):
    fig, axes = plt.subplots(3, 2, figsize=(20, 14))
    fig.suptitle(
        'Representative Week Dispatch — 300 MWac PV + BESS\n'
        'self_consumption dispatch | 200 MW flat load', fontsize=12,
    )
    for row, case in enumerate([CASE_A, CASE_B, CASE_C]):
        ts  = ts_data[case]
        lbl = CASE_LABELS[case]
        _plot_dispatch_panel(axes[row, 0], ts, SUMMER_SLICE,
                             f'Summer week (Jul 1–7)  |  {lbl}', show_legend=(row == 0))
        _plot_dispatch_panel(axes[row, 1], ts, WINTER_SLICE,
                             f'Winter week (Jan 1–7)  |  {lbl}')
    plt.tight_layout()
    out = _HERE / 'outputs' / 'dispatch_weekly.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


# ── Chart 2: SOC heatmap ──────────────────────────────────────────────────────
def chart2_soc(ts_data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Full Year State of Charge Heatmap — Scurry County TX 2017', fontsize=11)
    _im = None
    for col, case in enumerate([CASE_A, CASE_B, CASE_C]):
        ts     = ts_data[case]
        soc_2d = ts['hourly_soc_pct'][:8760].reshape(365, 24).T
        energy = ts['energy_kwh']
        total_dis = np.maximum(ts['hourly_batt_mw'], 0).sum() * 1000
        avg_cyc   = total_dis / (energy * 365)
        ax  = axes[col]
        _im = ax.pcolormesh(np.arange(365), np.arange(24), soc_2d,
                            cmap='RdYlGn', vmin=0, vmax=100)
        for d, m in zip(MONTH_START_DAYS, MONTH_ABBR):
            if d > 0:
                ax.axvline(d, color='grey', linewidth=0.5, alpha=0.5)
            ax.text(d + 1, 23.5, m, fontsize=6, color='black', va='top', ha='left')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Hour of Day' if col == 0 else '')
        ax.set_title(f'{CASE_LABELS[case]}\navg daily cycles = {avg_cyc:.2f}', fontsize=9)
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_yticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'], fontsize=7)
    fig.colorbar(_im, ax=axes.tolist(), shrink=0.55, label='SOC (%)', pad=0.02)
    plt.tight_layout()
    out = _HERE / 'outputs' / 'soc_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


# ── Chart 3: duration curves ──────────────────────────────────────────────────
def chart3_duration(ts_data):
    _ref      = ts_data[CASE_A]
    pv_only   = _ref['hourly_gen_mw'] - _ref['hourly_batt_mw']
    x_hrs     = np.arange(8760)
    pv_dc     = np.sort(pv_only)[::-1]
    dc_a      = np.sort(ts_data[CASE_A]['hourly_gen_mw'])[::-1]
    dc_b      = np.sort(ts_data[CASE_B]['hourly_gen_mw'])[::-1]
    dc_c      = np.sort(ts_data[CASE_C]['hourly_gen_mw'])[::-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x_hrs, pv_dc, color='grey',    linewidth=2.0, label='PV only')
    ax.plot(x_hrs, dc_a,  color='#AED6F1', linewidth=1.5, label=f'Case A  {CASE_LABELS[CASE_A]}')
    ax.plot(x_hrs, dc_b,  color='#2980B9', linewidth=1.5, label=f'Case B  {CASE_LABELS[CASE_B]}')
    ax.plot(x_hrs, dc_c,  color='#1A5276', linewidth=2.0, label=f'Case C  {CASE_LABELS[CASE_C]}')
    ax.fill_between(x_hrs, pv_dc, dc_c, alpha=0.10, color='#3498DB',
                    label='BESS net contribution (PV-only vs Case C)')
    ax.axvline(4380, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(300,  color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.text(4390, max(pv_dc[0], dc_c[0]) * 0.96, '← 4380 h (CF = 50 %)', fontsize=8)
    ax.text(150, 305, '300 MW nameplate', fontsize=8, alpha=0.6)
    for dc, lbl, col in [(pv_dc,'PV','grey'),(dc_a,'Case A','#AED6F1'),
                          (dc_b,'Case B','#2980B9'),(dc_c,'Case C','#1A5276')]:
        cf = dc[4380] / 300 * 100
        ax.annotate(f'{lbl}: {cf:.0f} % CF', xy=(4380, dc[4380]),
                    xytext=(4520, dc[4380] - 5), fontsize=7, color=col,
                    arrowprops=dict(arrowstyle='-', color=col, lw=0.8))
    ax.set_xlabel('Hours (ranked, descending)')
    ax.set_ylabel('Power (MW)')
    ax.set_title('Duration Curves — PV-only vs PV+BESS Cases')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 8760)
    plt.tight_layout()
    out = _HERE / 'outputs' / 'duration_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


# ── Chart 4: monthly waterfall ────────────────────────────────────────────────
def chart4_monthly(ts_data):
    ts = ts_data[CASE_B]
    df = pd.DataFrame({
        'gen_mw':    ts['hourly_gen_mw'],
        'batt_mw':   ts['hourly_batt_mw'],
        'soc_pct':   ts['hourly_soc_pct'],
        'export_mw': ts['hourly_export_mw'],
    }, index=DT_IDX)
    pv_m  = (df['gen_mw'] - df['batt_mw'].clip(lower=0)).resample('ME').sum() / 1000
    dis_m = df['batt_mw'].clip(lower=0).resample('ME').sum() / 1000
    chg_m = (-df['batt_mw'].clip(upper=0)).resample('ME').sum() / 1000
    exp_m = df['export_mw'].resample('ME').sum() / 1000
    soc_m = df['soc_pct'].resample('ME').mean()
    months = [m.strftime('%b') for m in pv_m.index]
    x = np.arange(len(months))
    w = 0.20
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle(f'Monthly Energy Balance — Case B: {CASE_LABELS[CASE_B]}', fontsize=11)
    b1 = ax1.bar(x - 1.5*w, pv_m,   w, color='#FFD700', alpha=0.85, label='PV gen (GWh)')
    b2 = ax1.bar(x - 0.5*w, -chg_m, w, color='#E74C3C', alpha=0.80, label='BESS charge (GWh)')
    b3 = ax1.bar(x + 0.5*w, dis_m,  w, color='#27AE60', alpha=0.80, label='BESS discharge (GWh)')
    b4 = ax1.bar(x + 1.5*w, exp_m,  w, color='#2C3E50', alpha=0.85, label='Net export (GWh)')
    for bar_grp, is_neg in [(b1, False), (b3, False), (b4, False), (b2, True)]:
        for bar in bar_grp:
            h = bar.get_height()
            if abs(h) > 1.0:
                ypos = h + 0.5 if not is_neg else h - 0.5
                va   = 'bottom' if not is_neg else 'top'
                ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                         f'{abs(h):.0f}', ha='center', va=va, fontsize=5.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    ax1.set_ylabel('Energy (GWh)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.plot(x, soc_m, color='grey', marker='o', markersize=4, linewidth=1.5,
             linestyle='--', alpha=0.75, label='Avg SOC %')
    ax2.set_ylabel('Avg SOC (%)', color='grey', fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor='grey')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    plt.tight_layout()
    out = _HERE / 'outputs' / 'monthly_waterfall.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


# ── Chart 5: dispatch heatmap ─────────────────────────────────────────────────
def chart5_heatmap(ts_data):
    ts = ts_data[CASE_B]
    export_2d = ts['hourly_export_mw'][:8760].reshape(365, 24).T
    vmax = float(np.percentile(np.abs(ts['hourly_export_mw']), 99))
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.pcolormesh(np.arange(365), np.arange(24), export_2d,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label='Net export (MW)', shrink=0.8, pad=0.01)
    for d, m in zip(MONTH_START_DAYS, MONTH_ABBR):
        if d > 0:
            ax.axvline(d, color='white', linewidth=0.9, alpha=0.7)
        ax.text(d + 1, 22.5, m, fontsize=6.5, color='white', va='top', ha='left')
    ax.invert_yaxis()
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Hour of Day')
    ax.set_title(
        f'Hourly Dispatch Heatmap — Case B: {CASE_LABELS[CASE_B]}\n'
        'Red = exporting to grid  |  Blue = net import (battery charging from excess PV)',
        fontsize=9,
    )
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'])
    plt.tight_layout()
    out = _HERE / 'outputs' / 'dispatch_hourly_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


# ── Chart 6: utilization analysis ────────────────────────────────────────────
def chart6_utilization(ts_data):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Battery Utilization Analysis — Three Representative Cases', fontsize=11)
    palette = {CASE_A: '#AED6F1', CASE_B: '#2980B9', CASE_C: '#1A5276'}
    for case in [CASE_A, CASE_B, CASE_C]:
        ts    = ts_data[case]
        color = palette[case]
        label = CASE_LABELS[case]
        energy_kwh = ts['energy_kwh']
        batt_mw    = ts['hourly_batt_mw']
        daily_dis  = np.maximum(batt_mw * 1000, 0).reshape(365, 24).sum(axis=1)
        cycle_d    = daily_dis / energy_kwh
        ax_l.hist(cycle_d, bins=40, alpha=0.5, color=color, label=label)
        ax_l.axvline(cycle_d.mean(), color=color, linewidth=1.5, linestyle='--',
                     label=f'Mean = {cycle_d.mean():.2f}')
        soc_2d   = ts['hourly_soc_pct'][:8760].reshape(365, 24)
        mean_soc = soc_2d.mean(axis=0)
        std_soc  = soc_2d.std(axis=0)
        hrs = np.arange(24)
        ax_r.plot(hrs, mean_soc, color=color, linewidth=1.5, label=label)
        ax_r.fill_between(hrs, mean_soc - std_soc, mean_soc + std_soc,
                          color=color, alpha=0.15)
    ax_l.set_xlabel('Daily Cycle Depth (fraction of energy capacity)')
    ax_l.set_ylabel('Number of Days')
    ax_l.set_title('Daily Cycle Depth Distribution')
    ax_l.legend(fontsize=7)
    ax_r.set_xlabel('Hour of Day')
    ax_r.set_ylabel('Mean SOC (%)')
    ax_r.set_title('Average SOC by Hour of Day (±1 std dev shaded)')
    ax_r.set_xticks(range(0, 24, 3))
    ax_r.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=8)
    ax_r.legend(fontsize=7)
    plt.tight_layout()
    out = _HERE / 'outputs' / 'utilization_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


if __name__ == '__main__':
    ts_data = simulate_all()
    chart1_weekly(ts_data)
    chart2_soc(ts_data)
    chart3_duration(ts_data)
    chart4_monthly(ts_data)
    chart5_heatmap(ts_data)
    chart6_utilization(ts_data)
    print('\nAll 6 PNGs generated.')

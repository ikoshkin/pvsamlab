# PV+BESS Dispatch Analysis

Deep-dive hourly dispatch diagnostics for the 25-case sizing matrix produced
by `bess_sizing_study/`.

## Dependencies

Requires `bess_sizing_study/outputs/pv_bess_sizing_study_results.csv` for the
interactive widget summary stats. The simulation itself re-runs all 25 cases
from scratch (cached in kernel memory after first run).

## Charts produced

| Chart | Description |
|-------|-------------|
| 1 — Weekly dispatch | Stacked area: PV gen, BESS discharge/charge, grid export |
| 2 — SOC heatmap | Full year 24×365 state of charge (green=high, red=low) |
| 3 — Duration curves | Ranked generation curves vs PV-only baseline |
| 4 — Monthly waterfall | Monthly energy balance with avg SOC secondary axis |
| 5 — Dispatch heatmap | Hour-of-day × day-of-year net export (RdBu diverging) |
| 6 — Utilization | Daily cycle depth histogram + hourly mean SOC ±1σ |

Three representative cases are compared:
- **Case A** — 50 MW / 2 hr (best IRR)
- **Case B** — 100 MW / 4 hr (balanced)
- **Case C** — 150 MW / 8 hr (max storage)

## Part 2 — Interactive widget

Dropdowns for all 25 cases. 4-panel output:
- Summer week dispatch stack
- Full year SOC heatmap
- Duration curve vs PV-only
- Daily cycle depth histogram

Summary stats (IRR, LCOS, NPV) loaded from sizing study CSV.

## Expected outputs (written to `outputs/`)

`dispatch_weekly.png`, `soc_heatmap.png`, `duration_curves.png`,
`monthly_waterfall.png`, `dispatch_hourly_heatmap.png`, `utilization_analysis.png`

## Running

Open `pv_bess_dispatch_analysis.ipynb` in Jupyter, or headless:
```bash
cd examples/bess_dispatch_analysis
python run_dispatch_analysis.py
```

## Estimated runtime

~5 min (25 simulations, no parallelism in notebook mode).

# PV+BESS Parametric Sizing Study

Sweeps BESS **power** (50–150 MW) × **duration** (1–8 hr) against a fixed
300 MWac PV plant to produce a 5×5 = 25-case sizing matrix.

## Configuration (`pv_bess_sizing_study.ipynb` Cell 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SITE_LAT/LON` | 33.0278, -100.0814 | Scurry County TX |
| `PV_TARGET_KWAC` | 300,000 kW | Fixed PV plant size |
| `POWER_MW_LIST` | [50,75,100,125,150] | BESS power sweep |
| `DURATION_HR_LIST` | [1,2,4,6,8] | BESS duration sweep |
| `LOAD_KW` | 200,000 | Flat dispatch load (self-consumption) |
| `PPA_RATE` | 45.0 $/MWh | Revenue placeholder |
| `NUM_WORKERS` | 8 | Parallel worker processes |

## Expected outputs (written to `outputs/`)

| File | Description |
|------|-------------|
| `pv_bess_sizing_study_results.csv` | 25-row results table |
| `pv_bess_sizing_heatmap.png` | 2×2 heatmap grid (IRR, NPV, LCOS, Discharge) |
| `pv_bess_sizing_table.png` | 25 cases ranked by IRR |

## Running

Open `pv_bess_sizing_study.ipynb` in Jupyter.

Or headless:
```bash
cd examples/bess_sizing_study
python run_sizing_study.py
```

## Estimated runtime

~5 min with 8 workers (25 PySAM simulations).

## Notes

- Uses `self_consumption` dispatch (not `price_signal`) — see notebook comments
  for why `price_signal` produces zero discharge without a full rate matrix.
- `pv_bess_sizing_worker.py` must be in the same folder for multiprocessing.
- The output CSV is consumed by `bess_dispatch_analysis/` as an input.

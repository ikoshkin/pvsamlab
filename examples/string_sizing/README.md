# String Sizing Tool

Sweeps PV string configurations to find the maximum open-circuit voltage (Voc)
across historical weather years. Use the results to confirm that no combination
of module and string length can exceed the inverter DC voltage limit.

## Inputs

- `.PAN` module files in `pan_files/` (or any folder you specify in `PAN_FOLDERS`)
- `.OND` inverter file (defaults to the bundled Sungrow SG4400)
- NSRDB weather data (downloaded automatically on first run, then cached)

## Configuration

Edit the **Configuration** cell in `string_sizing.ipynb`:

| Parameter | Description |
|-----------|-------------|
| `VOLTAGE_LIMIT` | DC voltage limit (V) — typically 1500 V |
| `LAT`, `LON` | Site coordinates |
| `YEAR_RANGE` | Weather years to simulate (NSRDB: 1998–2023) |
| `STRING_RANGE` | Modules-per-string values to sweep |
| `PAN_FOLDERS` | List of folders containing `.PAN` files |
| `NUM_WORKERS` | Parallel worker processes |

## Expected outputs (written to `outputs/`)

| File | Description |
|------|-------------|
| `string_sizing_results_summary.csv` | Per-case max Voc and MPPT loss |
| `string_sizing_results_hourly.csv` | Full 8760-hour Voc, Vmp, Isc timeseries |
| `voc_by_year.png` | Max Voc vs year line plot for each module/string combo |
| `voc_heatmap.png` | Colour-coded grid: green=safe, amber=marginal, red=fail |
| `assumptions_table.png` | Summary of system & loss assumptions |

## Running

Open `string_sizing.ipynb` in Jupyter and run all cells top to bottom.

Or run headless via `parallel_string_sizing.py`:
```bash
cd examples/string_sizing
python parallel_string_sizing.py
```

## Estimated runtime

~3 min for 2 modules × 26 years × 5 string lengths with 8 workers.

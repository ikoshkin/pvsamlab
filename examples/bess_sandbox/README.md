# BESS Sandbox

Verification sandbox for the `pvsamlab` BESS simulation API. Runs four
simulation modes and prints key metrics to confirm physically plausible results.

## Simulation modes

| Cell | Mode | Description |
|------|------|-------------|
| 3 | PV+BESS | 300 MWac PV + 400 MWh LFP battery (self-consumption dispatch) |
| 4 | Standalone BESS | 400 MWh battery only (grid arbitrage) |
| 5 | Financial | SAM Singleowner LCOE/IRR + Python LCOS |
| 6 | Merchant price curve | Price-signal dispatch with DAM prices + RevenueStack |
| 7 | Sizing loop | LCOS vs energy capacity sweep (400–550 MWh) |

## Running

Open `bess_sandbox.ipynb` in Jupyter and run all cells top to bottom.

The executed snapshot `bess_sandbox_executed.ipynb` shows expected output
for Cells 1–5 at commit `120d4ac`.

## Weather file dependency

The notebook uses the NSRDB API to fetch weather data for
Scurry County TX (`lat=33.0278, lon=-100.0814, year=2017`).
The file is cached in `pvsamlab/data/weather_files/` after the first run.

Set `NSRDB_API_KEY`, `NSRDB_API_NAME`, and `NSRDB_API_EMAIL` in
`pvsamlab/secrets.env` before running on a fresh clone.

## Verified metrics (commit `120d4ac`)

| Metric | Value |
|--------|-------|
| PV+BESS annual discharge | 43.9 M kWh |
| Standalone BESS annual discharge | 87.5 M kWh |
| PV-only IRR | 12.82% |
| LCOS | $0.37 /kWh |

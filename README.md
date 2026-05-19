# pvsamlab

**Python wrapper around NREL's System Advisor Model (SAM) for utility-scale PV and BESS analysis**

`Python 3.10+` | `PySAM >= 4.0` | `License: MIT`

---

## What it does

pvsamlab wraps NREL's [PySAM](https://nrel-pysam.readthedocs.io/) library behind clean dataclasses. Instead of manually wiring hundreds of SSC variable names, you configure a `System`, `Battery`, and `Financial` object with the parameters that matter — DC/AC ratio, tracking mode, battery energy/power, ITC rate — and pvsamlab handles the rest.

The library supports three simulation modes: **PV-only** (`System`), **PV + BESS** (`PvBessSystem`), and **standalone BESS** (`StandaloneBessSystem`). All three share the same weather-download pipeline (NSRDB via REST API), the same PVsyst PAN/OND parser, and the same financial post-processing functions.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | >= 3.10 | 3.13 tested via conda |
| NREL-PySAM | >= 4.0 | `pip install nrel-pysam` |
| numpy | >= 1.24 | |
| pandas | >= 2.0 | |
| scipy | >= 1.10 | |
| pvlib | >= 0.10 | |
| python-dotenv | >= 1.0 | Loads `secrets.env` |
| requests | >= 2.28 | NSRDB weather download |

You also need a **free API key** for weather data downloads: [https://developer.nlr.gov/signup/](https://developer.nlr.gov/signup/)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/ikoshkin/pvsamlab.git
cd pvsamlab
```

### 2. Create conda environment (recommended)

```bash
conda create -n pvsamlab python=3.13
conda activate pvsamlab
```

### 3. Install the package

```bash
pip install -e ".[dev]"
```

The `-e` flag installs in editable mode so changes to `pvsamlab/` take effect immediately without reinstalling.

### 4. Set up API credentials

Create a `secrets.env` file in the repo root (gitignored — never commit it):

```
NSRDB_API_KEY=your_key_here
NSRDB_API_EMAIL=your_email_here
NSRDB_API_NAME=your_name_here
```

---

## Quick start

### PV-only simulation

```python
from pvsamlab import System, TrackingMode

plant = System(
    lat=33.0278,
    lon=-100.0814,
    target_kwac=100_000,   # 100 MWac
    target_dcac=1.35,
    met_year='2017',
    tracking_mode=TrackingMode.SAT,
)
results = plant.run()
print(f"Annual energy: {results['annual_energy']:,.0f} kWh")
print(f"Performance ratio: {results['performance_ratio']:.1f}%")
```

### PV + BESS simulation

```python
from pvsamlab import Battery, BessDispatch, PvBessSystem

batt = Battery(energy_kwh=400_000, power_kw=100_000)
disp = BessDispatch(
    strategy='price_signal',
    energy_arbitrage_prices=hourly_prices,  # 8760 $/MWh
)
plant = PvBessSystem(
    lat=33.0278, lon=-100.0814,
    target_kwac=100_000, target_dcac=1.35,
    met_year='2017',
    battery=batt, dispatch=disp,
)
results = plant.run()
```

### Financial metrics

```python
from pvsamlab import Financial, compute_lcoe, compute_lcos

fin = Financial(
    ppa_rate=45.0,       # $/MWh
    itc_rate=30.0,       # % IRA ITC
    discount_rate=8.0,   # % real WACC
)
metrics = compute_lcoe(plant, fin)
print(f"LCOE: {metrics['lcoe_real_cents_per_kwh']:.2f} ¢/kWh")
print(f"IRR:  {metrics['irr_pct']:.1f}%")
```

---

## Repository structure

```
pvsamlab/
├── pvsamlab/                    ← Python package
│   ├── system.py                ← System dataclass, PV simulation
│   ├── battery.py               ← Battery, BessDispatch, PvBessSystem, StandaloneBessSystem
│   ├── financial.py             ← Financial, RevenueStack, compute_lcoe/lcos/irr/npv
│   ├── components.py            ← Module, Inverter dataclasses
│   ├── climate.py               ← NSRDB weather download and caching
│   ├── utils.py                 ← String sizing utilities
│   └── data/
│       ├── modules/             ← PAN files (add yours here)
│       ├── inverters/           ← OND files
│       └── _ashraeDB.csv        ← ASHRAE extreme temperature data
├── examples/                    ← Ready-to-run engineering tools (see examples/ for READMEs)
│   ├── string_sizing/           ← NEC 690.7 string sizing study
│   ├── bess_sandbox/            ← BESS simulation verification
│   ├── bess_sizing_study/       ← PV+BESS parametric sweep
│   ├── bess_dispatch_analysis/  ← Dispatch visualizations
│   └── bess_surplus/            ← Brownfield BESS optimizer
├── tests/
└── pyproject.toml
```

---

## Adding PAN / OND files

Place PAN files in `pvsamlab/data/modules/` and OND files in `pvsamlab/data/inverters/`, then load them:

```python
from pvsamlab import Module, Inverter

module   = Module.from_pan('pvsamlab/data/modules/your_module.PAN')
inverter = Inverter.from_ond('pvsamlab/data/inverters/your_inverter.OND')
```

The string sizing tool scans its own folder — copy or symlink your PAN files there:

```bash
cp pvsamlab/data/modules/your_module.PAN examples/string_sizing/pan_files/
```

---

## Weather data

pvsamlab downloads weather files from the **NSRDB** via `https://developer.nlr.gov` (the `developer.nrel.gov` domain was shut down May 29, 2026). Files are cached under `pvsamlab/data/weather_files/<lat>_<lon>/` so repeated runs at the same location do not re-download.

Credentials are loaded from `secrets.env` via `python-dotenv`. Required variables: `NSRDB_API_KEY`, `NSRDB_API_EMAIL`, `NSRDB_API_NAME`.

Pass `met_year='tmy'` for a Typical Meteorological Year or `met_year='2017'` (any year `'1998'`–`'2023'`) for historical data. Verify credentials before running:

```python
from pvsamlab import check_nsrdb_connectivity
check_nsrdb_connectivity()
```

---

## Working offline / corporate networks

To use pre-downloaded weather files, drop them in `pvsamlab/data/weather_files/`. Files are matched by lat/lon and year in the filename. CSV, EPW, and TMY3 formats are accepted.

pvsamlab checks that folder recursively before making any API call, so any file whose name contains the site lat/lon (to 2 decimal places) and year will be found automatically — no configuration required.

---

## Known limitations

- **`price_signal` dispatch** requires an 8760-element price array; shorter arrays raise a PySAM SSC error.
- **Standalone BESS** is always AC-coupled — a PySAM/SAM architectural constraint.
- **Lifetime degradation** requires `system_use_lifetime_output=1` and `analysis_period > 1`; single-year runs return only Year 1.
- **LCOS** is computed by pvsamlab's `compute_lcos()`, not natively by SAM; follows NREL methodology but has not been independently audited.
- **PAN/OND parser** covers fields used by PySAM simulation inputs; PVsyst-specific fields are parsed but ignored.

---

## License

MIT — see [LICENSE](LICENSE) for details.

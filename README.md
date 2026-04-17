# pvsamlab

**Python wrapper around NREL's System Advisor Model (SAM) for utility-scale PV and BESS analysis**

`Python 3.10+` | `PySAM >= 6.0` | `License: MIT`

---

## What it does

pvsamlab wraps NREL's [PySAM](https://nrel-pysam.readthedocs.io/) library — the Python interface to SAM's Simulation Core (SSC) — behind clean, business-logic-friendly dataclasses. Instead of manually wiring hundreds of SSC variable names, you configure a `System`, `Battery`, and `Financial` object with the parameters that matter for your project (DC/AC ratio, tracking mode, battery energy/power, ITC rate, discount rate), and pvsamlab handles the rest.

The library supports three simulation modes: **PV-only** (`System`), **PV + BESS** (`PvBessSystem`), and **standalone BESS** (`StandaloneBessSystem`). All three modes share the same weather-download pipeline (NSRDB via REST API), the same PVsyst PAN/OND module and inverter parser, and the same financial post-processing functions (`compute_lcoe`, `compute_lcos`, `compute_irr`, `compute_npv`).

On top of the core library, the `examples/` directory contains five ready-to-run engineering tools: a NEC 690.7 string sizing study (parallel, multi-module, multi-location), a BESS sizing parametric sweep, a dispatch visualization notebook, a brownfield BESS surplus optimizer, and a sandbox for exploring raw PySAM outputs. Each tool is a self-contained Jupyter notebook with a configuration cell at the top — no code changes required for typical use.

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Python | >= 3.10 | 3.13 tested via conda |
| NREL-PySAM | >= 6.0.1 | `pip install nrel-pysam` |
| numpy | >= 1.24 | |
| pandas | >= 2.0 | |
| scipy | >= 1.10 | |
| pvlib | >= 0.10 | |
| python-dotenv | >= 1.0 | Loads `secrets.env` |
| requests | >= 2.28 | NSRDB weather download |
| pytest | >= 7.0 | Dev/test only |
| tqdm | latest | Dev/example scripts only |

You also need a **free NREL API key** for weather data downloads: [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/)

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

Create a `secrets.env` file in the repo root (this file is gitignored — never commit it):

```
NSRDB_API_KEY=your_key_here
NSRDB_API_EMAIL=your_email_here
NSRDB_API_NAME=your_name_here
```

Get a free API key at [https://developer.nrel.gov/signup/](https://developer.nrel.gov/signup/). pvsamlab loads these automatically via `python-dotenv` when you call `System.run()` or `PvBessSystem.run()`.

---

## Repository structure

```
pvsamlab/
├── pvsamlab/                    ← Python package
│   ├── system.py                ← System dataclass, PV simulation
│   ├── battery.py               ← Battery, BessDispatch, PvBessSystem,
│   │                               StandaloneBessSystem
│   ├── financial.py             ← Financial, RevenueStack, compute_lcoe,
│   │                               compute_lcos, compute_irr, compute_npv
│   ├── components.py            ← Module, Inverter dataclasses
│   ├── climate.py               ← NSRDB weather download
│   ├── utils.py                 ← String sizing utilities
│   └── data/
│       ├── modules/             ← PAN files for PV modules
│       │   └── *.PAN            ← PVsyst module files (add your own here)
│       ├── inverters/           ← OND files for inverters
│       │   └── *.OND            ← PVsyst inverter files
│       ├── _modulesDB.csv       ← Module database for Module.from_db()
│       └── _ashraeDB.csv        ← ASHRAE extreme temperature data
├── examples/
│   ├── string_sizing/           ← NEC 690.7 string sizing tool
│   │   ├── pan_files/           ← PUT YOUR PAN FILES HERE for string sizing
│   │   │                           (copied or symlinked from pvsamlab/data/modules/)
│   │   ├── parallel_string_sizing.py
│   │   ├── string_sizing.ipynb
│   │   └── outputs/             ← generated CSVs and PNGs saved here
│   ├── bess_sandbox/            ← BESS simulation verification notebook
│   │   └── bess_sandbox.ipynb
│   ├── bess_sizing_study/       ← 25-case PV+BESS parametric sizing
│   │   ├── pv_bess_sizing_study.ipynb
│   │   └── outputs/
│   ├── bess_dispatch_analysis/  ← Operational dispatch visualizations
│   │   ├── pv_bess_dispatch_analysis.ipynb
│   │   └── outputs/
│   ├── bess_surplus/            ← Brownfield BESS optimization tool
│   │   ├── bess_surplus_optimization.ipynb
│   │   ├── bess_surplus_worker.py
│   │   └── outputs/
│   ├── sandbox/                 ← PySAM/pvsamlab exploration notebooks
│   └── CONTRIBUTING.md          ← How to add new tools
├── tests/                       ← Unit and integration tests
├── AUDIT.md                     ← Code audit from Phase 0
├── PROGRESS.md                  ← Development progress log
├── pyproject.toml
├── requirements.txt
└── secrets.env                  ← NOT committed — create manually
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

## Adding PAN / OND files

### For the pvsamlab library (used by System and PvBessSystem)

Place PAN files in `pvsamlab/data/modules/` and OND files in `pvsamlab/data/inverters/`, then load them:

```python
from pvsamlab import Module, Inverter

module   = Module.from_pan('pvsamlab/data/modules/your_module.PAN')
inverter = Inverter.from_ond('pvsamlab/data/inverters/your_inverter.OND')
```

You can also look up modules by name from the bundled database:

```python
module = Module.from_db('SunPower SPR-X21-335')
```

### For the string sizing tool specifically

The string sizing tool scans its own dedicated folder:

```
examples/string_sizing/pan_files/
```

Copy or symlink your PAN files there:

```bash
cp pvsamlab/data/modules/your_module.PAN \
   examples/string_sizing/pan_files/
```

Then open `examples/string_sizing/string_sizing.ipynb` and set `PAN_FOLDERS` in the configuration cell at the top.

---

## Available tools

### String sizing — `examples/string_sizing/string_sizing.ipynb`

Runs NEC 690.7-compliant string sizing across one or more modules and locations. Uses ASHRAE extreme cold temperatures and module `Voc` temperature coefficients from PAN files. Computes maximum string length, minimum string length for MPPT compliance, and hours outside MPPT range. Runs in parallel across module/location combinations.

- **Open:** `examples/string_sizing/string_sizing.ipynb`
- **Key config:** `PAN_FOLDERS`, `LOCATIONS` (list of `(lat, lon, name)`), `INVERTER_OND`
- **Outputs:** heatmap PNG, per-module/location CSV — saved to `outputs/`
- **Runtime:** ~1–5 min depending on number of modules and locations

### BESS sandbox — `examples/bess_sandbox/bess_sandbox.ipynb`

Minimal notebook for verifying BESS simulation behaviour: charge/discharge curves, SOC trajectories, round-trip efficiency. Use this to sanity-check PySAM battery model settings before running larger studies.

- **Open:** `examples/bess_sandbox/bess_sandbox.ipynb`
- **Key config:** battery capacity, power, dispatch strategy, single location
- **Outputs:** time-series plots inline
- **Runtime:** < 1 min

### BESS sizing study — `examples/bess_sizing_study/pv_bess_sizing_study.ipynb`

Parametric sweep over battery energy/power ratios (e.g., 25 cases: 5 C-rates × 5 energy sizes) for a fixed PV plant. Computes annual generation, battery throughput, LCOS, and IRR for each case. Produces a results table and heatmap.

- **Open:** `examples/bess_sizing_study/pv_bess_sizing_study.ipynb`
- **Key config:** `ENERGY_SIZES_KWH`, `POWER_SIZES_KW`, plant location, dispatch strategy
- **Outputs:** results CSV, heatmap PNG — saved to `outputs/`
- **Runtime:** 5–20 min (25 simulations)

### Dispatch analysis — `examples/bess_dispatch_analysis/pv_bess_dispatch_analysis.ipynb`

Loads a completed PV+BESS simulation result and produces operational visualizations: daily dispatch stacking, monthly energy balance, price vs. dispatch correlation, SOC histogram.

- **Open:** `examples/bess_dispatch_analysis/pv_bess_dispatch_analysis.ipynb`
- **Key config:** path to simulation outputs CSV, price signal array
- **Outputs:** multi-panel figures saved to `outputs/`
- **Runtime:** < 1 min (post-processing only)

### BESS surplus optimizer — `examples/bess_surplus/bess_surplus_optimization.ipynb`

Brownfield tool for evaluating adding a BESS to an existing PV plant with a fixed interconnection limit. Sweeps battery size and charging mode to maximize revenue from surplus PV energy that would otherwise be curtailed. Uses `bess_surplus_worker.py` for parallelized runs.

- **Open:** `examples/bess_surplus/bess_surplus_optimization.ipynb`
- **Key config:** existing plant capacity, interconnection limit, price signal, charging mode
- **Outputs:** optimization surface plots, optimal sizing recommendation — saved to `outputs/`
- **Runtime:** 10–30 min depending on sweep resolution

---

## Dispatch strategies

| Strategy | PySAM value | Use case | `load_profile` needed |
|---|---|---|---|
| `manual` | 0 | Custom hourly schedule | No |
| `peak_shaving` | 2 | C&I demand charge reduction | Yes (8760 kW) |
| `self_consumption` | 3 | BTM solar + storage | Yes (8760 kW) |
| `price_signal` | 4 | Merchant energy arbitrage | No |

Set the strategy via `BessDispatch(strategy='price_signal', ...)`. For `price_signal`, pass an 8760-element array to `energy_arbitrage_prices`. For `self_consumption` and `peak_shaving`, pass an 8760-element load profile to `load_profile`.

---

## Charging modes (brownfield / surplus projects)

When adding a BESS to an existing plant with a constrained interconnection, three charging modes determine the battery's energy source:

| Mode | `batt_dispatch_auto_can_gridcharge` | When to use |
|---|---|---|
| **Solar only** | `0` | Battery charges exclusively from PV clipping; minimizes import costs |
| **Grid only** | `1` | Battery charges from grid during off-peak hours; maximises arbitrage flexibility |
| **Unrestricted** | `2` (default) | Battery charges from solar or grid, whichever is cheaper; general-purpose |

Configure via `BessDispatch(charging_mode='solar_only')` or by setting the SSC variable directly.

---

## Weather data

pvsamlab downloads weather files from NREL's **National Solar Radiation Database (NSRDB)** via REST API. Files are cached locally as CSVs so subsequent runs at the same location/year do not re-download.

**Credentials** are loaded from `secrets.env` in the repo root via `python-dotenv`. The three required variables are `NREL_API_KEY`, `NREL_EMAIL`, and `NREL_NAME`.

**`met_year` options:**

- `'tmy'` — Typical Meteorological Year, a synthetic year constructed from the statistically most representative months across the full NSRDB record. Use for long-run P50 energy estimates.
- `'2017'` (or any year string `'1998'`–`'2022'`) — Actual measured data for a specific year. Use for historical analysis, P90 studies, or validating against metered generation.

Weather files are cached under `pvsamlab/data/weather/<lat>_<lon>_<year>.csv`. Delete the cache file to force a fresh download.

---

## Known limitations

- **`price_signal` dispatch** requires an 8760-element price array. Shorter arrays (e.g., monthly averages) will raise a PySAM SSC error.
- **Standalone BESS** (`StandaloneBessSystem`) is always AC-coupled — this is a PySAM/SAM architectural constraint, not a pvsamlab limitation.
- **Lifetime degradation output** (annual energy by year) requires setting `system_use_lifetime_output=1` and `analysis_period` > 1 on the `System` object. Single-year runs return only Year 1 results.
- **LCOS** is computed post-simulation by pvsamlab's `compute_lcos()` function, not natively by SAM. The calculation follows NREL's standard methodology (present value of total costs / present value of energy discharged) but has not been independently audited.
- **PAN/OND parser** covers the fields used by pvsamlab's simulation inputs. Fields not referenced by PySAM (e.g., PVsyst-specific loss breakdown) are parsed but ignored.

---

## License

MIT — see [LICENSE](LICENSE) for details.

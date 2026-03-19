# pvsamlab — Full Repo Audit

> **Auditor:** Senior Python / Energy-Modeling Engineer
> **Date:** 2026-03-18
> **Scope:** All phases — Repo Audit, Improvement Suggestions, BESS Roadmap, Input Mapping, Financial Metrics

---

## Phase 1 — Repo Audit

### Repository Layout

```
pvsamlab/
├── pvsamlab/
│   ├── __init__.py          — empty package init (exports nothing)
│   ├── core.py              — dead stub (print statement only)
│   ├── models.py            — OLDER/BROKEN: System + Module + Inverter + Losses, broken import
│   ├── system.py            — ACTIVE: System dataclass, generate_pysam_inputs(), process_outputs()
│   ├── components.py        — Module + Inverter dataclasses, .from_pan()/.from_ond() classmethods
│   ├── climate.py           — NSRDB weather download, ASHRAE lookup, uses dotenv correctly
│   ├── utils.py             — OLDER/BROKEN: duplicate functions, hardcoded credentials
│   └── data/
│       ├── _ashraeDB.csv         — ASHRAE extreme low temperatures
│       ├── _modulesDB.csv        — referenced but existence not confirmed (from_db())
│       ├── _invertersDB.csv      — referenced but existence not confirmed (from_db())
│       ├── modules/              — PAN files for various modules
│       ├── inverters/            — OND files + sam_sg4400.csv
│       ├── tmp/                  — gitignored but model_export.json + 1 JSON committed
│       ├── weather_files/        — gitignored but pre-downloaded CSV files committed
│       └── untitled_pvsamv1.json — accidentally committed full SAM parameter export
├── examples/
│   ├── inputs_sample.json
│   ├── sandbox_pysam.ipynb       — PySAM API exploration
│   ├── sandbox_pvsamlab.ipynb    — pvsamlab usage demos
│   └── string_sizing/
│       ├── parallel_string_sizing.py — BROKEN: logic bug (see below)
│       ├── string_sizing.ipynb
│       ├── log.txt                   — unmerged in git (UU status)
│       └── *.csv                     — generated outputs, should be gitignored
├── tests/
│   └── test_core.py         — single trivial test (callable check only)
├── df.csv                   — debug artifact in repo root
├── pyproject.toml           — minimal, missing all runtime dependencies
├── requirements.txt         — pinned deps (not wired to pyproject.toml)
├── README.md                — placeholder text only
└── .gitignore               — missing several patterns
```

---

### `pvsamlab/__init__.py`

**Purpose:** Makes the directory a Python package.

**Issues:**
- Exports nothing. Users must know to import from `pvsamlab.system`, `pvsamlab.components`, etc. A proper `__init__.py` should expose the public API.

---

### `pvsamlab/core.py`

**Purpose:** Placeholder / original scaffold.

```python
def run_simulation():
    print("Running PV simulation with PySAM...")
```

**Issues:**
- Dead code. The actual simulation is in `system.py`. This file has no purpose and its only test (`test_core.py`) tests nothing meaningful.
- Should either be deleted or repurposed as the library's public entry point.

---

### `pvsamlab/models.py` (542 lines — BROKEN, SUPERSEDED)

**Purpose:** Original monolithic implementation: `Module`, `Inverter`, `Losses`, `System` dataclasses + `assign_pysam_values()` + `generate_ssc_input()`.

**Red flags:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Line 12 | **ImportError on startup**: `from pvsamlab.utils import ... fetch_weather_file` — `fetch_weather_file` does not exist in `utils.py`. Any `import pvsamlab.models` crashes. |
| 2 | Line 2 & 4 | `import os` appears twice |
| 3 | Lines 235–236 | Calls `calculate_string_size(self.module, self.design_low_temp, self.system_voltage)` but `utils.calculate_string_size` takes `(module_voc, module_tc_voc, design_low_temp, system_voltage)` — signature mismatch, would crash at runtime |
| 4 | Line 245 | `inv_tdc_ds = [[1, 52.8, -0.021]]` — hardcoded voltage/temp derating curve, should come from inverter spec |
| 5 | Line 411 | `ssc_input['subarray1_soiling'] = [3.00] * 12` — hardcoded soiling override inside `generate_ssc_input()`, ignores `plant.losses.subarray1_soiling` |
| 6 | All | **No docstrings** on any class or function |

**Verdict:** This file is broken and superseded by `system.py` + `components.py`. It should be deleted.

---

### `pvsamlab/system.py` (613 lines — ACTIVE)

**Purpose:** Primary module. `System` dataclass that orchestrates module/inverter/weather loading, string sizing, inverter sizing, PySAM configuration, execution, and output extraction. Key functions: `generate_pysam_inputs()`, `process_outputs()`.

**Red flags:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Lines 15 & 21 | `from pathlib import Path` imported twice |
| 2 | Line 1–3 | Module docstring has mismatched quotes: `'''...system'` |
| 3 | Lines 25–26 | `LATITUDE = 30.9759 / LONGITUDE = -97.2465` hardcoded — default location tied to pre-downloaded weather data, not a meaningful user-facing default |
| 4 | Line 110 | If `download_nsrdb_csv()` returns `None` (on API failure), the next `open(self.weather_file, ...)` crashes with `TypeError: expected str, bytes or os.PathLike object, not NoneType` — no null-check |
| 5 | Lines 113–116 & 122–125 | Header row indices differ for TMY vs NSRDB downloaded files: `tmy_header[4]/[5]` vs `tmy_header[5]/[6]` — must verify against actual file format |
| 6 | Line 148 | `inv_tdc_ds = [[1100, 50, -0.01, 55, -0.085, 60, -0.085]]` — hardcoded nominal voltage (1100 V) and temperature derating coefficients; should come from inverter data |
| 7 | Lines 271–272 | Subarrays 2, 3, 4 `track_mode` hardcoded to `TrackingMode.SAT` — ignores `plant.tracking_mode`, so a fixed-tilt plant would have subarrays 1=FT and 2–4=SAT |
| 8 | Line 404 | `'sixpar_mounting': 0` — hardcoded "open rack" mounting; should be user-configurable |
| 9 | Line 406 | `'sixpar_standoff': 6.0` — hardcoded standoff category; should be configurable |
| 10 | Line 408 | `'sixpar_transient_thermal_model_unit_mass': 11.0919` — magic number, no documentation |
| 11 | Lines 471–473 | `grid_interconnection_limit_kwac: 100000.0` — hardcoded 100 MW grid limit, not derived from plant size |
| 12 | Line 505 | Commented-out line has syntax error: `plant.model Outputs.annual_ac_interconnect_loss_percent` (missing `.`) |
| 13 | Lines 482–514 | `process_outputs()`: mixed key naming style — `annual_ghi` (snake_case) vs `'Nominal POA Irradiance'` (title case with spaces) |
| 14 | Line 157–159 | **`__post_init__` runs the simulation** — construction and execution are coupled; impossible to construct a `System` object without triggering a full weather download + PySAM run |
| 15 | Lines 517–613 | `if __name__ == '__main__'` block uses an absolute Windows path: `r'C:\Users\KV6378\...'` — committed test code that only works on one developer's machine |
| 16 | Lines 181–231 | Losses for subarrays 2–4 are hardcoded inside `generate_pysam_inputs()` and do not read from any user-supplied `Losses` object (no `Losses` dataclass is even used here, unlike `models.py`) |
| 17 | None | No type hints on `generate_pysam_inputs()` or `process_outputs()` return types |
| 18 | None | No validation on inputs: negative DC:AC ratios, zero `target_kwac`, impossible string voltage — all silently produce nonsensical results |

---

### `pvsamlab/components.py` (265 lines — ACTIVE)

**Purpose:** Defines `CellTech`, `MlModelParameters`, `Module`, and `Inverter` dataclasses with `.from_pan()` / `.from_ond()` / `.from_db()` classmethods for loading from PVsyst files and a local CSV database.

**Red flags:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Line 165 | `"noct": 46, # FIXME` — hardcoded NOCT, not read from PAN file (pvlib `read_panond` does not expose this field directly; workaround needed) |
| 2 | Line 167 | `"bifacial_transmission_factor": 0.05` — hardcoded, should be derived from module spec or user-provided |
| 3 | Lines 235–236 | `"inv_num_mppt": 1, #TODO for now hardcoded for Sungrow` — overwrites the value read from OND file on the previous line; dead assignment |
| 4 | Lines 253–257 | **Unreachable dead code**: duplicate `return cls(**extracted_params)` / `except` block after the try/except in `Inverter.from_ond()` — will never execute |
| 5 | Lines 75–80 | `MlModelParameters.__post_init__` assigns to **local variables** (`AM_c_sa = ...`, `IAM_c_cs_iamValue = ...`) instead of `self.AM_c_sa` etc. — initialization has no effect; all these fields remain empty lists |
| 6 | Lines 111–118 | `Module.from_db()` references `_modulesDB.csv` — not confirmed present in repo |
| 7 | Lines 192–200 | `Inverter.from_db()` references `_invertersDB.csv` — not confirmed present in repo |
| 8 | Line 73 | Typo in field name: `ground_relfection_fraction` (should be `reflection`) |
| 9 | Line 154 | `"pmax": pan_dict.get("Vmp") * pan_dict.get("Imp")` — computes Pmax from Vmp×Imp rather than using the datasheet `PNom`. This will give a slightly different (usually lower) value. The commented-out `pan_dict.get("PNom")` on the previous line is more accurate. |
| 10 | None | `Module.from_pan()` does not extract or pass `MlModelParameters` — the MLM model pathway (model 5) is incomplete |

---

### `pvsamlab/climate.py` (141 lines — ACTIVE)

**Purpose:** Weather resource download via NREL NSRDB GOES v4 (using `PySAM.ResourceTools.FetchResourceFiles`), ASHRAE extreme low temperature lookup, spatial nearest-neighbor search.

**Issues:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Lines 19–21 | `api_key`, `email`, `your_name` are module-level globals — if `secrets.env` is missing, these are `None` and `FetchResourceFiles` will fail silently with an opaque error; should validate at function call time |
| 2 | Line 87 | Returns `None` on download failure without raising — callers get `None` and typically crash with unhelpful `TypeError` |
| 3 | Line 57 | Only recognizes `year.isdigit()` for annual data — would silently fall through to ValueError for something like `'1998.0'` |
| 4 | Lines 134–140 | `if __name__ == '__main__'` block downloads 27 years of data with no safeguard — running the module directly costs significant API quota |
| 5 | None | `attrs` (line 23) is module-level but never used by `FetchResourceFiles` — it's a remnant from the old `utils.py` implementation |
| 6 | None | `find_nearest()` rebuilds the KDTree on every call to `get_ashrae_design_low()` — for multi-simulation runs this is wasteful; the tree should be cached |

---

### `pvsamlab/utils.py` (210 lines — OLDER, PARTIALLY BROKEN)

**Purpose:** Originally the combined utility module. Now superseded by `climate.py` for weather/ASHRAE work and `components.py` for file parsing. Still imported by `components.py` (`parse_pan_file`).

**Red flags:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Lines 146–148 | **Hardcoded API credentials in source code**: `api_key = 'DEMO_KEY'`, `your_name = 'Lolo+Pepe'`, `email = 'lolo@pepe.com'` — these are plaintext placeholder credentials committed to version control |
| 2 | Lines 12–15 | `logging.basicConfig()` called at **module-import time** — this mutates the global logger configuration as a side effect of `import pvsamlab.utils`, polluting any downstream application's logging |
| 3 | Line 167 | `f'{coords}_nsrdb_{year}.csv'` where `coords` is a tuple — produces `(30.9759, -97.2465)_nsrdb_tmy.csv` with parentheses, an **invalid filename on Windows** |
| 4 | Lines 130–143 | `parse_ond_file()` reads OND as a semicolon-delimited CSV with columns `Max_AC_Power(W)`, `Efficiency(%)`, etc. — incompatible with actual PVsyst `.OND` format; only matches the custom `sam_sg4400.csv` file |
| 5 | Lines 6–7 | `import PySAM.Wfreader as wf` imported but never used |
| 6 | Lines 7 | `from pkg_resources import resource_filename` — `pkg_resources` is deprecated; should use `importlib.resources` |
| 7 | None | `download_nsrdb_csv()` in this file uses the old PSM3 API endpoint, while `climate.py` uses the newer GOES v4 via PySAM FetchResourceFiles — two completely different implementations of the same feature |
| 8 | None | `calculate_string_size()` signature is `(module_voc, module_tc_voc, design_low_temp, system_voltage)` but `models.py` calls it as `(module, design_low_temp, system_voltage)` — type mismatch |

---

### `examples/string_sizing/parallel_string_sizing.py` (144 lines)

**Purpose:** Batch parallel simulation over year range × string-length range for multiple modules at a given location. Uses `ProcessPoolExecutor`.

**Red flags:**

| # | Location | Issue |
|---|----------|-------|
| 1 | Lines 44–51 & 64–90 | **Logic bug (critical)**: the function `return`s a dict at line 51, making lines 64–90 dead code. But line 90 then tries `return summary, df_monthly, df_hourly` where `summary` is never defined (the dict returned on line 45 is anonymous). The `main()` function at line 120 unpacks `future.result()` as `summary, monthly, hourly` — this will raise `ValueError: not enough values to unpack` on every task. |
| 2 | Lines 120–127 | `tqdm.write(f"✓ {summary.get('Module')} ...")` — `summary` here refers to the variable from the unpacking, which would be a plain dict from the early return; `.get()` is valid but if the early-return dict lacks `'Module'`, it silently prints `None` |
| 3 | Lines 130–132 | `pd.concat(monthly_rows, ...)` and `pd.concat(hourly_rows, ...)` — `monthly_rows` and `hourly_rows` will always be empty because the monthly/hourly DataFrames are never built (dead code), causing `pd.concat([])` to raise `ValueError` |
| 4 | Line 28 | `LAT, LON = swenson_upper_left` — hardcoded single location for all runs, not configurable at CLI |
| 5 | Line 70 | `pd.date_range(..., freq='H')` — `'H'` is deprecated in pandas ≥ 2.2, should be `'h'` |
| 6 | Line 29 | `NUM_WORKERS = 8` hardcoded; should respect available CPU count |
| 7 | None | No `if __name__ == '__main__'` guard around top-level `LAT, LON = swenson_upper_left` — this runs on import, not harmful but sloppy |

---

### `tests/test_core.py`

**Purpose:** Unit test file.

**Issues:**
- Only tests `assert callable(run_simulation)` — a tautological test that checks if a function object is callable.
- Zero tests for `system.py`, `components.py`, `climate.py`, or any simulation logic.
- No integration tests, no fixture-based tests.

---

### `pyproject.toml`

**Issues:**
- `dependencies = ["NREL-PySAM"]` — only lists PySAM. The library also requires `pandas`, `scipy`, `pvlib`, `python-dotenv`, `numpy`, and `requests`. `pip install pvsamlab` would fail at runtime.
- `authors = [{ name = "Your Name", ... }]` — placeholder name not replaced.
- No `[project.optional-dependencies]` for dev/test extras.
- Version pinning is only in `requirements.txt`, not in `pyproject.toml`.

---

### `README.md`

**Issues:**
- Contains only: `"A brief description of your project."` — placeholder text. No installation instructions, no usage examples, no API documentation.

---

### `.gitignore`

**Missing patterns (artifacts currently committed):**
- `df.csv` (debug file in repo root)
- `examples/**/*_results*.csv` (generated output CSVs)
- `examples/**/pivot_*.csv`
- `examples/**/log.txt`
- `pvsamlab/data/untitled_pvsamv1.json`

The `pvsamlab/data/tmp/*` and `pvsamlab/data/weather_files/*` patterns exist but one JSON in `tmp/` and all weather CSVs are already committed and will persist until explicitly removed with `git rm`.

---

## Phase 2 — Improvement Suggestions (Prioritized)

### A. Correctness Issues (fix immediately)

| Priority | File | Issue | Fix |
|----------|------|--------|-----|
| P0 | `models.py` | `ImportError` on import (`fetch_weather_file` missing) | Delete this file or fix import |
| P0 | `parallel_string_sizing.py` | `run_simulation()` returns dict early; lines 64–90 unreachable; `return summary, df_monthly, df_hourly` references undefined `summary`; caller unpacks 3 values from 1 | Restructure function: build summary/monthly/hourly in all paths, always return 3 values |
| P0 | `system.py:110` | No null-check on `download_nsrdb_csv()` return; crashes with opaque `TypeError` | `if self.weather_file is None: raise RuntimeError(...)` |
| P0 | `components.py:254-257` | Unreachable dead code (duplicate try/except after return) | Delete lines 253–257 |
| P1 | `components.py:75-80` | `MlModelParameters.__post_init__` assigns to locals, not `self` — all list fields stay empty | Change to `self.AM_c_sa = ...` etc. |
| P1 | `system.py:271-310` | Subarrays 2–4 track_mode hardcoded to `TrackingMode.SAT` | Use `plant.tracking_mode` for all subarrays |
| P1 | `utils.py:167` | Tuple-in-f-string produces invalid filename on Windows | Use `f'{lat:.4f}_{lon:.4f}_nsrdb_{year}.csv'` |
| P1 | `components.py:154` | `Pmax = Vmp × Imp` underestimates relative to datasheet PNom | Use `pan_dict.get("PNom")` as primary, fallback to `Vmp × Imp` |
| P1 | `utils.py:12-15` | `logging.basicConfig()` at module import | Remove; configure logging only in application entry points, not library code |

### B. Architecture Issues

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| P1 | `System.__post_init__` runs weather download + full PySAM simulation at construction time | Separate into `System.__init__` (stores config) and `System.run()` (executes simulation) |
| P1 | Two parallel implementations: `models.py`+`utils.py` vs `system.py`+`components.py`+`climate.py` | Delete `models.py`, `core.py`; keep the newer stack; strip dead functions from `utils.py` |
| P1 | Losses not user-configurable via `system.py` | Accept a `Losses` dataclass in `System`; merge into `generate_pysam_inputs()` |
| P2 | `generate_pysam_inputs()` is a 300-line flat dict builder — untestable monolith | Split into per-group builders: `_solar_resource_group()`, `_losses_group()`, `_system_design_group()`, etc. |
| P2 | `pyproject.toml` dependencies incomplete | Add all runtime deps: `pandas`, `scipy`, `pvlib`, `python-dotenv`, `numpy`, `requests` |
| P2 | `__init__.py` exports nothing | Add `from pvsamlab.system import System` and other public symbols |
| P3 | Weather caching done by directory structure inside the package — not portable | Move weather cache to `~/.pvsamlab/cache/` or user-specified directory |
| P3 | `find_nearest()` rebuilds KDTree on every ASHRAE lookup | Cache the tree as a module-level singleton after first build |

### C. Code Quality

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| P1 | Hardcoded API credentials in `utils.py` source | Remove entirely; only `climate.py` (via dotenv) should handle auth |
| P1 | `inv_tdc_ds` temperature-derating curve hardcoded | Derive from inverter spec or accept as user parameter |
| P1 | `grid_interconnection_limit_kwac` hardcoded to 100 MW | Derive from `plant.kwac` or make user-configurable |
| P2 | `process_outputs()` key naming inconsistency (`annual_ghi` vs `'Nominal POA Irradiance'`) | Standardize all keys to snake_case |
| P2 | `if __name__ == '__main__'` block in `system.py` uses absolute path | Replace with relative path using `pathlib` or remove entirely |
| P2 | `parallel_string_sizing.py` date range uses deprecated `freq='H'` | Change to `freq='h'` |
| P2 | `pkg_resources.resource_filename` (deprecated) | Replace with `importlib.resources.files()` |
| P3 | Typo: `ground_relfection_fraction` in `MlModelParameters` | Rename to `ground_reflection_fraction` |
| P3 | Double imports (`os` in models.py, `Path` in system.py) | Clean up |

### D. Documentation

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| P1 | `README.md` is a placeholder | Write installation, quick-start, and API reference |
| P1 | Zero docstrings on `System`, `generate_pysam_inputs()`, `process_outputs()` | Add NumPy-style docstrings to all public classes and functions |
| P2 | No changelog or version history | Add `CHANGELOG.md` |
| P2 | `# FIXME` and `# TODO` comments not tracked | Resolve or open issues |
| P3 | No type annotations on function return types | Add return type annotations throughout |

---

## Phase 3 — BESS Extension Roadmap

### 3.1 — PV Only (existing — gaps noted)

**SAM module:** `PySAM.Pvsamv1` with `en_batt=0` (default)

**Current gaps:**
- No `Losses` dataclass integration in `system.py` (losses are hardcoded in `generate_pysam_inputs()`)
- No multi-subarray support (subarrays 2–4 always disabled)
- No lifetime / degradation modeling (`Lifetime` group is empty `{}`)
- No financial compute module attached
- No grid curtailment curve (only a flat interconnection limit)
- Module models 3 (Sandia), 4 (IEC 61853), and 5 (MLM) not wired up
- Inverter models 0 (CEC database), 3 (part-load curve), 4 (Mermoud-Lejeune) not wired up

---

### 3.2 — BESS Only (standalone storage)

**SAM module:** `PySAM.StandAloneBattery`

This module takes an hourly AC load profile and simulates a battery dispatching against it, with no solar generation.

**New input categories needed:**

| Category | Key Inputs |
|----------|-----------|
| Battery specs | Chemistry (LFP/NMC), usable energy (kWh), C-rate, roundtrip efficiency, degradation rate |
| Dispatch strategy | Self-consumption, time-of-use (TOU), price-signal, manual schedule, peak shaving |
| Load profile | 8760-hour AC load series (kW) |
| Tariff structure | Energy charges ($/kWh per TOU period), demand charges ($/kW) |
| Grid connection | Import/export limits, interconnection capacity |
| Degradation | Calendar degradation (%/year), cycle degradation model |
| Thermal model | Cell temperature model, ambient temperature input |

**Suggested Python abstraction:**

```python
@dataclass
class Battery:
    chemistry: str = "LFP"           # LFP | NMC | Lead-acid
    energy_kwh: float = 4000.0       # Usable DC energy
    power_kw: float = 2000.0         # Max charge/discharge power
    soc_min: float = 10.0            # % min state of charge
    soc_max: float = 95.0            # % max state of charge
    roundtrip_efficiency: float = 87.5  # %
    calendar_degradation: float = 2.0   # %/year
    cycle_degradation: float = 0.03     # %/cycle (per MWh throughput)

@dataclass
class BessDispatch:
    strategy: str = "manual"         # manual | self_consumption | tou | price_signal
    schedule: List[List[int]] = ...  # 12×24 dispatch periods for manual mode

@dataclass
class StandaloneBessSystem:
    battery: Battery
    dispatch: BessDispatch
    load_profile: List[float]        # 8760 AC load values (kW)
    tariff: Tariff                   # see below

    def run(self) -> dict: ...
```

---

### 3.3 — PV + BESS (co-located)

**SAM module:** `PySAM.Pvsamv1` with `en_batt=1`

When battery is enabled in Pvsamv1, the following groups become active:
- `BatterySystem` — enable flag, capacity, voltage, chemistry
- `BatteryCell` — cell-level chemistry parameters
- `BatteryDispatch` — dispatch algorithm and schedules
- `Load` — optional load profile for self-consumption dispatch
- `ElectricityRates` — TOU tariff structure for rate-based dispatch
- `PriceSignal` — wholesale price signal for arbitrage dispatch
- `SystemCosts` — CAPEX/OPEX for each technology

**New input categories beyond PV-only:**

| Category | Key Inputs |
|----------|-----------|
| Battery specs | Same as BESS-only |
| Dispatch strategy | Residential self-consumption, commercial TOU, wholesale arbitrage, manual |
| AC/DC coupling | DC-coupled (charges from PV clipping) vs AC-coupled (charges from grid) |
| Load profile | 8760 load series if self-consumption dispatch |
| Tariff | TOU periods + energy/demand charges |
| Revenue | PPA rate, capacity payment, ancillary services |

---

### Suggested Class Hierarchy

```
BaseSystem (ABC)
├── attributes: lat, lon, weather_file, analysis_period
├── methods: run() → dict, validate() → None

PvSystem(BaseSystem)
├── attributes: module, inverter, losses, tracking_mode, ...
├── SAM: Pvsamv1(en_batt=0)

BessSystem(BaseSystem)
├── attributes: battery, dispatch, load_profile, tariff
├── SAM: StandAloneBattery

PvBessSystem(PvSystem)
├── attributes: battery, dispatch, coupling (AC|DC), load_profile, tariff
├── SAM: Pvsamv1(en_batt=1) + battery subgroups

# Shared components (no SAM dependency)
Battery       — chemistry, size, efficiency, degradation
BessDispatch  — strategy, schedule
Tariff        — TOU periods, rates, demand charges
Losses        — DC and AC loss categories
Financial     — CAPEX, OPEX, discount_rate, analysis_period, tax_rate
```

**Suggested module layout:**

```
pvsamlab/
├── __init__.py          — public API exports
├── system.py            — PvSystem (rename/refactor current System)
├── battery.py           — Battery, BessDispatch, BessSystem, PvBessSystem
├── components.py        — Module, Inverter (keep as-is)
├── climate.py           — weather download / ASHRAE (keep as-is)
├── losses.py            — Losses dataclass (extract from models.py)
├── financial.py         — Financial dataclass, IRR/NPV/LCOE/LCOS calculators
├── tariff.py            — Tariff, ElectricityRate structures
└── utils.py             — pure utility functions only (unit conversions, string sizing)
```

---

## Phase 4 — Input Mapping

> Legend: **R** = Required, **O** = Optional
> Config: **PV** = PV-only, **B** = BESS-only, **PB** = PV+BESS

### 4.1 Location & Weather

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| latitude | `lat` (weather file header) | ° | -90 to 90 | R | PV, B, PB |
| longitude | `lon` (weather file header) | ° | -180 to 180 | R | PV, B, PB |
| weather_year | — (download parameter) | — | 'tmy' or '1998'–'2023' | R | PV, B, PB |
| albedo (monthly) | `albedo` | fraction | [0.2]*12 | O | PV, PB |
| irradiance_mode | `irrad_mode` | enum | 0=DNI+DHI | O | PV, PB |
| sky_model | `sky_model` | enum | 2=Perez | O | PV, PB |
| design_low_temp | (ASHRAE lookup) | °C | computed | O | PV, PB |

### 4.2 Module (PV)

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| module_model | `module_model` | enum | 2 (6-par CEC user) | R | PV, PB |
| pmax | `sixpar_vmp × sixpar_imp` | W | 400–700 W | R | PV, PB |
| vmp | `sixpar_vmp` | V | 35–55 V | R | PV, PB |
| imp | `sixpar_imp` | A | 8–18 A | R | PV, PB |
| voc | `sixpar_voc` | V | 42–65 V | R | PV, PB |
| isc | `sixpar_isc` | A | 8–20 A | R | PV, PB |
| tc_pmax | `sixpar_gpmp` | %/°C | -0.5 to -0.2 | R | PV, PB |
| tc_voc | `sixpar_bvoc` | V/°C | derived from %/°C × Voc | R | PV, PB |
| tc_isc | `sixpar_aisc` | A/°C | derived from %/°C × Isc | R | PV, PB |
| n_series_cells | `sixpar_nser` | — | 60–144 | R | PV, PB |
| module_area | `sixpar_area` | m² | length × width | R | PV, PB |
| noct | `sixpar_tnoct` | °C | 40–50 | R | PV, PB |
| standoff | `sixpar_standoff` | enum | 6=open rack | O | PV, PB |
| mounting | `sixpar_mounting` | enum | 0=rack, 1=flush | O | PV, PB |
| is_bifacial | `sixpar_is_bifacial` | bool | True | O | PV, PB |
| bifaciality | `sixpar_bifaciality` | fraction | 0.65–0.80 | O | PV, PB |
| bifacial_transmission_factor | `sixpar_bifacial_transmission_factor` | fraction | 0.05 | O | PV, PB |
| bifacial_ground_clearance | `sixpar_bifacial_ground_clearance_height` | m | 1.0 | O | PV, PB |
| cell_technology | `sixpar_celltech` | enum | 0=Mono-Si | R | PV, PB |
| module_aspect_ratio | `module_aspect_ratio` | — | length/width | O | PV, PB |

### 4.3 Inverter (PV)

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| inverter_model | `inverter_model` | enum | 1=datasheet | R | PV, PB |
| pac_max | `inv_ds_paco` | W | 1e6–5e6 | R | PV, PB |
| eff_max | `inv_ds_eff` | % | 95–99 | R | PV, PB |
| night_loss | `inv_ds_pnt` | W | 100–1000 | O | PV, PB |
| oper_loss (Pso) | `inv_ds_pso` | W | 5000–30000 | O | PV, PB |
| vdc_nominal | `inv_ds_vdco` | V | 800–1200 | O | PV, PB |
| vdc_max | `inv_ds_vdcmax` | V | 1000–1500 | R | PV, PB |
| mppt_vmp_min | `mppt_low_inverter` | V | 800–1000 | R | PV, PB |
| mppt_vmp_max | `mppt_hi_inverter` | V | 1100–1500 | R | PV, PB |
| n_mppt_inputs | `inv_num_mppt` | — | 1–4 | O | PV, PB |
| temp_derate_curve | `inv_tdc_ds` | [[V,°C,%]] | per spec | O | PV, PB |
| inverter_count | `inverter_count` | — | computed | R | PV, PB |
| inverter_derate | (Python only) | fraction | 2000/2200 | O | PV, PB |

### 4.4 System Design (PV)

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| system_capacity_dc | `system_capacity` | kW | computed | R | PV, PB |
| target_kwac | (Python only) | kW | 100,000 | R | PV, PB |
| target_dcac_ratio | (Python only) | — | 1.25–1.40 | R | PV, PB |
| modules_per_string | `subarray1_modules_per_string` | — | computed from string sizing | R | PV, PB |
| n_strings | `subarray1_nstrings` | — | computed | R | PV, PB |
| system_voltage | (Python only) | V | 1500 | O | PV, PB |
| tracking_mode | `subarray1_track_mode` | enum | 0=fixed, 1=SAT, 2=DAT | R | PV, PB |
| tilt | `subarray1_tilt` | ° | 0–35 | R (FT) | PV, PB |
| azimuth | `subarray1_azimuth` | ° | 180 (south) | O | PV, PB |
| gcr | `subarray1_gcr` | fraction | 0.25–0.50 | R | PV, PB |
| rotation_limit | `subarray1_rotlim` | ° | 45–60 | O (SAT) | PV, PB |
| backtracking | `subarray1_backtrack` | bool | True | O (SAT) | PV, PB |
| n_modules_x | `subarray1_nmodx` | — | = modules_per_string | O | PV, PB |
| n_modules_y | `subarray1_nmody` | — | 1 | O | PV, PB |
| module_orientation | `subarray1_mod_orient` | enum | 0=portrait | O | PV, PB |
| enable_mismatch_vmax | `enable_mismatch_vmax_calc` | bool | False | O | PV, PB |
| mppt_input | `subarray1_mppt_input` | — | 1 | O | PV, PB |

### 4.5 Losses (PV)

| Business Name | SAM Variable | Units | Default | R/O | Config |
|---------------|-------------|-------|---------|-----|--------|
| ac_wiring_loss | `acwiring_loss` | % | 0.8 | R | PV, PB |
| dc_wiring_loss | `subarray1_dcwiring_loss` | % | 1.5 | R | PV, PB |
| diode_conn_loss | `subarray1_diodeconn_loss` | % | 0.5 | O | PV, PB |
| mismatch_loss | `subarray1_mismatch_loss` | % | 1.0 | O | PV, PB |
| nameplate_loss | `subarray1_nameplate_loss` | % | -0.4 (gain) | O | PV, PB |
| soiling (monthly) | `subarray1_soiling` | % | [2.5]*12 | R | PV, PB |
| tracking_loss | `subarray1_tracking_loss` | % | 0.5 | O (SAT) | PV, PB |
| rear_soiling_loss | `subarray1_rear_soiling_loss` | % | 0.0 | O (bifacial) | PV, PB |
| transformer_load_loss | `transformer_load_loss` | % | 0.6 | O | PV, PB |
| transformer_no_load_loss | `transformer_no_load_loss` | % | 0.1 | O | PV, PB |
| transmission_loss | `transmission_loss` | % | 0.5 | O | PV, PB |
| dc_optimizer_loss | `dcoptimizer_loss` | % | 0.0 | O | PV, PB |
| calculate_rack_shading | `calculate_rack_shading` | bool | True | O | PV, PB |
| calculate_bifacial_mismatch | `calculate_bifacial_electrical_mismatch` | bool | True | O (bifacial) | PV, PB |
| snow_model | `en_snow_model` | bool | False | O | PV, PB |

### 4.6 Shading (PV)

| Business Name | SAM Variable | Units | Default | R/O | Config |
|---------------|-------------|-------|---------|-----|--------|
| shade_mode | `subarray1_shade_mode` | enum | 1=standard | O | PV, PB |
| shading_string_option | `subarray1_shading_string_option` | enum | -1=none | O | PV, PB |

### 4.7 Grid

| Business Name | SAM Variable | Units | Default | R/O | Config |
|---------------|-------------|-------|---------|-----|--------|
| enable_interconnection_limit | `enable_interconnection_limit` | bool | True | O | PV, B, PB |
| interconnection_limit_kwac | `grid_interconnection_limit_kwac` | kW | = plant.kwac | O | PV, B, PB |
| grid_curtailment | `grid_curtailment` | kW array | — | O | PV, B, PB |

### 4.8 Battery (BESS and PV+BESS)

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| en_batt | `en_batt` | bool | 1 | R | B, PB |
| batt_simple_enable | `batt_simple_enable` | bool | 0 | O | B, PB |
| batt_chem | `batt_chem` | enum | 1=LFP, 0=lead acid | R | B, PB |
| batt_computed_bank_capacity | `batt_computed_bank_capacity` | kWh | 4000–50000 | R | B, PB |
| batt_power_charge_max_kwdc | `batt_power_charge_max_kwdc` | kW | capacity/4h | R | B, PB |
| batt_power_discharge_max_kwdc | `batt_power_discharge_max_kwdc` | kW | capacity/4h | R | B, PB |
| batt_minimum_SOC | `batt_minimum_SOC` | % | 10 | O | B, PB |
| batt_maximum_SOC | `batt_maximum_SOC` | % | 95 | O | B, PB |
| batt_initial_SOC | `batt_initial_SOC` | % | 50 | O | B, PB |
| batt_Vnom_default | `batt_Vnom_default` | V | 500–1000 | R | B, PB |
| batt_ac_or_dc | `batt_ac_or_dc` | enum | 0=DC, 1=AC coupled | R | B, PB |
| batt_dc_ac_efficiency | `batt_dc_ac_efficiency` | % | 96 | R | B, PB |
| batt_ac_dc_efficiency | `batt_ac_dc_efficiency` | % | 96 | R | B, PB |
| batt_roundtrip_efficiency | `batt_roundtrip_efficiency` | % | 87–92 | O | B, PB |
| batt_calendar_choice | `batt_calendar_choice` | enum | 0=none, 1=linear | O | B, PB |
| batt_calendar_q0 | `batt_calendar_q0` | fraction | 1.02 | O | B, PB |
| batt_calendar_a | `batt_calendar_a` | — | 0.003 | O | B, PB |
| batt_calendar_b | `batt_calendar_b` | — | -7280 | O | B, PB |
| batt_calendar_c | `batt_calendar_c` | — | 930 | O | B, PB |
| batt_cycle_degradation | `batt_cycle_degradation` | — | from chemistry | O | B, PB |
| batt_replacement_capacity | `batt_replacement_capacity` | % | 80 | O | B, PB |
| batt_replacement_schedule_percent | `batt_replacement_schedule_percent` | %/year array | — | O | B, PB |
| en_batt_replacement | `en_batt_replacement` | bool | False | O | B, PB |

### 4.9 Battery Dispatch

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| dispatch_strategy | `batt_dispatch_choice` | enum | 0=manual, 1=automated self-consumption, 2=automated peak shaving, 3=front-of-meter | R | B, PB |
| dispatch_manual_charge | `batt_dispatch_manual_charge` | 12×24 matrix | — | R (manual) | B, PB |
| dispatch_manual_discharge | `batt_dispatch_manual_discharge` | 12×24 matrix | — | R (manual) | B, PB |
| dispatch_manual_percent_discharge | `batt_dispatch_manual_percent_discharge` | %/period | — | O | B, PB |
| batt_target_choice | `batt_target_choice` | enum | 0=auto, 1=custom | O | B, PB |
| batt_target_power | `batt_target_power` | kW array | — | O | B, PB |

### 4.10 Load (for self-consumption dispatch)

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| load_profile | `load` | kW array (8760) | — | R (self-consump.) | B, PB |
| crit_load | `crit_load` | kW array | — | O | B, PB |

### 4.11 Tariff / Electricity Rates

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| ur_en_ts_buy_rate | `ur_en_ts_buy_rate` | bool | False | O | B, PB |
| ur_ts_buy_rate | `ur_ts_buy_rate` | $/kWh array | — | O | B, PB |
| ur_ts_sell_rate | `ur_ts_sell_rate` | $/kWh array | — | O | B, PB |
| ur_ec_sched_weekday | `ur_ec_sched_weekday` | 12×24 matrix | — | O | B, PB |
| ur_ec_sched_weekend | `ur_ec_sched_weekend` | 12×24 matrix | — | O | B, PB |
| ur_ec_tou_mat | `ur_ec_tou_mat` | matrix | — | O | B, PB |
| ur_dc_enable | `ur_dc_enable` | bool | False | O | B, PB |
| ur_dc_sched_weekday | `ur_dc_sched_weekday` | 12×24 matrix | — | O | B, PB |
| ur_dc_tou_mat | `ur_dc_tou_mat` | matrix | — | O | B, PB |

### 4.12 Lifetime / Degradation

| Business Name | SAM Variable | Units | Default / Range | R/O | Config |
|---------------|-------------|-------|-----------------|-----|--------|
| analysis_period | `analysis_period` | years | 25–35 | O | PV, B, PB |
| system_use_lifetime_output | `system_use_lifetime_output` | bool | False | O | PV, B, PB |
| dc_degradation | `dc_degradation` | %/year | 0.5–0.7 | O | PV, PB |
| en_dc_lifetime_losses | `en_dc_lifetime_losses` | bool | False | O | PV, PB |
| dc_lifetime_losses | `dc_lifetime_losses` | %/year array | — | O | PV, PB |

---

## Phase 5 — Financial Metrics

### Overview

None of the four target metrics (IRR, NPV, LCOE, LCOS) are currently computed by pvsamlab. The library returns only annual energy output and loss percentages. To compute financial metrics, either:
1. Attach a SAM financial module (`SingleOwner`, `Cashloan`, `Lcoefcr`) to the simulation, or
2. Compute post-simulation in Python using SAM energy outputs + user-supplied financial parameters.

The table below describes each metric.

---

### 5.1 IRR — Internal Rate of Return

**Definition:** Discount rate at which NPV = 0 over the project lifetime.

**SAM native computation?**
Yes — `PySAM.Cashloan` (residential/commercial) and `PySAM.SingleOwner` (utility-scale) both compute after-tax IRR natively via `output.project_return_aftertax_irr`.

**Inputs required:**

| Input | SAM Variable / Source | Units | Notes |
|-------|-----------------------|-------|-------|
| PV CAPEX | `total_installed_cost` | $ | Modules + inverter + BOS + EPC |
| BESS CAPEX | `batt_computed_bank_capacity × batt_cost_per_kwh` | $ | Separate from PV |
| Annual OPEX | `om_fixed` or `om_production` | $/year or $/kWh | O&M, insurance, land |
| Debt fraction | `debt_fraction` | % | 0–100 |
| Loan interest rate | `loan_rate` | % | |
| Loan term | `loan_term` | years | |
| Tax rate (federal + state) | `federal_tax_rate`, `state_tax_rate` | % | |
| Depreciation schedule | `depr_fed_type`, `depr_sta_type` | enum | MACRS-5 typical for US |
| ITC / PTC | `itc_fed_amount` or `ptc_fed_amount` | % or $/kWh | |
| Revenue / PPA rate | `ppa_price_input` or `ur_ts_sell_rate` | $/kWh | |
| Analysis period | `analysis_period` | years | |
| Annual energy | `annual_energy` (from Pvsamv1) | kWh/year | Passed via `gen` output array |

**SAM output:** `SingleOwner.Outputs.project_return_aftertax_irr` (%)

---

### 5.2 NPV — Net Present Value

**Definition:** Present value of all after-tax cash flows discounted at the required rate of return.

**SAM native computation?**
Yes — `PySAM.SingleOwner` computes NPV as `output.project_return_aftertax_npv`.

**Additional inputs beyond IRR:**

| Input | SAM Variable | Units | Notes |
|-------|-------------|-------|-------|
| Discount rate | `real_discount_rate` | % | Real (inflation-adjusted) discount rate |
| Inflation rate | `inflation_rate` | % | |
| Annual price escalation | `ppa_escalation` or `ur_annual_escalation` | % | |

**SAM output:** `SingleOwner.Outputs.project_return_aftertax_npv` ($)

**Post-simulation alternative (if not using financial module):**

```python
def npv(cash_flows: list[float], discount_rate: float) -> float:
    return sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows, 1))

# cash_flows[t] = revenue[t] - opex[t] - debt_service[t] + tax_benefits[t]
```

---

### 5.3 LCOE — Levelized Cost of Energy

**Definition:** Total life-cycle cost ($ NPV) ÷ total discounted energy production (kWh NPV).

**SAM native computation?**
Yes — `PySAM.Lcoefcr` computes LCOE directly (simplified fixed-charge-rate method, suitable for utility-scale).
`PySAM.SingleOwner` also outputs `lcoe_real` and `lcoe_nom`.

**Inputs required:**

| Input | SAM Variable | Units | Notes |
|-------|-------------|-------|-------|
| Total installed cost | `total_installed_cost` | $ | All-in CAPEX |
| Fixed annual O&M | `om_fixed` | $/year | |
| Variable O&M | `om_production` | $/kWh | |
| FCR (fixed charge rate) | `capital_cost_multiplier` | — | If using Lcoefcr |
| Discount rate | `real_discount_rate` | % | |
| Analysis period | `analysis_period` | years | |
| Degradation rate | `dc_degradation` | %/year | |
| Annual energy year 1 | `annual_energy` | kWh | From Pvsamv1 output |

**SAM output:**
- `Lcoefcr.Outputs.lcoe_real` (¢/kWh, real)
- `SingleOwner.Outputs.lcoe_real` (¢/kWh, real)
- `SingleOwner.Outputs.lcoe_nom` (¢/kWh, nominal)

**Post-simulation formula:**

```
LCOE = (CAPEX × FCR + annual_OPEX) / annual_energy_year1
```

Where FCR = `discount_rate / (1 - (1 + discount_rate)^-n)` for simplified calculation.

---

### 5.4 LCOS — Levelized Cost of Storage

**Definition:** Total life-cycle cost of the BESS ÷ total energy discharged over project life.

**SAM native computation?**
**No** — SAM does not compute LCOS natively. Must be computed post-simulation.

**Inputs required:**

| Input | Source | Units | Notes |
|-------|--------|-------|-------|
| BESS CAPEX (energy) | user input | $/kWh | Typically $150–$300/kWh (2025) |
| BESS CAPEX (power) | user input | $/kW | Converters, BOS |
| Battery replacement cost | user input | $/kWh | Triggered at EOL capacity threshold |
| Replacement year(s) | `batt_replacement_schedule_percent` | year | When SOH < 80% |
| Annual BESS O&M | user input | $/kWh-installed/year | ~$5–$10/kWh |
| Discount rate | user input | % | Same as system-level |
| Annual energy discharged | `batt_annual_discharge_energy` (SAM output) | kWh/year | From Pvsamv1 or StandAloneBattery |
| Battery degradation profile | `batt_SOC_percent`, `batt_capacity_percent` | — | From lifetime simulation |
| Total throughput | `batt_annual_discharge_energy` summed over life | kWh | |

**Post-simulation formula:**

```python
def lcos(
    batt_capex: float,          # $ (total battery system cost)
    annual_opex: float,         # $/year
    replacement_costs: list,    # [(year, cost_$), ...] for each replacement event
    annual_discharge: list,     # kWh/year for each year of project life
    discount_rate: float,       # decimal
) -> float:
    """Returns LCOS in $/kWh."""
    n = len(annual_discharge)
    pv_costs = batt_capex + sum(
        annual_opex / (1 + discount_rate)**t for t in range(1, n + 1)
    ) + sum(
        cost / (1 + discount_rate)**yr for yr, cost in replacement_costs
    )
    pv_discharge = sum(
        kwh / (1 + discount_rate)**t for t, kwh in enumerate(annual_discharge, 1)
    )
    return pv_costs / pv_discharge if pv_discharge > 0 else float('inf')
```

**Key SAM outputs needed:**

| SAM Output Variable | Description |
|--------------------|-------------|
| `batt_annual_discharge_energy` | kWh discharged per year (array over project life) |
| `batt_annual_charge_energy` | kWh charged per year |
| `batt_capacity_percent` | SOH per year (for replacement trigger) |
| `batt_cycles` | Total cycles per year |
| `batt_roundtrip_efficiency` | Realized roundtrip efficiency |

---

### 5.5 Financial Module Integration Plan

```
pvsamlab/financial.py

@dataclass
class Financial:
    analysis_period: int = 25             # years
    discount_rate: float = 8.0            # % real
    inflation_rate: float = 2.5           # %
    federal_tax_rate: float = 21.0        # %
    state_tax_rate: float = 0.0           # %
    debt_fraction: float = 70.0           # %
    loan_rate: float = 5.0                # %
    loan_term: int = 18                   # years
    pv_capex_per_kwdc: float = 700.0      # $/kWdc
    bess_capex_per_kwh: float = 250.0     # $/kWh
    bess_capex_per_kw: float = 150.0      # $/kW
    opex_per_kwac_year: float = 15.0      # $/kWac/year (PV)
    opex_bess_per_kwh_year: float = 8.0   # $/kWh/year (BESS)
    ppa_rate: float = 40.0                # $/MWh
    ppa_escalation: float = 1.0           # %/year
    itc_rate: float = 30.0                # % (IRA ITC)
    degradation_rate: float = 0.5         # %/year

# Attach to simulation:
class PvSystem:
    ...
    financial: Financial = field(default_factory=Financial)

    def compute_financials(self) -> dict:
        # 1. Build gen array from model_results['annual_energy'] + degradation
        # 2. Attach SingleOwner or Cashloan to the Pvsamv1 instance
        # 3. Execute financial model
        # 4. Return IRR, NPV, LCOE
        ...

    def compute_lcos(self) -> float:
        # Post-simulation LCOS calculation
        # Only valid for B or PB configurations
        ...
```

**SAM financial modules by use case:**

| Use Case | SAM Module | Key Outputs |
|----------|-----------|-------------|
| Utility-scale PV/BESS (PPA) | `PySAM.SingleOwner` | `lcoe_real`, `lcoe_nom`, `project_return_aftertax_irr`, `project_return_aftertax_npv` |
| Commercial C&I | `PySAM.Cashloan` | `payback`, `discounted_payback`, `after_tax_irr` |
| Simple LCOE | `PySAM.Lcoefcr` | `lcoe_real`, `lcoe_nom` |
| LCOS | Post-simulation Python | Custom (see formula above) |

---

## Summary — Critical Actions

| Priority | Action |
|----------|--------|
| P0 | Fix logic bug in `parallel_string_sizing.py` — function returns early, caller gets wrong number of values |
| P0 | Delete or fix `models.py` (ImportError on `fetch_weather_file`) |
| P0 | Add null-check after `download_nsrdb_csv()` return in `system.py` |
| P0 | Fix `MlModelParameters.__post_init__` — assignments to locals, not `self` |
| P0 | Remove hardcoded API credentials from `utils.py` |
| P1 | Decouple `System` construction from simulation execution (add `run()` method) |
| P1 | Make `tracking_mode` propagate to subarrays 2–4 in `generate_pysam_inputs()` |
| P1 | Fix duplicate try/except dead code in `Inverter.from_ond()` |
| P1 | Integrate `Losses` dataclass into `system.py`'s `generate_pysam_inputs()` |
| P1 | Fix `pyproject.toml` to list all runtime dependencies |
| P1 | Update `.gitignore` and purge committed artifacts (`df.csv`, result CSVs, tmp JSONs) |
| P2 | Write real `README.md` with installation + usage examples |
| P2 | Add docstrings to all public classes and functions |
| P2 | Write meaningful integration tests against a fixed weather file |
| P3 | Add `financial.py` with `Financial` dataclass + LCOE/LCOS computation |
| P3 | Add `battery.py` and extend to BESS-only and PV+BESS configurations |

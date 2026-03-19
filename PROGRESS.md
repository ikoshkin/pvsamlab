# Refactor Progress

**Branch:** `refactor/p0-fixes`
**Base:** `main`
**Status:** Pushed to origin. PR pending manual creation.

---

## Commits (oldest → newest)

| SHA | Message |
|-----|---------|
| `f5837bc` | Delete models.py — broken and superseded by system.py |
| `1a7be07` | Fix logic bug in parallel_string_sizing.py: early return + undefined summary |
| `c0a6925` | Add null-check in system.py after download_nsrdb_csv() |
| `334f124` | Fix MlModelParameters.__post_init__: assign to self, remove dead code and unused import |
| `dbd3045` | Remove hardcoded API credentials from utils.py |
| `4da677a` | Decouple System construction from simulation — add System.run() |
| `d3d759f` | Propagate tracking_mode to subarrays 2-4; convert InitVar to regular field |
| `4978078` | Integrate Losses dataclass into generate_pysam_inputs() |
| `666b38e` | Fix pyproject.toml: add all runtime dependencies |

---

## P0 Fixes Applied

1. **Delete `models.py`** — crashed on import (`fetch_weather_file` missing from `utils.py`); fully superseded by `system.py`, `components.py`, and `climate.py`.

2. **Fix `parallel_string_sizing.py` logic bug** — `return {...}` at line ~45 made the remainder of `run_simulation()` dead code. `summary`, `df_monthly`, and `df_hourly` were never built, causing `NameError` on the return statement. Fixed to assign `summary = {...}`, build all three outputs, and always return a 3-tuple. Guarded `pd.concat` calls with `if monthly_rows:` / `if hourly_rows:`. Fixed deprecated `freq='H'` → `freq='h'`.

3. **Null-check in `system.py`** — `download_nsrdb_csv()` can return `None` on network/API failure. `open(None)` raised an opaque `TypeError`. Added explicit check raising `RuntimeError` with coordinates and year.

4. **Fix `MlModelParameters.__post_init__`** — four fields (`AM_c_sa`, `AM_c_lp`, `IAM_c_cs_incAngle`, `IAM_c_cs_iamValue`) were assigned to local variables, never stored on `self`. Fixed to `self.*` assignments. Also removed duplicate unreachable `try/except` block in `Inverter.from_ond()` and unused `parse_pan_file` import from `components.py`.

5. **Remove hardcoded API credentials** — `api_key`, `your_name`, `email` were literal strings in `utils.py`. Replaced with `os.getenv('NSRDB_API_KEY')`, `os.getenv('NSRDB_API_NAME')`, `os.getenv('NSRDB_API_EMAIL')`. Credentials now loaded from `pvsamlab/secrets.env` via `python-dotenv`.

---

## P1 Fixes Applied

1. **Decouple `System` construction from simulation** — `__post_init__` previously called `model.assign`, `model.execute`, and `process_outputs` immediately on construction, making `System(...)` a blocking simulation call with no way to inspect inputs first. Moved all three calls into a new `System.run() -> dict` method. Added `model_results: dict = field(default=None, init=False)` to store results. Updated `__main__` block and `parallel_string_sizing.py` to call `plant.run()`.

2. **Propagate `tracking_mode` to subarrays 2–4** — subarrays 2, 3, and 4 had `track_mode` hardcoded to `TrackingMode.SAT` regardless of user input. Fixed to use `plant.tracking_mode`. Also converted `tracking_mode` from `InitVar[int]` (never stored as attribute) to a regular `int` field so `plant.tracking_mode` is accessible everywhere.

3. **Integrate `Losses` dataclass** — added a new `Losses` dataclass (64 fields across 4 subarrays) with defaults matching the previous hardcoded values. Added `losses: Losses = field(default_factory=Losses)` to `System`. Replaced 28 hardcoded loss literals in `generate_pysam_inputs()` with reads from `plant.losses.*`, making all loss parameters user-configurable without changing default behavior.

4. **Fix `pyproject.toml` runtime dependencies** — `tqdm` was in `[project.dependencies]` (runtime) but is only needed for the dev example script. Moved to `[project.optional-dependencies] dev`. Added all missing runtime dependencies: `scipy`, `pvlib`, `python-dotenv`, `requests`.

---

---

## Phase 3 — BESS Extension (branch: `feature/bess`)

### Current state

- Branch: `feature/bess`
- Last action: `examples/` folder reorganization into per-tool subfolders with `outputs/`, READMEs, and `CONTRIBUTING.md`

### Completed this session

- BESS Phase 3 full implementation: `battery.py`, `financial.py`, `PvBessSystem`, `StandaloneBessSystem`
- `bess_sandbox.ipynb` — all 5 cells running clean
- `pv_bess_sizing_study` — 25-case matrix with heatmaps
- `pv_bess_dispatch_analysis` — 6 charts + interactive widget
- `bess_surplus_optimization` — 75-case matrix (25 sizes × 3 charging modes) with merchant price curve and interactive dashboard
- `examples/` folder reorganized into per-tool structure
- `CONTRIBUTING.md` and `README.md` for each tool added

### Known issues / pending validation

- `price_signal` dispatch behavior needs validation with real `DAMPriceExample.csv` — confirm battery is actually responding to price signal not flat dispatch
- Negative net export in winter months (monthly waterfall) needs investigation — likely grid charging issue
- `batt_dispatch_auto_can_gridcharge=0` fix not yet verified in surplus notebook

### Next session priorities (in order)

1. Validate merchant price curve dispatch behavior
2. Fix winter negative export / grid charging issue
3. Add SoH lifetime tracking (`system_use_lifetime_output=1`, `analysis_period=25`, `batt_capacity_percent` per-year array)
4. Equipment database — `Battery.from_db()` classmethod with CSV of OEM specs (same pattern as `Module.from_pan`)
5. Merge `feature/bess` to main via PR

### File locations after reorganization

```
examples/
  string_sizing/          — string sizing tool
  bess_sandbox/           — simulation verification
  bess_sizing_study/      — 25-case parametric study
  bess_dispatch_analysis/ — operational visualizations
  bess_surplus/           — brownfield optimization tool
  sandbox/                — exploration notebooks
  CONTRIBUTING.md         — folder conventions
```

### Implemented

| File | Status |
|------|--------|
| `pvsamlab/financial.py` | Done — `Financial`, `RevenueStack`, `compute_lcoe` (with RevenueStack), `compute_lcos`, `compute_npv`, `compute_irr` |
| `pvsamlab/battery.py` | Done — `Battery`, `BessDispatch`, `PvBessSystem`, `StandaloneBessSystem`, `process_bess_outputs` (with `batt_capacity_percent` array) |
| `pvsamlab/__init__.py` | Done — all BESS + Financial + RevenueStack symbols exported |
| `BESS_DESIGN.md` | Done — updated with all 5 pending design items |
| `examples/bess_sandbox/bess_sandbox.ipynb` | Done — Cells 1–5 verified; Cells 6–7 added |
| `examples/bess_sizing_study/pv_bess_sizing_study.ipynb` | Done — 25-case matrix |
| `examples/bess_dispatch_analysis/pv_bess_dispatch_analysis.ipynb` | Done — 6 charts + widget |
| `examples/bess_surplus/bess_surplus_optimization.ipynb` | Done — 75-case dashboard |

### Verified simulation results (Cells 1–5, commit `120d4ac`)

| Cell | Metric | Value |
|------|--------|-------|
| Cell 3 — PV+BESS | Annual discharge | 43.9 M kWh |
| Cell 3 — PV+BESS | Annual charge | 45.3 M kWh |
| Cell 3 — PV+BESS | Roundtrip efficiency | 97% |
| Cell 4 — Standalone BESS | Annual discharge | 87.5 M kWh |
| Cell 4 — Standalone BESS | Annual charge | 95.7 M kWh |
| Cell 4 — Standalone BESS | Roundtrip efficiency | 91.5% |
| Cell 5 — Financial | PV-only IRR | 12.82% |
| Cell 5 — Financial | LCOS | $0.37 /kWh |

### Key PySAM discoveries this session

- `StandaloneBatteryResidential` clamps to 12 kWh / 5 kW — use `StandaloneBatterySingleOwner` for utility scale
- `dispatch_manual_sched` overrides `batt_dispatch_choice` silently — only assign when `strategy == 'manual'`
- `from_existing()` without config name shares C pointer
- `FlatPlatePVSingleOwner` default has `debt_option=1` (DSCR) — set `debt_option=0` for standard financing
- `batt_ac_or_dc` forced to 1 for `StandaloneBessSystem`
- `en_batt=1` must be set before `BatterySystem` group assign
- `price_signal` dispatch requires `PriceSignal.dispatch_factors_ts` (NOT `Revenue` or `ElectricityRates`)
- `PriceSignal.ppa_multiplier_model=1` must be set to enable 8760-element time-series
- `GridLimits.enable_interconnection_limit=1` + `GridLimits.grid_interconnection_limit_kwac` for POI enforcement

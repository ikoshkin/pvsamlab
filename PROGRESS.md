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

**Status:** `bess_sandbox.ipynb` runs all 5 cells clean as of commit `2a83202`.

### Implemented

| File | Status |
|------|--------|
| `pvsamlab/financial.py` | Done — `Financial`, `compute_lcoe`, `compute_lcos`, `compute_npv`, `compute_irr` |
| `pvsamlab/battery.py` | Done — `Battery`, `BessDispatch`, `PvBessSystem`, `StandaloneBessSystem`, `process_bess_outputs` |
| `pvsamlab/__init__.py` | Done — all BESS + Financial symbols exported |
| `examples/bess_sandbox.ipynb` | Done — executes clean, all 5 cells pass |

### Key PySAM 6 compatibility fixes applied (all in `2a83202`)

- `PvBessSystem` uses `pv.default("PVBatterySingleOwner")` as base model (all BatteryCell defaults pre-populated)
- `BessDispatch._STRATEGY_MAP`: `self_consumption=3` (not 1); `price_signal=4`
- Added `dispatch_manual_btm_discharge_to_grid`, `batt_dispatch_charge_only/discharge_only`, auto-dispatch vars
- `Lifetime` group (`analysis_period=1`, `system_use_lifetime_output=0`) required when `en_batt=1`
- `StandaloneBessSystem` uses `ba.default("StandaloneBatteryResidential")` (provides `timestep_minutes`)
- Standalone battery forced to `batt_ac_or_dc=1` (PySAM constraint: no-PV system must be AC-coupled)
- `Singleowner.from_existing(model)` without config arg shares C data pointer (gen available); import defaults separately
- `federal_tax_rate`, `state_tax_rate`, `ppa_price_input` expect arrays in Singleowner
- Added MACRS-5 allocation depreciation block to `_assign_single_owner`
- Added `construction_financing_cost=0`, `cost_other_financing=0` to `FinancialParameters`

### Pending design updates (requested, not yet implemented)

The following BESS_DESIGN.md updates are needed before further implementation:

1. `load_profile` optional, defaults to zeros for utility-scale
2. Add `RevenueStack` dataclass for merchant/capacity/ancillary revenue streams
3. Financial architecture split: SAM handles ITC/MACRS/debt; Python handles LCOS/revenue stacking
4. `Battery.coupling` maps to `batt_ac_or_dc` (0=DC, 1=AC); note capex difference between AC and DC coupling

### Next steps (in order)

1. Update `BESS_DESIGN.md` with above changes
2. Begin incremental implementation only after design is confirmed
3. Add `RevenueStack` to financial module
4. Make `load_profile` optional (default `[0.0]*8760`) in both `PvBessSystem` and `StandaloneBessSystem`

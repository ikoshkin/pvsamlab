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

## Next Steps

### Immediate
1. **Open PR** — navigate to:
   ```
   https://github.com/ikoshkin/pvsamlab/pull/new/refactor/p0-fixes
   ```
   Use title: `refactor: P0 + P1 fixes — correctness, safety, and API decoupling`

   PR body covers: deleted models.py, parallel_string_sizing bug fix, null-check, MlModelParameters fix, credential removal, System.run() decoupling, tracking_mode propagation, Losses dataclass, pyproject.toml cleanup.

### Feature Work (per AUDIT.md)

2. **Create `feature/bess` branch** from `main` (or from `refactor/p0-fixes` once merged):
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/bess
   ```

3. **Implement `pvsamlab/battery.py`** — classes per AUDIT.md Phase 3:
   - `Battery` dataclass: capacity_kwh, power_kw, roundtrip_efficiency, soc_min/max, chemistry
   - `BessDispatch` dataclass: dispatch strategy, charge/discharge windows, price thresholds
   - `BessSystem(System)`: extends System with battery; maps to PySAM `Battwatts` or `BatteryStateful`
   - `PvBessSystem`: combined PV + BESS simulation using `PySAM.Pvsamv1` + `PySAM.Battery`

4. **Implement `pvsamlab/financial.py`** — per AUDIT.md Phase 5:
   - `Financial` dataclass: capex, opex, degradation, discount_rate, offtake_price, incentives
   - `calculate_lcoe(system, financial) -> float`: uses PySAM `Utilityrate5` + `Cashloan`
   - `calculate_lcos(bess_system, financial) -> float`: levelized cost of storage
   - `calculate_npv(cashflows, discount_rate) -> float`
   - `calculate_irr(cashflows) -> float`

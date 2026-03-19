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

**Status:** All design updates implemented and committed as of commit `bf4bb04`. Cells 6 and 7 added to sandbox (not yet executed).

### Implemented

| File | Status |
|------|--------|
| `pvsamlab/financial.py` | Done — `Financial`, `RevenueStack`, `compute_lcoe` (with RevenueStack), `compute_lcos`, `compute_npv`, `compute_irr` |
| `pvsamlab/battery.py` | Done — `Battery`, `BessDispatch`, `PvBessSystem`, `StandaloneBessSystem`, `process_bess_outputs` (with `batt_capacity_percent` array) |
| `pvsamlab/__init__.py` | Done — all BESS + Financial + RevenueStack symbols exported |
| `BESS_DESIGN.md` | Done — updated with all 5 pending design items |
| `examples/bess_sandbox.ipynb` | Done — Cells 1–5 verified; Cells 6–7 added (pending execution) |
| `examples/bess_sandbox_executed.ipynb` | Cells 1–5 snapshot committed |

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

### Design updates applied (commits `ffac6dd`, `0863669`, `bf4bb04`)

1. **`load_profile` optional** — defaults to `[0.0]*8760` in both `PvBessSystem` and `StandaloneBessSystem` (documented in BESS_DESIGN.md).
2. **`RevenueStack` dataclass** — added to `financial.py` and exported from `__init__.py`. Fields: `energy_arbitrage_prices` (8760 $/MWh array), `capacity_payment_per_kw_year`, `ancillary_services_per_kw_year`, `capacity_kw`.
3. **Financial architecture split confirmed** — SAM Singleowner handles ITC/MACRS/debt/LCOE/IRR; Python handles LCOS and capacity/ancillary NPV bonus via `_npv_of_annuity()`.
4. **`Battery.coupling` → `batt_ac_or_dc` mapping** — `"DC"→0`, `"AC"→1`; `StandaloneBessSystem` always forces AC (documented in BESS_DESIGN.md).
5. **`process_bess_outputs` returns `batt_capacity_percent` array** — full per-year SOH list added to output dict.

### Additional battery.py fixes (commit `bf4bb04`)

- `batt_dispatch_auto_can_gridcharge` now derived from `any(disp.can_gridcharge)` instead of hardcoded 0. Default `BessDispatch` has `can_gridcharge=[0]*6`, so existing behavior unchanged. Set `can_gridcharge=[1]*6` for arbitrage dispatch.
- `_last_output` helper removed (now unused).

### Cell 6 — Merchant price curve (commit `bf4bb04`, pending execution)

- Part A: `StandaloneBessSystem` with `price_signal` dispatch (choice=4) + `can_gridcharge=[1]*6` for grid arbitrage; flat $35/MWh placeholder (swap in real CSV column for actual market prices).
- Part B: `compute_lcoe(pvbess_plant, fin, revenue_stack=rev)` with `RevenueStack(energy_arbitrage_prices=prices, capacity_payment_per_kw_year=80.0, ancillary_services_per_kw_year=15.0)`.
- Compares NPV vs Cell 5 flat PPA case.

### Cell 7 — 100 MW / 4-HR sizing loop (commit `bf4bb04`, pending execution)

- Sweeps `energy_kwh` from 400,000 to 550,000 kWh, step 10,000.
- Projects SOH using linear calendar degradation (2%/yr); finds first year < 100%.
- Computes LCOS for each size.
- Reports table `installed_MWh | years_at_nameplate | LCOS` and highlights minimum size for `years_at_nameplate >= 10`.

### Key PySAM 6 compatibility notes

**Battery dispatch:**
- `PvBessSystem` uses `pv.default("PVBatterySingleOwner")` as base model
- `BessDispatch._STRATEGY_MAP`: `self_consumption=3`; `price_signal=4`
- `Lifetime` group (`analysis_period=1`, `system_use_lifetime_output=0`) required when `en_batt=1`
- `batt_replacement_option=0` required when `system_use_lifetime_output=0`
- `dispatch_manual_sched*` must NOT be set for non-manual strategies

**Standalone battery:**
- Must use `ba.default("StandaloneBatterySingleOwner")` — Residential default clamps to residential scale
- Both `batt_power_charge_max_kwac` and `batt_power_discharge_max_kwac` required for AC-coupled battery

**Financial (Singleowner):**
- `Singleowner.from_existing(model)` without config arg shares C data pointer (gen available)
- `federal_tax_rate`, `state_tax_rate`, `ppa_price_input` expect arrays
- `fp.debt_option=0` required — FlatPlatePVSingleOwner default is DSCR-based (ignores debt_percent)

---

## Phase 3 Complete — Summary of all notebooks (branch: `feature/bess`)

### Notebooks delivered

| Notebook | Content | Status |
|----------|---------|--------|
| `examples/bess_sandbox.ipynb` | 7-cell sandbox: PV+BESS, standalone BESS, financial, merchant price curve, sizing loop | Complete |
| `examples/pv_bess_sizing_study.ipynb` | 5×5 BESS sizing matrix (25 cases), parallel sweep, 2×2 heatmaps, ranked table | Complete |
| `examples/pv_bess_dispatch_analysis.ipynb` | 6 deep-dive charts (weekly dispatch, SOC heatmap, duration curves, monthly waterfall, hourly heatmap, utilization) + 25-case widget | Complete |
| `examples/bess_surplus_optimization.ipynb` | 75-case (25 sizes × 3 charging modes) brownfield BESS optimization, IRR heatmaps, revenue waterfall, price-dispatch scatter + comprehensive interactive dashboard | Complete |

### Static chart files (generated by runner scripts, not tracked in git)

- `dispatch_weekly.png`, `soc_heatmap.png`, `duration_curves.png`, `monthly_waterfall.png`, `dispatch_hourly_heatmap.png`, `utilization_analysis.png` — from `run_dispatch_analysis.py`
- `surplus_irr_heatmap.png`, `surplus_revenue_waterfall.png`, `surplus_price_dispatch.png` — from `run_surplus_analysis.py`

### Source data file (tracked in git)

- `pvsamlab/data/DAMPriceExample.csv` — synthetic 2017 ERCOT West Texas DAM price curve (8760 hourly $/MWh values, avg $44/MWh)

### Key PySAM 6 discovery (this phase)

- `price_signal` dispatch (`batt_dispatch_choice=4`) requires `PriceSignal.dispatch_factors_ts` (NOT `Revenue.dispatch_factors_ts` or `ElectricityRates`)
- `PriceSignal.ppa_multiplier_model=1` must be set to enable the 8760-element time-series
- `GridLimits.enable_interconnection_limit=1` + `GridLimits.grid_interconnection_limit_kwac` for POI enforcement
- With proper setup, price_signal produces realistic arbitrage: ~155 GWh/yr discharge for 400 MWh/100 MW battery

### Dashboard features (bess_surplus_optimization.ipynb Cell 9)

- Row 1: Case selection (power, duration, charging mode)
- Row 2: Financial sliders (CAPEX, capacity payment, discount rate) — recalculate NPV/IRR/LCOS without re-simulation
- Row 3: POI target, week selector, price curve toggle (Merchant / Flat $45 / TOU)
- 4-panel output: KPI box, dispatch stack, SOC heatmap, price-dispatch scatter
- Breakeven button: iterate capacity payment 0→150 $/kW-yr, find NPV=0 crossing
- Pareto frontier button: IRR vs NPV scatter for all 75 cases, Pareto-optimal highlighted
- Export button: dump filtered results to CSV

### Next steps

1. **SoH lifetime tracking** — extend `process_bess_outputs` to project degradation over full analysis period
2. **Equipment database integration** — link `Battery` dataclass to real cell/module datasheets from `pvsamlab/data/`
3. **Real merchant curve validation** — replace `DAMPriceExample.csv` with actual ERCOT HB_WEST 2017 settlement prices
4. **Price-signal dispatch validation** — confirm `PriceSignal` group behavior matches SAM desktop for edge cases (negative prices, gridcharge interactions)

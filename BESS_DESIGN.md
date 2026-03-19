# BESS Extension — Implementation Design

**Branch:** `feature/bess`
**Status:** Phase 3 implemented and verified. Design updated post-implementation.
**Prerequisite:** All P0 + P1 fixes merged to `main` ✓

---

## 0. Guiding Constraints

1. **Do not touch existing PV-only behavior.** `System`, `generate_pysam_inputs()`, and
   `process_outputs()` in `system.py` are off-limits for modification (only additive changes
   are acceptable, e.g. new commented-out battery groups that already exist).
2. New classes **inherit from or compose with** `System`; they do not replace it.
3. All new public symbols will be exported from `pvsamlab/__init__.py` alongside the
   existing ones.

---

## 1. `pvsamlab/battery.py`

### 1.1 `Battery` Dataclass

Pure data; no PySAM dependency.

```python
@dataclass
class Battery:
    # Size
    energy_kwh: float = 4000.0          # Usable DC nameplate capacity
    power_kw: float = 1000.0            # Max charge / discharge power (DC)

    # Chemistry
    chemistry: str = "LFP"              # "LFP" | "NMC" | "lead_acid"
    batt_Vnom_default: float = 500.0    # Nominal DC bus voltage (V)

    # State-of-charge limits
    soc_min: float = 10.0               # % — min SOC (never discharged below)
    soc_max: float = 95.0               # % — max SOC (never charged above)
    soc_init: float = 50.0              # % — initial SOC

    # Efficiency
    roundtrip_efficiency: float = 87.5  # % DC-to-DC; used for reporting
    dc_ac_efficiency: float = 96.0      # % — inverter DC→AC
    ac_dc_efficiency: float = 96.0      # % — inverter AC→DC

    # Coupling
    coupling: str = "DC"                # "DC" | "AC"
                                        # Maps to batt_ac_or_dc: "DC"→0, "AC"→1
                                        # Note: StandaloneBessSystem always forces AC (batt_ac_or_dc=1)
                                        # because PySAM.Battery requires AC-coupled for no-PV systems

    # Degradation
    calendar_degradation: float = 2.0   # %/year (linear calendar fade)
    replacement_threshold: float = 80.0 # % SOH — replace battery when below

    # Costs (used by financial.py; not by PySAM directly)
    capex_per_kwh: float = 250.0        # $/kWh installed
    capex_per_kw: float = 150.0         # $/kW inverter/BOS
    opex_per_kwh_year: float = 8.0      # $/kWh/year O&M
```

**Chemistry → PySAM enum mapping** (stored as a class-level dict, not a field):

```python
CHEMISTRY_MAP = {"LFP": 1, "NMC": 0, "lead_acid": 2}
```

**Coupling → PySAM `batt_ac_or_dc` mapping:**

| `Battery.coupling` | PySAM `batt_ac_or_dc` | Notes |
|---|---|---|
| `"DC"` | `0` | Used for co-located PV+BESS (DC bus shared) |
| `"AC"` | `1` | Required for `StandaloneBessSystem` (no PV DC bus) |

### 1.2 `BessDispatch` Dataclass

```python
@dataclass
class BessDispatch:
    strategy: str = "manual"
    # Strategy → PySAM batt_dispatch_choice mapping:
    #   "manual"           → 0  (requires manual schedule vars)
    #   "self_consumption" → 3
    #   "peak_shaving"     → 2
    #   "price_signal"     → 4  (requires hourly price array; see notes below)

    # Manual schedule — 12 months × 24 hours, values = period index (1–6)
    # Default: charge periods 0–6, discharge periods 13–21, idle otherwise
    schedule_charge: List[List[int]] = field(default_factory=_default_charge_schedule)
    schedule_discharge: List[List[int]] = field(default_factory=_default_discharge_schedule)
    percent_discharge: List[float] = field(default_factory=lambda: [100.0] * 6)
    percent_charge: List[float] = field(default_factory=lambda: [100.0] * 6)
```

**Implementation notes:**
- `dispatch_manual_sched` and `dispatch_manual_sched_weekend` must **only** be assigned when
  `strategy == "manual"`. Assigning them with any other strategy overrides `batt_dispatch_choice`
  in the Battery module, regardless of its value.
- Manual dispatch (`choice=0`) is non-functional in `PySAM.Battery` standalone mode; use
  `self_consumption` (choice=3) for `StandaloneBessSystem`.
- `price_signal` (choice=4) requires an 8760-element price array in `batt_dispatch_auto_can_charge`
  and related fields.

### 1.3 `PvBessSystem(System)` — Co-located PV + BESS

Extends the existing `System` dataclass by adding battery and dispatch fields.
Uses **`PySAM.Pvsamv1` with `en_batt=1`** — the same model instance as PV-only but
with battery groups activated.

```python
@dataclass
class PvBessSystem(System):
    battery: Battery = field(default_factory=Battery)
    dispatch: BessDispatch = field(default_factory=BessDispatch)
    load_profile: List[float] = field(default_factory=lambda: [0.0] * 8760)
    # load_profile is optional for utility-scale; defaults to zeros (no behind-meter load).
    # Required for self_consumption and peak_shaving dispatch strategies.
```

Key design decisions:
- Inherits `System.run()` pattern — calls `model.assign(...)`, `model.execute()`,
  then `process_outputs()` + `process_bess_outputs()`.
- Overrides `generate_pysam_inputs()` to add battery groups on top of the base PV dict.
  Battery-specific groups: `BatterySystem`, `BatteryCell`, `BatteryDispatch`, `Load`,
  `SystemCosts`.
- `en_batt=1` is the single flag that activates battery simulation inside Pvsamv1.

### 1.4 `StandaloneBessSystem` — BESS-Only (no PV)

Standalone battery against a load profile. Uses **`PySAM.Battery`** (module name in PySAM 6).
Does **not** inherit from `System` (no PV hardware, no weather for irradiance).

```python
@dataclass
class StandaloneBessSystem:
    battery: Battery
    dispatch: BessDispatch
    load_profile: List[float] = field(default_factory=lambda: [0.0] * 8760)
    # load_profile defaults to zeros for utility-scale front-of-meter arbitrage.
    # Provide an 8760 kW array for behind-the-meter self_consumption dispatch.
    lat: float = LATITUDE
    lon: float = LONGITUDE
    weather_year: str = "tmy"
    analysis_period: int = 25

    model: Any = field(default=None, init=False)
    model_results: dict = field(default=None, init=False)

    def run(self) -> dict: ...
```

**PySAM 6 notes:**
- Must use `ba.default("StandaloneBatterySingleOwner")` — the `Residential` default
  (12 kWh / 5 kW) clamps power to residential scale even when `bank_capacity` is overridden.
- `batt_ac_or_dc=1` (AC-coupled) is required for no-PV systems; `Battery.coupling` field
  is ignored for `StandaloneBessSystem` and always overridden to AC.

---

## 2. `pvsamlab/financial.py`

### 2.1 `Financial` Dataclass

```python
@dataclass
class Financial:
    # Project timeline
    analysis_period: int = 25               # years
    degradation_rate: float = 0.5           # %/year PV output decay

    # Costs — PV
    pv_capex_per_kwdc: float = 700.0        # $/kWdc all-in
    opex_per_kwac_year: float = 15.0        # $/kWac/year (O&M + insurance + land)

    # Revenue
    ppa_rate: float = 40.0                  # $/MWh flat rate
    ppa_escalation: float = 1.0             # %/year

    # Financing
    discount_rate: float = 8.0             # % real (WACC)
    inflation_rate: float = 2.5            # %
    debt_fraction: float = 70.0            # %
    loan_rate: float = 5.0                 # % nominal
    loan_term: int = 18                    # years

    # Tax
    federal_tax_rate: float = 21.0         # %
    state_tax_rate: float = 0.0            # %
    itc_rate: float = 30.0                 # % investment tax credit (IRA)
    depreciation_schedule: str = "MACRS5"  # "MACRS5" | "straight_line" | "none"
```

### 2.2 `RevenueStack` Dataclass

Merchant / stacked revenue sources for batteries. Used as an optional argument to
`compute_lcoe()` to override or supplement the flat `ppa_rate` in `Financial`.

```python
@dataclass
class RevenueStack:
    # Energy arbitrage: 8760-element hourly price array ($/MWh)
    # If provided, replaces Financial.ppa_rate as the ppa_price_input sent to Singleowner.
    # If None, flat Financial.ppa_rate is used (standard PPA case).
    energy_arbitrage_prices: List[float] = None

    # Capacity market payment: $/kW-year × battery.power_kw
    # Added as a post-simulation Python NPV term (not routed through SAM).
    capacity_payment_per_kw_year: float = 0.0

    # Ancillary services: $/kW-year × battery.power_kw
    # Added as a post-simulation Python NPV term (not routed through SAM).
    ancillary_services_per_kw_year: float = 0.0

    # Reference power for capacity/ancillary payments; defaults to battery.power_kw if None.
    capacity_kw: float = None
```

**Design split:**
- SAM `Singleowner` handles: ITC, MACRS depreciation, debt service, after-tax cash flows,
  LCOE real/nominal, project IRR.
- Python handles: LCOS, capacity payments, ancillary payments, revenue stacking across
  multiple revenue streams.

When `energy_arbitrage_prices` is provided:
- Array is divided by 1000 ($/MWh → $/kWh) and passed as `ppa_price_input` to Singleowner.
- `ppa_escalation` from `Financial` is set to 0.0 (prices already time-varying).

When capacity/ancillary payments are non-zero:
- Annual revenue bonus = `(capacity_payment_per_kw_year + ancillary_services_per_kw_year) × capacity_kw`
- Bonus is added as a post-simulation NPV adjustment (not routed through Singleowner).

### 2.3 LCOE — `compute_lcoe(system, financial, revenue_stack=None) -> dict`

**Financial module selection:**
- `PySAM.Singleowner` for `system.kwac > 1000 kW` (utility-scale)
- `PySAM.Cashloan` for `system.kwac <= 1000 kW` (C&I / residential)

**Singleowner pattern (PySAM 6):**
```python
fin_model = so_mod.from_existing(system.model)          # shares C data pointer → gen accessible
_defaults = so_mod.default("FlatPlatePVSingleOwner").export()
_defaults.pop("SystemOutput", None)
_defaults.pop("Outputs", None)
fin_model.assign(_defaults)                              # import Singleowner defaults
fin_model.assign({...overrides...})                      # apply Financial fields
fin_model.execute()
```

**Key Singleowner field quirks (PySAM 6):**
- `federal_tax_rate`, `state_tax_rate`, `ppa_price_input` expect **arrays**, not scalars.
- `fp.debt_option = 0` required — FlatPlatePVSingleOwner default is `debt_option=1` (DSCR-based),
  which ignores `debt_percent` entirely regardless of what value is assigned.
- Depreciation uses allocation percentages (`depr_alloc_macrs_5_percent`, etc.), not `depr_fed_type`.

Returns:
```python
{
    "lcoe_real_cents_per_kwh": ...,
    "lcoe_nom_cents_per_kwh": ...,
    "npv_usd": ...,
    "irr_pct": ...,
    "total_installed_cost_usd": ...,
}
```

### 2.4 LCOS — `compute_lcos(battery, annual_discharge_kwh, annual_opex, replacement_events, discount_rate) -> float`

SAM does **not** compute LCOS natively. Always a post-simulation Python calculation.

```python
def compute_lcos(
    battery: Battery,
    annual_discharge_kwh: List[float],  # per-year array from SAM output
    annual_opex: float,                 # $/year total BESS O&M
    replacement_events: List[Tuple[int, float]],  # [(year, cost_$), ...]
    discount_rate: float,               # decimal
) -> float:
    """Returns LCOS in $/kWh."""
    batt_capex = battery.energy_kwh * battery.capex_per_kwh \
               + battery.power_kw  * battery.capex_per_kw
    n = len(annual_discharge_kwh)
    pv_costs = batt_capex \
        + sum(annual_opex / (1 + discount_rate)**t for t in range(1, n + 1)) \
        + sum(cost / (1 + discount_rate)**yr for yr, cost in replacement_events)
    pv_discharge = sum(
        kwh / (1 + discount_rate)**t
        for t, kwh in enumerate(annual_discharge_kwh, 1)
    )
    return pv_costs / pv_discharge if pv_discharge > 0 else float("inf")
```

### 2.5 IRR / NPV helpers (standalone, no SAM dependency)

Used for parameter sweeps, unit tests, and post-simulation revenue stacking.

```python
def compute_npv(cash_flows: List[float], discount_rate: float) -> float:
    return sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cash_flows))

def compute_irr(cash_flows: List[float]) -> float:
    # scipy.optimize.brentq on NPV(r) = 0
    ...
```

---

## 3. How `System` Is Extended for Each Mode

### 3.1 PV-Only (existing — no changes)

```
System  →  generate_pysam_inputs()  →  Pvsamv1(en_batt=0).assign().execute()
        →  process_outputs() → dict (energy + losses only)
```

### 3.2 PV + BESS (`PvBessSystem`)

```
PvBessSystem(System)
    ↳ _generate_battery_inputs()   — Battery + BessDispatch → battery groups dict
    ↳ generate_pysam_inputs()      — calls super() then merges battery groups
    ↳ Pvsamv1(en_batt=1).assign().execute()
    ↳ process_outputs()            — inherited, returns PV energy+losses dict
    ↳ process_bess_outputs()       — returns battery energy dict
    ↳ run() → merged dict
```

### 3.3 BESS-Only (`StandaloneBessSystem`)

```
StandaloneBessSystem
    ↳ _generate_bess_inputs()      — maps Battery + BessDispatch + load_profile
    ↳ Battery.assign().execute()
    ↳ process_bess_outputs()       — same function, different model handle
    ↳ run() → dict (battery outputs only, no PV keys)
```

### 3.4 Financial Overlay (all modes)

Financial metrics are intentionally **not** merged into `run()`. They require
additional user inputs (`Financial`) and involve a separate SAM module.

```python
plant = PvBessSystem(...)
results = plant.run()                     # energy simulation

fin = Financial(ppa_rate=45.0, itc_rate=30.0)
rev = RevenueStack(
    energy_arbitrage_prices=hourly_prices,
    capacity_payment_per_kw_year=80.0,
    ancillary_services_per_kw_year=15.0,
)
fin_results = compute_lcoe(plant, fin, revenue_stack=rev)  # Singleowner + Python overlay
lcos = compute_lcos(plant.battery, ...)                    # post-simulation Python

all_results = {**results, **fin_results, "lcos": lcos}
```

---

## 4. PySAM Modules by Mode

| Mode | PySAM Module | `en_batt` | Notes |
|------|-------------|-----------|-------|
| PV-only | `PySAM.Pvsamv1` | 0 (default) | Existing behavior — unchanged |
| PV + BESS | `PySAM.Pvsamv1` | 1 | Same model; battery groups activated |
| BESS-only | `PySAM.Battery` | n/a | No PV; `StandaloneBatterySingleOwner` default |
| LCOE/IRR/NPV | `PySAM.Singleowner` | n/a | Attached to Pvsamv1 via `from_existing()` |
| LCOE (C&I) | `PySAM.Cashloan` | n/a | Used for kwac ≤ 1000 kW |

### Active PySAM groups for PV+BESS (additive on top of PV-only)

| Group | Key variables |
|-------|--------------|
| `BatterySystem` | `en_batt=1`, `batt_chem`, `batt_computed_bank_capacity`, `batt_power_charge_max_kwdc`, `batt_power_discharge_max_kwdc`, `batt_Vnom_default`, `batt_ac_or_dc` |
| `BatteryCell` | Derived from chemistry; LFP defaults used initially |
| `BatteryDispatch` | `batt_dispatch_choice`, `batt_minimum_SOC`, `batt_maximum_SOC`, `batt_initial_SOC`; manual schedule vars **only** when `strategy == "manual"` |
| `Load` | `load` (8760 array) — assigned for all strategies (defaults to zeros) |
| `SystemCosts` | `om_batt_fixed_cost` — passed through from `Battery.opex_per_kwh_year × energy_kwh` |

---

## 5. `process_bess_outputs()` — Full Specification

### 5.1 Existing function — untouched

`process_outputs(plant: System) -> dict` in `system.py` stays exactly as-is.
Called by `PvBessSystem.run()` through `super()`.

### 5.2 `process_bess_outputs(model) -> dict` in `battery.py`

Takes the raw PySAM model object (either `Pvsamv1` or `Battery`),
reads battery-specific outputs:

```python
def process_bess_outputs(model) -> dict:
    return {
        "batt_annual_discharge_energy_kwh": round(sum(model.Outputs.batt_annual_discharge_energy), 3),
        "batt_annual_charge_energy_kwh":    round(sum(model.Outputs.batt_annual_charge_energy), 3),
        "batt_roundtrip_efficiency_pct":    round(model.Outputs.average_battery_roundtrip_efficiency, 2),
        "batt_capacity_end_of_life_pct":    round(model.Outputs.batt_capacity_percent[-1], 1),
        "batt_capacity_percent":            list(model.Outputs.batt_capacity_percent),
        "batt_cycles_total":                round(sum(model.Outputs.batt_cycles), 1),
    }
```

**`batt_capacity_percent` key (added):**
- Returns the **full per-year array** of battery State of Health (SOH) as a Python list.
- For single-year simulation (`system_use_lifetime_output=0`), this is a 1-element list `[100.0]`.
- For lifetime simulation, this is an `analysis_period`-length array (e.g. 25 elements).
- Used by the sizing loop (Cell 7) to find the first year SOH < 100% (i.e., degradation onset).
- `batt_capacity_end_of_life_pct` (scalar) is kept alongside for backward compatibility.

### 5.3 Financial outputs from `compute_lcoe(system, financial, revenue_stack=None) -> dict`

```python
{
    "lcoe_real_cents_per_kwh": ...,     # real, inflation-adjusted
    "lcoe_nom_cents_per_kwh": ...,      # nominal
    "npv_usd": ...,                     # after-tax NPV (+ Python capacity/ancillary bonus if provided)
    "irr_pct": ...,                     # after-tax IRR
    "total_installed_cost_usd": ...,    # CAPEX as computed
}
```

`compute_lcos()` returns a single `float` ($/kWh).

---

## 6. File Layout

```
pvsamlab/
├── __init__.py          — exports: PvBessSystem, StandaloneBessSystem, Battery,
│                          BessDispatch, Financial, RevenueStack, compute_lcoe,
│                          compute_lcos, compute_npv, compute_irr
├── system.py            — UNCHANGED (PV-only System)
├── battery.py           — Battery, BessDispatch, PvBessSystem, StandaloneBessSystem,
│                          process_bess_outputs, _generate_battery_inputs, _generate_bess_inputs
├── financial.py         — Financial, RevenueStack, compute_lcoe, compute_lcos,
│                          compute_npv, compute_irr
├── components.py        — UNCHANGED
├── climate.py           — UNCHANGED
└── utils.py             — UNCHANGED
```

---

## 7. Pending Implementation Items

### 7.1 `RevenueStack` integration (Step 2 — next)

Add `RevenueStack` dataclass to `financial.py`. Update `compute_lcoe()` signature:
```python
def compute_lcoe(system, financial, revenue_stack=None) -> dict:
```
- When `revenue_stack.energy_arbitrage_prices` is not None, use as `ppa_price_input` array.
- Add capacity + ancillary as post-simulation Python NPV addition.
Export `RevenueStack` from `__init__.py`.

### 7.2 Notebook Cell 6 — Merchant price curve (Step 3 — next after 7.1)

- Load prices from CSV if path provided, else use flat $35/MWh placeholder (8760 array).
- Run `StandaloneBessSystem` with `price_signal` dispatch.
- Run `compute_lcoe` with `RevenueStack(energy_arbitrage_prices=..., capacity_payment_per_kw_year=80.0, ancillary_services_per_kw_year=15.0)`.
- Print full financial results and compare NPV vs flat PPA case from Cell 5.

### 7.3 Notebook Cell 7 — 100MW/4HR nameplate sizing loop (Step 4 — next after 7.2)

- Sweep `energy_kwh` from 400,000 to 550,000 kWh, step 10,000.
- For each: run `StandaloneBessSystem`, check `batt_capacity_percent` per-year array,
  find first year SOH < 100%, compute LCOS.
- Print table: `installed_MWh | years_at_nameplate | LCOS`.
- Highlight minimum `installed_MWh` where `years_at_nameplate >= 10`.

---

## 8. Key PySAM 6 Compatibility Notes (Post-Implementation)

### Battery dispatch

- `PvBessSystem` uses `pv.default("PVBatterySingleOwner")` as base model
- `BessDispatch._STRATEGY_MAP`: `self_consumption=3`, `price_signal=4`
- `Lifetime` group (`analysis_period=1`, `system_use_lifetime_output=0`) required when `en_batt=1`
- `batt_replacement_option=0` required when `system_use_lifetime_output=0`
- `dispatch_manual_sched*` must NOT be set for non-manual strategies

### Standalone battery

- `StandaloneBessSystem` must use `ba.default("StandaloneBatterySingleOwner")` — the `Residential`
  default (12 kWh / 5 kW) clamps power to residential scale even when `bank_capacity` is overridden
- Both `batt_power_charge_max_kwac` and `batt_power_discharge_max_kwac` required for AC-coupled battery

### Financial (Singleowner)

- `Singleowner.from_existing(model)` without config arg shares C data pointer (gen available)
- `federal_tax_rate`, `state_tax_rate`, `ppa_price_input` expect arrays in Singleowner
- `fp.debt_option=0` required — `FlatPlatePVSingleOwner` default is `debt_option=1` (DSCR-based),
  which ignores `debt_percent` entirely regardless of what value is assigned

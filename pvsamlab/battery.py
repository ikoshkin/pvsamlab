"""
Battery dataclass and BESS simulation classes for pvsamlab.

Classes
-------
Battery             -- hardware spec (no PySAM dependency)
BessDispatch        -- dispatch strategy and manual schedules
PvBessSystem        -- co-located PV + BESS using Pvsamv1(en_batt=1)
StandaloneBessSystem -- BESS-only using PySAM.StandAloneBattery

Module-level functions
----------------------
process_bess_outputs(model) -> dict   -- extracts battery outputs from any
                                         executed PySAM model (Pvsamv1 or
                                         StandAloneBattery)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, ClassVar, List

import pandas as pd
import PySAM.Pvsamv1 as pv

from pvsamlab.system import (
    LATITUDE,
    LONGITUDE,
    System,
    generate_pysam_inputs,
    process_outputs,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_schedule() -> List[List[int]]:
    """12×24 period schedule with all hours mapped to period 1 (idle)."""
    return [[1] * 24 for _ in range(12)]


def _sum_output(val: Any) -> float:
    """Sum if iterable (lifetime arrays), cast directly if scalar."""
    try:
        return float(sum(val))
    except TypeError:
        return float(val)


def _last_output(val: Any) -> float:
    """Return last element if iterable, else the value itself."""
    try:
        seq = list(val)
        return float(seq[-1]) if seq else 0.0
    except (TypeError, IndexError):
        return float(val)


def _mean_output(val: Any) -> float:
    """Return mean if iterable, else the value itself."""
    try:
        seq = list(val)
        return float(sum(seq) / len(seq)) if seq else 0.0
    except (TypeError, ZeroDivisionError):
        return float(val)


# ---------------------------------------------------------------------------
# Battery hardware
# ---------------------------------------------------------------------------

@dataclass
class Battery:
    """Battery hardware specification.

    No PySAM dependency; shared by PvBessSystem and StandaloneBessSystem.
    All efficiency and degradation values are expressed as percentages.

    Chemistry to PySAM ``batt_chem`` mapping
    -----------------------------------------
    Both "LFP" and "NMC" map to ``batt_chem=1`` (lithium-ion) in PySAM's
    simplified battery model.  Detailed cell parameters (voltage, capacity
    curves) for each Li-ion chemistry are outside the scope of this release.
    """

    CHEMISTRY_MAP: ClassVar[dict] = {"LFP": 1, "NMC": 1, "lead_acid": 0}

    # Chemistry
    chemistry: str = "LFP"                  # "LFP" | "NMC" | "lead_acid"

    # Size
    energy_kwh: float = 4000.0              # Usable DC nameplate capacity (kWh)
    power_kw: float = 1000.0               # Max charge / discharge power DC (kW)

    # Bus voltage
    batt_Vnom_default: float = 500.0        # Nominal DC bank voltage (V)

    # State-of-charge limits
    soc_min: float = 10.0                   # % — lower bound
    soc_max: float = 95.0                   # % — upper bound
    soc_init: float = 50.0                  # % — starting SOC

    # Efficiency
    roundtrip_efficiency: float = 87.5      # % DC-to-DC (informational)
    dc_ac_efficiency: float = 96.0          # % DC → AC
    ac_dc_efficiency: float = 96.0          # % AC → DC

    # Coupling
    coupling: str = "DC"                    # "DC" | "AC"

    # Degradation
    calendar_degradation: float = 2.0       # %/year (linear calendar reference)
    replacement_threshold: float = 80.0     # % SOH — replace below this level

    # Costs (used by financial.compute_lcos; not passed to PySAM)
    capex_per_kwh: float = 250.0            # $/kWh installed energy capacity
    capex_per_kw: float = 150.0             # $/kW inverter + power electronics BOS
    opex_per_kwh_year: float = 8.0          # $/kWh/year fixed O&M


# ---------------------------------------------------------------------------
# Battery dispatch
# ---------------------------------------------------------------------------

@dataclass
class BessDispatch:
    """Battery dispatch configuration.

    Strategy
    --------
    "manual"           -- 12×24 period schedules + per-period charge/discharge flags
    "self_consumption" -- automated: maximise on-site solar consumption
    "peak_shaving"     -- automated look-ahead peak shaving
    "price_signal"     -- front-of-meter wholesale price arbitrage

    Manual mode details
    -------------------
    schedule_weekday / schedule_weekend
        12×24 matrices mapping each hour of each month to a period index (1–6).
        Default: all hours → period 1.
    can_charge / can_discharge / can_gridcharge
        6-element boolean arrays, one flag per period.
    percent_discharge / percent_gridcharge
        6-element float arrays giving the fraction of max power to use per period.
    """

    _STRATEGY_MAP: ClassVar[dict] = {
        "manual": 0,
        "self_consumption": 1,
        "peak_shaving": 2,
        "price_signal": 3,
    }

    strategy: str = "manual"

    # Manual schedules — 12 months × 24 hours (period index 1–6)
    schedule_weekday: List[List[int]] = field(default_factory=_default_schedule)
    schedule_weekend: List[List[int]] = field(default_factory=_default_schedule)

    # Per-period dispatch options (one entry per period 1–6)
    can_charge: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    can_discharge: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    can_gridcharge: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    percent_discharge: List[float] = field(default_factory=lambda: [100.0] * 6)
    percent_gridcharge: List[float] = field(default_factory=lambda: [100.0] * 6)

    def pysam_dispatch_choice(self) -> int:
        """Return the PySAM ``batt_dispatch_choice`` integer for this strategy."""
        return self._STRATEGY_MAP.get(self.strategy, 0)


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

def process_bess_outputs(model: Any) -> dict:
    """Extract battery simulation results from an executed PySAM model.

    Compatible with both ``PySAM.Pvsamv1`` (en_batt=1) and
    ``PySAM.StandAloneBattery`` output groups.  Handles both single-year
    (``system_use_lifetime_output=0``) and lifetime output modes.

    Parameters
    ----------
    model
        An already-executed PySAM model instance.

    Returns
    -------
    dict
        Keys:
        ``batt_annual_discharge_energy_kwh``,
        ``batt_annual_charge_energy_kwh``,
        ``batt_roundtrip_efficiency_pct``,
        ``batt_capacity_end_of_life_pct``,
        ``batt_cycles_total``.
    """
    out = model.Outputs

    discharge_kwh = _sum_output(out.batt_annual_discharge_energy)
    charge_kwh = _sum_output(out.batt_annual_charge_energy)

    # Roundtrip efficiency: use dedicated output, fall back to ratio
    try:
        rte = _mean_output(out.batt_roundtrip_efficiency)
    except AttributeError:
        rte = (discharge_kwh / charge_kwh * 100.0) if charge_kwh > 0.0 else 0.0

    # SOH at end of simulation period
    try:
        soh = _last_output(out.batt_capacity_percent)
    except AttributeError:
        soh = 100.0

    # Total cycle count
    try:
        cycles = _sum_output(out.batt_cycles)
    except AttributeError:
        cycles = 0.0

    return {
        "batt_annual_discharge_energy_kwh": round(discharge_kwh, 3),
        "batt_annual_charge_energy_kwh": round(charge_kwh, 3),
        "batt_roundtrip_efficiency_pct": round(rte, 2),
        "batt_capacity_end_of_life_pct": round(soh, 1),
        "batt_cycles_total": round(cycles, 1),
    }


# ---------------------------------------------------------------------------
# Co-located PV + BESS
# ---------------------------------------------------------------------------

@dataclass
class PvBessSystem(System):
    """Co-located PV + battery system.

    Extends ``System`` by adding battery hardware and dispatch.  Uses the same
    ``PySAM.Pvsamv1`` model as PV-only but activates battery groups via
    ``en_batt=1``.

    All PV fields and ``System.run()`` semantics are unchanged. This class
    overrides ``run()`` to additionally assign battery inputs and call
    ``process_bess_outputs()``.

    Parameters (additions to System)
    ---------------------------------
    battery : Battery
        Battery hardware specification.
    dispatch : BessDispatch
        Dispatch strategy and schedule.
    load_profile : list of float
        8760-hour AC load in kW.  Required when
        ``dispatch.strategy == "self_consumption"``.
    """

    battery: Battery = field(default_factory=Battery)
    dispatch: BessDispatch = field(default_factory=BessDispatch)
    load_profile: List[float] = field(default_factory=list)

    def __post_init__(
        self,
        target_kwac,
        target_dcac,
        met_year,
        pan_file,
        modules_per_string,
        ond_file,
    ):
        super().__post_init__(
            target_kwac,
            target_dcac,
            met_year,
            pan_file,
            modules_per_string,
            ond_file,
        )

    def run(self) -> dict:
        """Configure and execute the PV+BESS simulation.

        Assigns PV inputs (via inherited ``generate_pysam_inputs``), then
        overlays battery groups, executes, and returns a merged results dict
        containing both PV energy/loss keys and battery energy keys.

        Returns
        -------
        dict
            All keys from ``process_outputs()`` plus all keys from
            ``process_bess_outputs()``.
        """
        self.model.assign(generate_pysam_inputs(self))
        self.model.assign(self._generate_battery_inputs())
        self.model.execute()
        pv_results = process_outputs(self)
        bess_results = process_bess_outputs(self.model)
        self.model_results = {**pv_results, **bess_results}
        return self.model_results

    def _generate_battery_inputs(self) -> dict:
        """Build the PySAM input dict for battery groups.

        Returns a dict suitable for ``model.assign()``.  Merges into the PV
        inputs already assigned by ``generate_pysam_inputs()``.
        """
        batt = self.battery
        disp = self.dispatch

        inputs: dict = {
            "BatterySystem": {
                "en_batt": 1,
                "batt_chem": Battery.CHEMISTRY_MAP.get(batt.chemistry, 1),
                "batt_computed_bank_capacity": batt.energy_kwh,
                "batt_power_charge_max_kwdc": batt.power_kw,
                "batt_power_discharge_max_kwdc": batt.power_kw,
                "batt_Vnom_default": batt.batt_Vnom_default,
                "batt_ac_or_dc": 1 if batt.coupling == "AC" else 0,
                "batt_dc_ac_efficiency": batt.dc_ac_efficiency,
                "batt_ac_dc_efficiency": batt.ac_dc_efficiency,
                "batt_meter_position": 0,   # 0 = behind-the-meter
            },
            "BatteryCell": {
                "batt_initial_SOC": batt.soc_init,
                "batt_minimum_SOC": batt.soc_min,
                "batt_maximum_SOC": batt.soc_max,
                # Linear calendar degradation model
                "batt_calendar_choice": 1,
                "batt_calendar_q0": 1.02,
                "batt_calendar_a": 0.003,
                "batt_calendar_b": -7280.0,
                "batt_calendar_c": 930.0,
            },
            "BatteryDispatch": {
                "batt_dispatch_choice": disp.pysam_dispatch_choice(),
                "batt_dispatch_manual_charge": disp.can_charge,
                "batt_dispatch_manual_discharge": disp.can_discharge,
                "batt_dispatch_manual_gridcharge": disp.can_gridcharge,
                "batt_dispatch_manual_percent_discharge": disp.percent_discharge,
                "batt_dispatch_manual_percent_gridcharge": disp.percent_gridcharge,
                "batt_dispatch_manual_sched": disp.schedule_weekday,
                "batt_dispatch_manual_sched_weekend": disp.schedule_weekend,
            },
        }

        # Provide load profile for self-consumption dispatch
        if disp.strategy == "self_consumption" and self.load_profile:
            inputs["Load"] = {"load": self.load_profile}

        return inputs


# ---------------------------------------------------------------------------
# Standalone BESS (no PV)
# ---------------------------------------------------------------------------

@dataclass
class StandaloneBessSystem:
    """Standalone battery system (no PV).

    Uses ``PySAM.StandAloneBattery`` to simulate a battery dispatching against
    a load profile.  Downloads NSRDB weather data to extract an ambient
    temperature profile for the battery thermal model.

    Parameters
    ----------
    battery : Battery
        Battery hardware specification.
    dispatch : BessDispatch
        Dispatch strategy and schedule.
    load_profile : list of float
        8760-hour AC load in kW (required).
    lat : float
        Site latitude (degrees N).
    lon : float
        Site longitude (degrees E, negative = W).
    weather_year : str
        ``'tmy'`` or a four-digit year string (e.g. ``'2017'``).
    analysis_period : int
        Project lifetime in years (used for degradation and financial calcs).
    """

    battery: Battery
    dispatch: BessDispatch
    load_profile: List[float]

    lat: float = LATITUDE
    lon: float = LONGITUDE
    weather_year: str = "tmy"
    analysis_period: int = 25

    model: Any = field(default=None, init=False)
    model_results: dict = field(default=None, init=False)

    def __post_init__(self):
        import PySAM.StandAloneBattery as sa

        try:
            self.model = sa.default("StandAloneBatteryNone")
        except Exception:
            self.model = sa.new()

    def run(self) -> dict:
        """Download ambient temperature, assign all battery inputs, and execute.

        Returns
        -------
        dict
            Battery energy and performance outputs from ``process_bess_outputs()``.
        """
        ambient_temp = self._get_ambient_temperature()
        self.model.assign(self._generate_bess_inputs(ambient_temp))
        self.model.execute()
        self.model_results = process_bess_outputs(self.model)
        return self.model_results

    def _get_ambient_temperature(self) -> List[float]:
        """Download (or use cached) NSRDB weather file and return the 8760
        hourly ``Temperature`` column (°C).
        """
        from pvsamlab.climate import download_nsrdb_csv

        weather_file = download_nsrdb_csv(
            coords=(self.lat, self.lon),
            year=self.weather_year,
        )
        if weather_file is None:
            raise RuntimeError(
                f"Weather file download failed for ({self.lat}, {self.lon}), "
                f"year='{self.weather_year}'. "
                "Check NSRDB API credentials and network access."
            )

        # NSRDB CSV: 2 metadata rows, then column-header row, then data
        df = pd.read_csv(weather_file, skiprows=2)

        if "Temperature" not in df.columns:
            raise RuntimeError(
                f"'Temperature' column not found in weather file: {weather_file}. "
                f"Available columns: {list(df.columns)}"
            )

        return df["Temperature"].tolist()[:8760]

    def _generate_bess_inputs(self, ambient_temp: List[float]) -> dict:
        """Build the full PySAM input dict for StandAloneBattery."""
        batt = self.battery
        disp = self.dispatch

        return {
            "BatterySystem": {
                "en_batt": 1,
                "batt_chem": Battery.CHEMISTRY_MAP.get(batt.chemistry, 1),
                "batt_computed_bank_capacity": batt.energy_kwh,
                "batt_power_charge_max_kwdc": batt.power_kw,
                "batt_power_discharge_max_kwdc": batt.power_kw,
                "batt_Vnom_default": batt.batt_Vnom_default,
                "batt_ac_or_dc": 1 if batt.coupling == "AC" else 0,
                "batt_dc_ac_efficiency": batt.dc_ac_efficiency,
                "batt_ac_dc_efficiency": batt.ac_dc_efficiency,
                # Ambient temperature for cell thermal model (°C, 8760 values)
                "batt_room_temperature_celsius": ambient_temp,
            },
            "BatteryCell": {
                "batt_initial_SOC": batt.soc_init,
                "batt_minimum_SOC": batt.soc_min,
                "batt_maximum_SOC": batt.soc_max,
                "batt_calendar_choice": 1,
                "batt_calendar_q0": 1.02,
                "batt_calendar_a": 0.003,
                "batt_calendar_b": -7280.0,
                "batt_calendar_c": 930.0,
            },
            "BatteryDispatch": {
                "batt_dispatch_choice": disp.pysam_dispatch_choice(),
                "batt_dispatch_manual_charge": disp.can_charge,
                "batt_dispatch_manual_discharge": disp.can_discharge,
                "batt_dispatch_manual_gridcharge": disp.can_gridcharge,
                "batt_dispatch_manual_percent_discharge": disp.percent_discharge,
                "batt_dispatch_manual_percent_gridcharge": disp.percent_gridcharge,
                "batt_dispatch_manual_sched": disp.schedule_weekday,
                "batt_dispatch_manual_sched_weekend": disp.schedule_weekend,
            },
            "Load": {
                "load": self.load_profile,
            },
        }

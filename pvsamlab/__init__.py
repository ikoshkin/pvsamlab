# Public API for pvsamlab

# PV-only (existing)
from pvsamlab.system import System, Losses, TrackingMode, IrradianceMode, Orientation

# BESS extension
from pvsamlab.battery import (
    Battery,
    BessDispatch,
    PvBessSystem,
    StandaloneBessSystem,
    process_bess_outputs,
)

# Financial
from pvsamlab.financial import (
    Financial,
    RevenueStack,
    compute_lcoe,
    compute_lcos,
    compute_npv,
    compute_irr,
)

# Components (lower-level)
from pvsamlab.components import Module, Inverter

# Climate / NSRDB utilities
from pvsamlab.climate import check_nsrdb_connectivity, validate_weather_file

__all__ = [
    # PV-only
    "System",
    "Losses",
    "TrackingMode",
    "IrradianceMode",
    "Orientation",
    # BESS
    "Battery",
    "BessDispatch",
    "PvBessSystem",
    "StandaloneBessSystem",
    "process_bess_outputs",
    # Financial
    "Financial",
    "RevenueStack",
    "compute_lcoe",
    "compute_lcos",
    "compute_npv",
    "compute_irr",
    # Components
    "Module",
    "Inverter",
    # Climate
    "check_nsrdb_connectivity",
    "validate_weather_file",
]

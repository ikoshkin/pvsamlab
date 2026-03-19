"""
Financial dataclass and metric computation for pvsamlab.

Provides LCOE (via PySAM SingleOwner or Cashloan), LCOS (post-simulation Python),
NPV, and IRR calculations for PV-only, PV+BESS, and BESS-only systems.

Usage
-----
    fin = Financial(ppa_rate=45.0, itc_rate=30.0)

    # After system.run():
    lcoe_results = compute_lcoe(system, fin)

    # LCOS requires per-year discharge data:
    lcos = compute_lcos(system.battery, annual_discharge, annual_opex,
                        replacement_events=[], discount_rate=0.08)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from scipy.optimize import brentq

# PySAM depreciation schedule enum mapping
_DEPR_TYPE_MAP: dict = {
    "MACRS5": 2,
    "straight_line": 1,
    "none": 0,
}


@dataclass
class Financial:
    """Financial assumptions for a pvsamlab simulation.

    All rate fields are expressed as percentages (e.g., 8.0 for 8%).
    All cost fields are expressed in USD or USD per unit.

    Selecting SingleOwner vs Cashloan
    ----------------------------------
    compute_lcoe() automatically selects the financial module:
    - SingleOwner  when system.kwac > 1000 kW (utility-scale)
    - Cashloan     when system.kwac <= 1000 kW (C&I / residential)
    """

    # Project timeline
    analysis_period: int = 25               # years
    degradation_rate: float = 0.5           # %/year PV output decay

    # Costs — PV
    pv_capex_per_kwdc: float = 700.0        # $/kWdc all-in (modules + inverter + BOS + EPC)
    opex_per_kwac_year: float = 15.0        # $/kWac/year (O&M + insurance + land)

    # Revenue
    ppa_rate: float = 40.0                  # $/MWh
    ppa_escalation: float = 1.0             # %/year price escalation

    # Financing
    discount_rate: float = 8.0              # % real (WACC)
    inflation_rate: float = 2.5             # %
    debt_fraction: float = 70.0             # %
    loan_rate: float = 5.0                  # % nominal annual
    loan_term: int = 18                     # years

    # Tax
    federal_tax_rate: float = 21.0          # %
    state_tax_rate: float = 0.0             # %
    itc_rate: float = 30.0                  # % investment tax credit (IRA)
    depreciation_schedule: str = "MACRS5"   # "MACRS5" | "straight_line" | "none"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_total_installed_cost(system: Any, financial: Financial) -> float:
    """Return total installed cost in USD.

    Adds BESS capex automatically if system has a ``battery`` attribute
    (i.e., is a PvBessSystem).
    """
    pv_cost = system.kwdc * financial.pv_capex_per_kwdc
    battery = getattr(system, "battery", None)
    if battery is not None:
        bess_cost = (
            battery.energy_kwh * battery.capex_per_kwh
            + battery.power_kw * battery.capex_per_kw
        )
    else:
        bess_cost = 0.0
    return pv_cost + bess_cost


def _assign_single_owner(
    fin_model: Any,
    financial: Financial,
    total_installed_cost: float,
    annual_opex: float,
) -> None:
    """Populate a SingleOwner model instance from a Financial dataclass."""
    fp = fin_model.FinancialParameters
    fp.analysis_period = financial.analysis_period
    fp.real_discount_rate = financial.discount_rate
    fp.inflation_rate = financial.inflation_rate
    fp.federal_tax_rate = financial.federal_tax_rate
    fp.state_tax_rate = financial.state_tax_rate
    fp.debt_fraction = financial.debt_fraction
    fp.term_int_rate = financial.loan_rate
    fp.term_tenor = financial.loan_term

    sc = fin_model.SystemCosts
    sc.total_installed_cost = total_installed_cost
    sc.om_fixed = [annual_opex]

    rev = fin_model.Revenue
    rev.ppa_price_input = financial.ppa_rate / 1000.0   # $/MWh → $/kWh
    rev.ppa_escalation = financial.ppa_escalation

    tci = fin_model.TaxCreditIncentives
    tci.itc_fed_percent = [financial.itc_rate]
    tci.itc_fed_percent_deprbas_fed = 1
    tci.itc_fed_percent_deprbas_sta = 1

    dep = fin_model.Depreciation
    dep.depr_fed_type = _DEPR_TYPE_MAP.get(financial.depreciation_schedule, 2)
    dep.depr_sta_type = 0


def _assign_cashloan(
    fin_model: Any,
    financial: Financial,
    total_installed_cost: float,
    annual_opex: float,
) -> None:
    """Populate a Cashloan model instance from a Financial dataclass."""
    fp = fin_model.FinancialParameters
    fp.analysis_period = financial.analysis_period
    fp.real_discount_rate = financial.discount_rate
    fp.inflation_rate = financial.inflation_rate
    fp.federal_tax_rate = financial.federal_tax_rate
    fp.state_tax_rate = financial.state_tax_rate
    fp.debt_fraction = financial.debt_fraction
    fp.term_int_rate = financial.loan_rate
    fp.term_tenor = financial.loan_term

    sc = fin_model.SystemCosts
    sc.total_installed_cost = total_installed_cost
    sc.om_fixed = [annual_opex]

    tci = fin_model.TaxCreditIncentives
    tci.itc_fed_percent = [financial.itc_rate]
    tci.itc_fed_percent_deprbas_fed = 1
    tci.itc_fed_percent_deprbas_sta = 1

    dep = fin_model.Depreciation
    dep.depr_fed_type = _DEPR_TYPE_MAP.get(financial.depreciation_schedule, 2)
    dep.depr_sta_type = 0


# ---------------------------------------------------------------------------
# Public compute functions
# ---------------------------------------------------------------------------

def compute_lcoe(system: Any, financial: Financial) -> dict:
    """Compute LCOE, NPV, and IRR using PySAM SingleOwner or Cashloan.

    ``system.run()`` must have been called before invoking this function.

    The financial module is selected automatically:
    - ``PySAM.SingleOwner``  when ``system.kwac > 1000`` kW (utility-scale)
    - ``PySAM.Cashloan``     when ``system.kwac <= 1000`` kW (C&I)

    Parameters
    ----------
    system : System | PvBessSystem
        A pvsamlab system instance on which ``.run()`` has already been called.
    financial : Financial
        Financial assumptions.

    Returns
    -------
    dict
        Keys: ``lcoe_real_cents_per_kwh``, ``lcoe_nom_cents_per_kwh``,
        ``npv_usd``, ``irr_pct``, ``payback_years``, ``total_installed_cost_usd``.
    """
    if system.model_results is None:
        raise RuntimeError(
            "system.run() must be called before compute_lcoe(). "
            "model_results is None."
        )

    total_installed_cost = _compute_total_installed_cost(system, financial)
    annual_opex = system.kwac * financial.opex_per_kwac_year

    # Q4: SingleOwner for kwac > 1000 kW (>1 MWac), Cashloan for <= 1000 kW
    if system.kwac > 1000.0:
        import PySAM.SingleOwner as so_mod

        fin_model = so_mod.from_existing(system.model, "FlatPlatePVSingleOwner")
        _assign_single_owner(fin_model, financial, total_installed_cost, annual_opex)
        fin_model.execute()

        return {
            "lcoe_real_cents_per_kwh": round(fin_model.Outputs.lcoe_real, 4),
            "lcoe_nom_cents_per_kwh": round(fin_model.Outputs.lcoe_nom, 4),
            "npv_usd": round(fin_model.Outputs.project_return_aftertax_npv, 2),
            "irr_pct": round(fin_model.Outputs.project_return_aftertax_irr, 3),
            "payback_years": round(fin_model.Outputs.discounted_payback, 2),
            "total_installed_cost_usd": round(total_installed_cost, 2),
        }
    else:
        import PySAM.Cashloan as cl_mod

        fin_model = cl_mod.from_existing(system.model, "FlatPlatePVCommercial")
        _assign_cashloan(fin_model, financial, total_installed_cost, annual_opex)
        fin_model.execute()

        return {
            "lcoe_real_cents_per_kwh": round(fin_model.Outputs.lcoe_real, 4),
            "lcoe_nom_cents_per_kwh": round(fin_model.Outputs.lcoe_nom, 4),
            "npv_usd": round(fin_model.Outputs.npv, 2),
            "irr_pct": round(fin_model.Outputs.after_tax_irr, 3),
            "payback_years": round(fin_model.Outputs.discounted_payback, 2),
            "total_installed_cost_usd": round(total_installed_cost, 2),
        }


def compute_lcos(
    battery: Any,
    annual_discharge_kwh: List[float],
    annual_opex: float,
    replacement_events: List[Tuple[int, float]],
    discount_rate: float,
) -> float:
    """Compute Levelized Cost of Storage (LCOS) in $/kWh.

    SAM does not compute LCOS natively; this is a pure-Python post-simulation
    calculation based on discounted life-cycle costs and energy throughput.

    Parameters
    ----------
    battery : Battery
        Battery dataclass instance with ``energy_kwh``, ``power_kw``,
        ``capex_per_kwh``, and ``capex_per_kw`` attributes.
    annual_discharge_kwh : list of float
        Energy discharged from the battery per project year (kWh).
        Length determines the analysis period.
    annual_opex : float
        Fixed annual BESS O&M cost in $/year.
    replacement_events : list of (year, cost_$) tuples
        Each tuple is ``(project_year, replacement_cost_usd)`` where year 1
        is the first operating year.
    discount_rate : float
        Real discount rate as a decimal (e.g., 0.08 for 8%).

    Returns
    -------
    float
        LCOS in $/kWh, or ``inf`` if no energy is ever discharged.
    """
    batt_capex = (
        battery.energy_kwh * battery.capex_per_kwh
        + battery.power_kw * battery.capex_per_kw
    )
    n = len(annual_discharge_kwh)

    pv_costs = (
        batt_capex
        + sum(annual_opex / (1.0 + discount_rate) ** t for t in range(1, n + 1))
        + sum(
            cost / (1.0 + discount_rate) ** yr
            for yr, cost in replacement_events
        )
    )
    pv_discharge = sum(
        kwh / (1.0 + discount_rate) ** t
        for t, kwh in enumerate(annual_discharge_kwh, 1)
    )
    return pv_costs / pv_discharge if pv_discharge > 0.0 else float("inf")


def compute_npv(cash_flows: List[float], discount_rate: float) -> float:
    """Compute Net Present Value of a cash flow series.

    ``cash_flows[0]`` is the year-0 investment (typically negative); subsequent
    elements are year-1, year-2, … cash flows.

    Parameters
    ----------
    cash_flows : list of float
        Full cash flow series including year-0 investment.
    discount_rate : float
        Discount rate as a decimal (e.g., 0.08 for 8%).

    Returns
    -------
    float
        NPV in the same currency units as cash_flows.
    """
    return sum(
        cf / (1.0 + discount_rate) ** t
        for t, cf in enumerate(cash_flows)
    )


def compute_irr(cash_flows: List[float]) -> float:
    """Compute Internal Rate of Return of a cash flow series.

    ``cash_flows[0]`` must be negative (the initial investment).
    Uses ``scipy.optimize.brentq`` to solve NPV(r) = 0.

    Parameters
    ----------
    cash_flows : list of float
        Full cash flow series including year-0 investment (first value negative).

    Returns
    -------
    float
        IRR as a decimal (e.g., 0.12 for 12%), or ``nan`` if not solvable.
    """
    def _npv_at_rate(r: float) -> float:
        return sum(cf / (1.0 + r) ** t for t, cf in enumerate(cash_flows))

    try:
        return brentq(_npv_at_rate, -0.9999, 100.0, xtol=1e-8)
    except ValueError:
        return float("nan")

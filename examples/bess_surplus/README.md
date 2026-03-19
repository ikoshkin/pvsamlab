# Brownfield BESS Surplus Optimization

Optimizes BESS size and charging mode for an existing solar asset at a
fixed point-of-interconnection (POI) limit. Uses real merchant price curves
for price-signal dispatch.

## Concept

A "brownfield" solar site has unused interconnection headroom (e.g., 300 MWac
PV plant at a 300 MW POI — but PV output is intermittent, so the POI is
underutilized ~70% of hours). A co-located BESS can capture that unused
capacity and dispatch into high-price hours.

## Merchant curve CSV format

`pvsamlab/data/DAMPriceExample.csv` — single column, 8760 rows, $/MWh,
no header. Replace with actual ERCOT HB_WEST or other hub settlement prices.

## Charging modes

| Mode | Grid charge | Solar charge | Use case |
|------|-------------|--------------|----------|
| `solar_only` | No | Yes | ITC qualification (no grid charge) |
| `grid_only` | Yes | No | Pure arbitrage |
| `unrestricted` | Yes | Yes | Maximum revenue |

## Sizing matrix (75 cases = 25 × 3)

- Power: 50, 75, 100, 125, 150 MW
- Duration: 1, 2, 4, 6, 8 hr
- Modes: 3 (above)

## POI limit enforcement

`GridLimits.enable_interconnection_limit=1` with
`grid_interconnection_limit_kwac` set to POI capacity in kW.

## Interactive dashboard (Cell 9)

- **Row 1** — Case selection (power, duration, charging mode)
- **Row 2** — Financial sliders (CAPEX, capacity payment, discount rate) — recalculate NPV/IRR/LCOS without re-simulation
- **Row 3** — POI target %, week selector, price curve toggle (Merchant / Flat $45 / TOU)
- **4-panel output** — KPI box, dispatch stack, SOC heatmap, price-dispatch scatter
- **Breakeven button** — find capacity payment where NPV = 0
- **Pareto button** — IRR vs NPV scatter with Pareto-optimal cases highlighted
- **Export button** — dump filtered results to `outputs/bess_surplus_filtered_export.csv`

## Expected outputs (written to `outputs/`)

| File | Description |
|------|-------------|
| `bess_surplus_optimization_results.csv` | 75-row results table |
| `surplus_irr_heatmap.png` | 4-panel IRR heatmap (3 modes + delta) |
| `surplus_revenue_waterfall.png` | Revenue components per mode |
| `surplus_price_dispatch.png` | Price vs battery dispatch scatter (best case) |

## Running

Open `bess_surplus_optimization.ipynb` in Jupyter, or headless:
```bash
cd examples/bess_surplus
python run_surplus_analysis.py
```

## Estimated runtime

~60–90 seconds with 4 workers (75 PySAM simulations).

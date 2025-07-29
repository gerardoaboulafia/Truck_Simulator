# DISTRIMAX Logistics Simulator

> **Discrete-event simulation (DES)** built in Python 3.11 + Salabim to study how many trucks (and which dispatch
logic) a three-plant supply chain needs to run smoothly.  

---
DISTRIMAX operates three plants and a central distribution center (CD).  
The current layout—**one dedicated tractor per plant**—creates queues and long waits.  
Our goal is to compare:

1. **Baseline (Heuristic 1)** – each plant keeps its own truck.  
2. **Shared fleet (Heuristic 2)** – trucks choose the next plant by a score that blends ETA and workload, tunable with a β parameter.

Simulations show that four shared trucks with **β = 2** deliver ≈ 85.7 trailers per 8-h shift (vs 77.7 in the baseline) with far shorter empty-trailer waits at the CD.

---

## Decision variables & KPIs
| Type | Items |
|------|-------|
| **Decision** | dispatch rule (dedicated vs shared), number of trucks, β weight |
| **Reference metrics** | trailers delivered, mean wait (plant / CD), truck utilisation, per-plant deliveries |

---

## Main findings
| Scenario | Trucks | β | Deliveries/shift | Wait for empty at CD | Truck util. |
|----------|--------|---|------------------|----------------------|-------------|
| Heuristic 1 | 3 | — | **77.7** | 9.7 min | 97.8 % |
| Heuristic 2 | 4 | 2 | **85.7** | 1.4 min | 85.0 % |

The extra truck relieves CD congestion but shifts some waiting back to the plants: a trade-off the β score helps balance.

---

## How it works
* **Events & resources**: `Truck`, `Loader`, `Unloader`, trailer stores, one-slot dock at CD  
* **Stochastic timings**:  
  * Load ~ U(8, 16) min  
  * Unload ~ U(4, 8) min  
  * Hook / unhook ~ U(0.5, 1) min  
  * Travel speed ~ N(40 m/min, σ = 3) bounded > 15 m/min

* **Monte Carlo runs**: 100 replications per setting (400 when sweeping β) to build 90 % CIs  
* **Verification / validation**: step-by-step entity logs, unit tests with *pytest*, face-validity checks and sensitivity to speed changes.

---

## Repository layout
**Simulators**

└─ sim_animado.py           # Heuristic 1 — dedicated trucks per plant

└─ 2_sim_animado.py         # Heuristic 2 — shared/free trucks

└─ 2_sim_animado_4trucks.py # Heuristic 2 with one additional truck

**Batch runners**

└─ sim_out.py   # Runs sim_animado.py N times and records KPIs

└─ sim_out_2.py # Runs 2_sim_animado.py N times

└─ sim_out_3.py # Runs 2_sim_animado_4trucks.py N times

**Extracted data**

└─ sim_results_h1.csv     # Results for Heuristic 1

└─ sim_results_betas.csv  # Results for Heuristic 2 (β sweep)

└─ sim_results_4trucks.csv# Results for Heuristic 2 + extra truck

**Anlysis**

└─ Variables_Referencia.ipynb  # Notebook to explore the CSV outputs

**Other**

└─ test_invariantes.py    # Unit tests (invariant checks)

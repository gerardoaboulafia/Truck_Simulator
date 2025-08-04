import pytest
from sim_animado import run_sim

HOURS = 4           # horizonte de simulación
TOTAL_SEMIS = 9     # 3 plantas × 3 semirremolques

# ────────────────────────────────────────────────────────────────
# 1. Nunca “desaparecen” semirremolques: los que están en colas
#    (planta + CD) deben sumar entre 3 y 9 dependiendo del instante
#    en que se corte la simulación.
# ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [1, 42, 123])
def test_total_en_colas(seed):
    res = run_sim(hours=HOURS, seed=seed, animate=False, trace=False)

    total_colas = sum(res["queues_end"]["loaded"]) \
                + sum(res["queues_end"]["empty_p"]) \
                + sum(res["queues_end"]["empty_cd"]) \
                + res["loaded_cd_len"]

    # puede haber hasta 6 semis “en mano” (3 enganchados + loader/unloader),
    # así que en colas deben quedar entre 3 y 9 inclusive
    assert 3 <= total_colas <= TOTAL_SEMIS


# ────────────────────────────────────────────────────────────────
# 2. Todos los camiones hacen al menos un viaje
# ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [5, 99])
def test_viajes_positivos(seed):
    res = run_sim(hours=HOURS, seed=seed, animate=False, trace=False)
    assert all(t > 0 for t in res["truck_trips"])
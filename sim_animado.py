# Escenario 3 × 3 × 1 — métricas ampliadas (distribuciones alineadas con Escenario 2)
# --------------------------------------------------------------------------------------
# • viajes por camión
# • trailers cargados entregados
# • ⌛ espera promedio en planta  (por viaje)
# • ⌛ espera promedio en CD      (por viaje)
# • utilización del camión        (100-idle %)
# • “sin vacío” :  veces / min que llegó al CD sin trailer vacío
# • “sin lleno” :  veces / min que esperó un cargado en planta
# --------------------------------------------------------------------------------------

import salabim as sim, random
from math import hypot
from typing import Dict, List
import pandas as pd

# ─────────────  parámetros visuales  ──────────────
CARRY_OFFSET   = (0, -12)
TRUCK_COLORS   = ["blue", "green", "red"]
TRAILER_COLORS = ["lightblue", "lightgreen", "pink"]
TRAILER_ALPHA  = 150          # 60 % opaco

# ─────────────  helpers de rename  ────────────────
def set_loaded(tr):
    tr.loaded = True
    tr.name(f"P{tr.plant_id+1}_loaded")

def set_empty_cd(tr):
    tr.loaded = False
    tr.name(f"P{tr.plant_id+1}_emptyCD")

# ──────────────────  simulación  ──────────────────
def run_sim(
        hours: int = 8,
        seed: int = 42,
        plants: int = 3,
        animate: bool = True,
        trace: bool = False,
        debug: bool = False
) -> Dict:
    """Corre la simulación principal y devuelve un diccionario con métricas."""
    random.seed(seed)
    sim.yieldless(False)

    # ── distribuciones (idénticas al Escenario 2) ──────────────────────────
    load_t   = sim.Uniform(8, 16)   # carga
    unload_t = sim.Uniform(4, 8)   # descarga
    hook_t   = sim.Uniform(0.5, 1)     # enganche
    speed_dist = sim.Bounded(sim.Normal(40, 3), lowerbound=15)  

    # ── entorno ────────────────────────────────────────────────────────────
    env = sim.Environment(time_unit="minutes",
                          width=1200,
                          height=700,
                          animate=animate,
                          trace=trace)
    if animate:
        env.animation_parameters(background_color="90%gray")
        env.speed(2)

    def log(msg):
        if debug:
            print(f"[{env.now():5.1f}] {msg}")

    # ── geometría ──────────────────────────────────────────────────────────
    CD_x, CD_y = env.width() // 2, int(env.height() * 0.25)
    half = 150
    CD_PT = {i: (CD_x + (-1 + i) * half, CD_y) for i in range(plants)}

    P_X, P_YF = [350, 650, 900], [0.55, 0.68, 0.70]
    COORD = {i: (P_X[i], int(env.height() * P_YF[i])) for i in range(plants)}

    def t_travel(pid: int) -> float:
        dist = hypot(CD_PT[pid][0] - COORD[pid][0],
                     CD_PT[pid][1] - COORD[pid][1])
        v = speed_dist.sample()
        return dist / v

    # ── dibujo fijo (opcional) ─────────────────────────────────────────────
    if animate:
        cd_w = 300
        sx, sy = CD_x - cd_w // 2, CD_y - 5
        sim.AnimatePolygon([sx, sy, sx+cd_w, sy, sx+cd_w, sy+10, sx, sy+10],
                           fillcolor="orange", layer=0)
        sim.AnimateText("CD", x=CD_x - 10, y=CD_y + 14, layer=0)
        for i in range(plants):
            x, y = COORD[i]
            sim.AnimateCircle(25, x=x, y=y,
                              fillcolor=("lightgray", 120), layer=0)
            sim.AnimateText(f"P{i+1}", x=x-10, y=y+42, layer=0)

    # ── colas / recursos ──────────────────────────────────────────────────
    dock_cd  = sim.Resource("Dock_CD", capacity=1)
    loaded   = [sim.Store(f"Loaded_P{i+1}")   for i in range(plants)]
    empty_pl = [sim.Store(f"Empty_P{i+1}")    for i in range(plants)]
    empty_cd = [sim.Store(f"Empty_CD_P{i+1}") for i in range(plants)]
    loaded_cd = sim.Store("Loaded_CD")

    delivered = [0]*plants

    # ── helper trailer ────────────────────────────────────────────────────
    def trailer(name, pos, pid, *, loaded_flag=False):
        tr = sim.Component(name=name)
        tr.plant_id = pid
        tr.loaded = loaded_flag
        tr.sprite = sim.AnimateCircle(
            10, x=pos[0], y=pos[1],
            fillcolor=(TRAILER_COLORS[pid], TRAILER_ALPHA), layer=2)
        return tr

    # ── Loader ────────────────────────────────────────────────────────────
    class Loader(sim.Component):
        def setup(self, pid): self.pid = pid
        def process(self):
            while True:
                tr = yield self.from_store(empty_pl[self.pid])
                log(f"Loader P{self.pid+1} ← {tr.name()}")
                yield self.hold(load_t.sample())
                set_loaded(tr)
                log(f"Loader P{self.pid+1} → cargado {tr.name()}")
                yield self.to_store(loaded[self.pid], tr)

    # ── Unloader ──────────────────────────────────────────────────────────
    class Unloader(sim.Component):
        def process(self):
            while True:
                tr = yield self.from_store(loaded_cd)
                log(f"Unloader ↙ {tr.name()}")
                yield self.hold(unload_t.sample())
                delivered[tr.plant_id] += 1
                set_empty_cd(tr)
                log(f"Unloader ↗ {tr.name()}")
                yield self.to_store(empty_cd[tr.plant_id], tr)

    # ── Truck ─────────────────────────────────────────────────────────────
    class Truck(sim.Component):
        def setup(self, pid, first_trailer):
            self.pid = pid
            self.carry = first_trailer
            self.trips = 0
            self.wait_pl = 0.0
            self.wait_cd = 0.0
            self.no_empty = 0
            self.no_full = 0
            x, y = COORD[pid]
            self.sprite = sim.AnimateCircle(
                18, x=x, y=y-50, fillcolor=TRUCK_COLORS[pid],
                text=f"T{pid+1}", textcolor="white", layer=5)
            first_trailer.sprite.x = x + CARRY_OFFSET[0]
            first_trailer.sprite.y = y - 50 + CARRY_OFFSET[1]

        def _attach(self, tr):
            self.carry = tr
            tr.sprite.x = self.sprite.x(env.now()) + CARRY_OFFSET[0]
            tr.sprite.y = self.sprite.y(env.now()) + CARRY_OFFSET[1]

        def _detach(self): self.carry = None

        def _move(self, dst, dur, steps=40):
            x, y = self.sprite.x(env.now()), self.sprite.y(env.now())
            dx, dy, dt = (dst[0]-x)/steps, (dst[1]-y)/steps, dur/steps
            for _ in range(steps):
                x += dx; y += dy
                self.sprite.x, self.sprite.y = x, y
                if self.carry:
                    self.carry.sprite.x = x + CARRY_OFFSET[0]
                    self.carry.sprite.y = y + CARRY_OFFSET[1]
                yield self.hold(dt)

        def process(self):
            # primer viaje
            log(f"{self.name()} SALE P{self.pid+1}→CD con {self.carry.name()}")
            yield self.hold(hook_t.sample())
            yield from self._move(CD_PT[self.pid], t_travel(self.pid))
            yield self.request(dock_cd); yield self.hold(hook_t.sample())
            yield self.to_store(loaded_cd, self.carry)
            self._detach(); self.release(dock_cd)
            self.trips = 1
            log(f"{self.name()} TRIP #{self.trips}")

            while True:
                # CD → planta con vacío
                t0 = env.now()
                tr = yield self.from_store(empty_cd[self.pid])
                dt = env.now() - t0
                self.wait_cd += dt
                if dt > 0: self.no_empty += 1
                self._attach(tr)
                log(f"{self.name()} SALE CD→P{self.pid+1} con {tr.name()}")
                yield self.hold(hook_t.sample())
                yield from self._move(COORD[self.pid], t_travel(self.pid))
                yield self.hold(hook_t.sample())
                yield self.to_store(empty_pl[self.pid], tr)
                self._detach()

                # planta → CD con cargado
                t0 = env.now()
                tr = yield self.from_store(loaded[self.pid])
                dt = env.now() - t0
                self.wait_pl += dt
                if dt > 0: self.no_full += 1
                self._attach(tr)
                log(f"{self.name()} SALE P{self.pid+1}→CD con {tr.name()}")
                yield self.hold(hook_t.sample())
                yield from self._move(CD_PT[self.pid], t_travel(self.pid))
                yield self.request(dock_cd); yield self.hold(hook_t.sample())
                yield self.to_store(loaded_cd, tr)
                self._detach(); self.release(dock_cd)
                self.trips += 1
                log(f"{self.name()} TRIP #{self.trips}")

    # ── inventario inicial ────────────────────────────────────────────────
    trucks=[]
    for i in range(plants):
        Loader(name=f"LoaderP{i+1}", pid=i)
        empty_pl[i].add(trailer(f"P{i+1}_emptyPL",
                                (COORD[i][0]-30, COORD[i][1]-25), i))
        empty_cd[i].add(trailer(f"P{i+1}_emptyCD",
                                (CD_PT[i][0], CD_PT[i][1]), i))
        first_loaded = trailer(
            f"P{i+1}_loaded", (COORD[i][0]+30, COORD[i][1]-10),
            i, loaded_flag=True)
        trucks.append(
            Truck(name=f"T{i+1}", pid=i, first_trailer=first_loaded)
        )

    Unloader(name="UnloaderCD")

    # ── run ────────────────────────────────────────────────────────────────
    env.run(till=hours*60)
    horizon = env.now()

    # ── resumen ────────────────────────────────────────────────────────────
    print("\n=== RESUMEN ===")
    print(f"Tiempo total: {horizon:.0f} min ({hours} horas)")
    print(f"Plantas: {plants} | Camiones: {len(trucks)}\n")

    print("─ Métricas por planta ─")
    for p in range(plants):
        print(f"P{p+1}: entregados {delivered[p]:3d} | "
              f"cola loaded {loaded[p].length():2d} | "
              f"empty P {empty_pl[p].length():2d} | "
              f"empty CD {empty_cd[p].length():2d}")
    print()

    tot_trips = tot_del = tot_wp = tot_wc = tot_ne = tot_nf = 0
    for t in trucks:
        util = 100 * (horizon - (t.wait_pl + t.wait_cd)) / horizon
        awp = t.wait_pl / t.trips
        awc = t.wait_cd / t.trips
        print(f"{t.name()}: viajes {t.trips:2d} | "
              f"cargados {delivered[t.pid]:3d} | "
              f"⌛ espera P {awp:5.2f} | "
              f"⌛ espera CD {awc:5.2f} | "
              f"util {util:6.2f}% | "
              f"sin vacío {t.no_empty}/{t.wait_cd:5.2f} min | "
              f"sin lleno {t.no_full}/{t.wait_pl:5.2f} min")
        tot_trips += t.trips;  tot_del += delivered[t.pid]
        tot_wp += t.wait_pl;   tot_wc += t.wait_cd
        tot_ne += t.no_empty;  tot_nf += t.no_full

    if tot_trips:
        g_awp  = tot_wp / tot_trips
        g_awc  = tot_wc / tot_trips
        g_util = 100 * (horizon - (tot_wp + tot_wc)) / horizon
        print("-"*100)
        # print(f"TOTAL: viajes {tot_trips:3d} | cargados {tot_del:3d} | "
        #       f"⌛ espera P {g_awp:5.2f} | "
        #       f"⌛ espera CD {g_awc:5.2f} | "
        #       f"util {g_util:6.2f}% | "
        #       f"sin vacío {tot_ne}/{tot_wc:5.2f} min | "
        #       f"sin lleno {tot_nf}/{tot_wp:5.2f} min")

    return {
        "trips":      [t.trips for t in trucks],
        "delivered":  delivered,
        "wait_pl":    [t.wait_pl for t in trucks],
        "wait_cd":    [t.wait_cd for t in trucks],
        "no_empty":   [t.no_empty for t in trucks],
        "no_full":    [t.no_full for t in trucks]
    }

# ── demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_sim(animate=False, debug=False)


    
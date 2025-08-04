# sim_animado_4trucks.py — DISTRIMAX · Escenario 2 – 4 Tractores libres
# ---------------------------------------------------------------------
# • 3 plantas, 4 camiones.
# ---------------------------------------------------------------------

import salabim as sim, random, numpy as np
from math import hypot
from typing import List, Dict, Optional
import pandas as pd

CARRY_OFFSET   = (0, -12)
TRAILER_COLORS = ["lightblue", "lightgreen", "pink"]

def set_loaded(tr):
    tr.loaded = True
    tr.name(f"P{tr.plant_id+1}_loaded")

def set_empty_cd(tr):
    tr.loaded = False
    tr.name(f"P{tr.plant_id+1}_emptyCD")

def run_sim(hours: float = 8,
            seed: Optional[int] = None,
            plants: int = 3,
            beta: float = 3,
            AVG_SPEED: float = 35,
            animate: bool = False,
            debug: bool = False) -> Dict:

    # ─── inicialización ───────────────────
    random.seed(seed); np.random.seed(seed); sim.random_seed(seed)
    sim.yieldless(False)
    env = sim.Environment(time_unit="minutes",
                          width=1200, height=700,
                          animate=animate, trace=debug)
    env.random_seed(seed)

    # ─── distribuciones ───────────────────
    load_t     = sim.Uniform(8, 16)
    unload_t   = sim.Uniform(4, 8)
    hook_t     = sim.Uniform(0.5, 1)
    speed_dist = sim.Bounded(sim.Normal(40, 3), lowerbound=15)

    def dbg(msg):
        if debug:
            print(msg)

    # ─── geometría ────────────────────────
    CD_x, CD_y = env.width()//2, int(env.height()*0.25)
    half = 150
    CD_PT = {i:(CD_x + (-1+i)*half, CD_y) for i in range(plants)}
    P_X, P_YF = [350,650,900], [0.57,0.68,0.65]
    COORD = {i:(P_X[i], int(env.height()*P_YF[i])) for i in range(plants)}
    DIST = [hypot(CD_PT[p][0]-COORD[p][0],
                  CD_PT[p][1]-COORD[p][1]) for p in range(plants)]
    ETA_EST = [d/AVG_SPEED for d in DIST]

    # ─── animación fija ───────────────────
    if animate:
        env.animation_parameters(background_color="lightgray")
        env.speed(4)
        # CD
        cd_w, cd_h = 300, 16
        sx, sy = CD_x-cd_w//2, CD_y-cd_h//2
        sim.AnimatePolygon([sx,sy, sx+cd_w,sy, sx+cd_w,sy+cd_h, sx,sy+cd_h],
                           fillcolor="orange", layer=0)
        sim.AnimateText("CD", x=CD_x-12, y=CD_y+cd_h//2+14,
                        textcolor="black", fontsize=12, layer=0)
        # Plantas
        for i in range(plants):
            x,y = COORD[i]
            sim.AnimateCircle(22, x=x, y=y, fillcolor="gainsboro", layer=0)
            sim.AnimateText(f"P{i+1}", x=x-10, y=y+36,
                            textcolor="black", fontsize=12, layer=0)

    # ─── recursos y colas ──────────────────
    dock_cd   = sim.Resource("Dock_CD", capacity=1)
    loaded    = [sim.Store(f"Loaded_P{i+1}")   for i in range(plants)]
    empty_pl  = [sim.Store(f"Empty_P{i+1}")    for i in range(plants)]
    empty_cd  = [sim.Store(f"Empty_CD_P{i+1}") for i in range(plants)]
    loaded_cd = sim.Store("Loaded_CD")

    delivered, selected = [0]*plants, [0]*plants
    all_trucks: List["Truck"] = []
    dock_busy_time = [0.0]

    def wake():
        for t in all_trucks:
            if t.ispassive():
                t.activate()

    def trailer(name, pos, pid, *, loaded_flag=False):
        tr = sim.Component(name=name)
        tr.plant_id = pid
        tr.loaded   = loaded_flag
        tr.sprite = sim.AnimateCircle(10, x=pos[0], y=pos[1],
                                      fillcolor=TRAILER_COLORS[pid], layer=2)
        return tr

    # ─── Loader ───────────────────────────
    loaders = []
    class Loader(sim.Component):
        def setup(self, pid):
            self.pid = pid
            self.busy_until = 0
            loaders.append(self)
        def rem(self):
            return max(0, self.busy_until - env.now())
        def process(self):
            while True:
                tr = yield self.from_store(empty_pl[self.pid])
                d  = load_t.sample()
                dbg(f"[{env.now():6.2f}] Loader P{self.pid+1} carga {tr.name()} dur={d:.1f}")
                self.busy_until = env.now() + d
                yield self.hold(d)
                set_loaded(tr)
                yield self.to_store(loaded[self.pid], tr)
                wake()

    # ─── Unloader ─────────────────────────
    class Unloader(sim.Component):
        def process(self):
            while True:
                tr = yield self.from_store(loaded_cd)
                d  = unload_t.sample()
                yield self.hold(d)
                delivered[tr.plant_id] += 1
                dbg(f"[{env.now():6.2f}] Unloader entrega {tr.name()} total_dlv_P{tr.plant_id+1}={delivered[tr.plant_id]}")
                set_empty_cd(tr)
                yield self.to_store(empty_cd[tr.plant_id], tr)
                wake()

    # ─── Truck ────────────────────────────
    class Truck(sim.Component):
        def setup(self, idx, home_pid, first_trailer, x_shift=0):
            self.idx, self.home = idx, home_pid
            self.carry          = first_trailer
            self.trips          = 0
            self.wait_pl = self.wait_cd = 0.0
            self.no_empty = self.no_full = 0
            all_trucks.append(self)
            # sprite
            if idx < plants:
                x0,y0 = COORD[home_pid]
            else:
                x0,y0 = CD_PT[0]
            y0 -= 50
            self.sprite = sim.AnimateCircle(18,
                x=x0+x_shift, y=y0,
                fillcolor="gray", text=f"T{idx+1}",
                textcolor="white", layer=5)
            # posiciona trailer inicial
            first_trailer.sprite.x = x0 + x_shift + CARRY_OFFSET[0]
            first_trailer.sprite.y = y0 + CARRY_OFFSET[1]

        def _attach(self, tr):
            self.carry = tr
            tr.sprite.x = self.sprite.x(env.now()) + CARRY_OFFSET[0]
            tr.sprite.y = self.sprite.y(env.now()) + CARRY_OFFSET[1]

        def _detach(self):
            self.carry = None

        def _move(self, dst, dur, steps=40):
            x,y = self.sprite.x(env.now()), self.sprite.y(env.now())
            dx,dy,dt = (dst[0]-x)/steps, (dst[1]-y)/steps, dur/steps
            for _ in range(steps):
                x += dx; y += dy
                self.sprite.x, self.sprite.y = x,y
                if self.carry:
                    self.carry.sprite.x = x + CARRY_OFFSET[0]
                    self.carry.sprite.y = y + CARRY_OFFSET[1]
                yield self.hold(dt)

        def choose(self):
            best,mdel = None, min(delivered)
            for p in range(plants):
                if empty_cd[p].length() == 0:
                    continue
                eta   = ETA_EST[p] + max(0, loaders[p].rem() - ETA_EST[p])
                score = eta + beta * (delivered[p] - mdel)
                if best is None or score < best[1]:
                    best = (p,score)
            return None if best is None else best[0]

        def process(self):
            while True:
                # descarga si lleva cargado
                if self.carry is not None:
                    pid = self.carry.plant_id
                    yield self.hold(hook_t.sample())
                    yield from self._move(CD_PT[pid], DIST[pid]/speed_dist.sample())

                    # ── medir espera en cola del dock_CD ─────────────────────
                    t0 = env.now()
                    yield self.request(dock_cd)
                    self.wait_cd += env.now() - t0
                    yield self.hold(hook_t.sample())
                    yield self.to_store(loaded_cd, self.carry)
                    self._detach()
                    self.release(dock_cd)
                    dock_busy_time[0] += env.now() - t0
                    self.trips += 1
                    wake()

                # elige planta
                pid = self.choose()
                if pid is None:
                    t0 = env.now()
                    yield self.passivate()
                    dt = env.now() - t0
                    self.wait_cd += dt       # espera por semis vacíos
                    if dt > 0:
                        self.no_empty += 1
                    continue

                selected[pid] += 1

                # vacía: CD → planta
                tr = yield self.from_store(empty_cd[pid])
                self._attach(tr)
                yield self.hold(hook_t.sample())
                yield from self._move(COORD[pid], DIST[pid]/speed_dist.sample())
                yield self.hold(hook_t.sample())
                yield self.to_store(empty_pl[pid], tr)
                self._detach()

                # carga: planta → CD
                t0 = env.now()
                tr = yield self.from_store(loaded[pid])
                self._attach(tr)
                dt = env.now() - t0
                self.wait_pl += dt       # espera en planta
                if dt > 0:
                    self.no_full += 1
                yield self.hold(hook_t.sample())
                yield from self._move(CD_PT[pid], DIST[pid]/speed_dist.sample())

    # ─── inventario inicial (3×3 = 9 trailers) ──────────────
    trucks: List[Truck] = []
    for i in range(plants):
        Loader(name=f"LoaderP{i+1}", pid=i)
        empty_pl[i].add(trailer(f"P{i+1}_emptyPL", (COORD[i][0]-30, COORD[i][1]-25), i))
        empty_cd[i].add(trailer(f"P{i+1}_emptyCD", CD_PT[i], i))
        first_loaded = trailer(f"P{i+1}_loaded", (COORD[i][0]+30, COORD[i][1]-10), i, loaded_flag=True)
        trucks.append(Truck(name=f"T{i+1}", idx=i, home_pid=i,
                            first_trailer=first_loaded, x_shift=-20))

    t4_tr = empty_pl[0].pop()
    set_loaded(t4_tr)
    trucks.append(Truck(name="T4", idx=3, home_pid=0,
                        first_trailer=t4_tr, x_shift=20))

    # ─── Unloader ─────────────────────────
    Unloader(name="UnloaderCD")

    # ─── ejecutar ─────────────────────────
    env.run(till=hours*60)

    # ─── retorno métricas ──────────────────
    horizon = hours*60
    return {
        "truck_trips": [t.trips for t in trucks],
        "wait_pl":     [t.wait_pl for t in trucks],
        "wait_cd":     [t.wait_cd for t in trucks],
        "no_empty":    [t.no_empty for t in trucks],
        "no_full":     [t.no_full for t in trucks],
        "delivered":   delivered,
        "selected":    selected,
        "util_%":      [100*(horizon-(t.wait_pl+t.wait_cd))/horizon for t in trucks]
    }

# ------------------------------------------------------------------
#           GUARDAR RESULTADOS EN CSV
# ------------------------------------------------------------------
def monte_carlo_multi_beta(
    betas      = (0, 0.5, 1, 2),
    n_runs     = 100,
    hours      = 8,
    out_csv    = "sim_results_4trucks_planta.csv",
    seed_base  = 0,
    animate    = False,
    debug      = False
):
    rows = []
    horizon = hours * 60
    plants = 3

    for beta in betas:
        for it in range(n_runs):
            res = run_sim(
                hours   = hours,
                beta    = beta,
                seed    = seed_base + it,
                animate = animate,
                debug   = debug
            )

            n_trucks = len(res["truck_trips"])

            # — Filas por camión —
            for t in range(n_trucks):
                rows.append({
                    "entity"    : "truck",
                    "iter"      : it,
                    "beta"      : beta,
                    "truck"     : t + 1,
                    "plant"     : None,
                    "trips"     : res["truck_trips"][t],
                    "delivered" : res["truck_trips"][t],
                    "selected"  : None,
                    "wait_pl"   : round(res["wait_pl"][t], 2),
                    "wait_cd"   : round(res["wait_cd"][t], 2),
                    "no_empty"  : res["no_empty"][t],
                    "no_full"   : res["no_full"][t],
                    "util_%"    : round(res["util_%"][t], 2)
                })

            # — Filas por planta —
            for p in range(plants):
                rows.append({
                    "entity"    : "plant",
                    "iter"      : it,
                    "beta"      : beta,
                    "truck"     : None,
                    "plant"     : p + 1,
                    "trips"     : res["delivered"][p],
                    "delivered" : res["delivered"][p],
                    "selected"  : res["selected"][p],
                    "wait_pl"   : None,
                    "wait_cd"   : None,
                    "no_empty"  : None,
                    "no_full"   : None,
                    "util_%"    : None
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"✔ CSV generado → {out_csv}  ({len(df)} filas)")

if __name__ == "__main__":
    monte_carlo_multi_beta()
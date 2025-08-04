"""
DISTRIMAX · Escenario 2 – Tractores libres
score = ETA + β·(delivered – mínimo entregado)
-----------------------------------------------------------------------
Inventario inicial por planta (3 trailers):
  · 1 loaded enganchado al tractor
  · 1 empty  en la planta
  · 1 empty  en el CD
"""
import salabim as sim, random
from math import hypot
from typing import List, Dict, Optional
import numpy as np

# ──────────  visual ──────────
CARRY_OFFSET   = (0, -12)
TRAILER_COLORS = ["lightblue", "lightgreen", "pink"]

def set_loaded(tr):
    pid = tr.plant_id
    tr.loaded = True
    tr.name(f"P{pid+1}_loaded")

def set_empty_cd(tr):
    pid = tr.plant_id
    tr.loaded = False
    tr.name(f"P{pid+1}_emptyCD")

# ──────────  simulación ───────
def run_sim(hours: float        = 8,
            seed:  Optional[int] = None,
            plants:int           = 3,
            beta:  float         = 3,
            AVG_SPEED: float     = 35,        
            animate: bool        = False,
            trace:   bool        = False,
            debug:   bool        = False) -> Dict:
    
    random.seed(seed); np.random.seed(seed); sim.random_seed(seed)
    sim.yieldless(False)


    # ── entorno
    env = sim.Environment(time_unit="minutes",
                          width=1200, height=700,
                          animate=animate, trace=trace)

    env.random_seed(seed)                       
    
    # ── distribuciones
    load_t   = sim.Uniform(8, 16)   # carga
    unload_t = sim.Uniform(4, 8)   # descarga
    hook_t   = sim.Uniform(0.5, 1)     # enganche
    speed_dist = sim.Bounded(sim.Normal(40, 3), lowerbound=15)
    if animate:
        env.animation_parameters(background_color="90%gray")
        env.speed(4)

    def log(msg):
        if debug:
            print(f"[{env.now():6.2f}] {msg}")

    # ── geometría
    CD_x, CD_y = env.width()//2, int(env.height()*0.25)
    half  = 150
    CD_PT = {i:(CD_x+(-1+i)*half, CD_y) for i in range(plants)}
    P_X , P_YF = [350,650,900], [0.57,0.68,0.65]
    COORD = {i:(P_X[i], int(env.height()*P_YF[i])) for i in range(plants)}

    DIST = [hypot(CD_PT[p][0]-COORD[p][0],
                  CD_PT[p][1]-COORD[p][1]) for p in range(plants)]
    ETA_EST = [d/AVG_SPEED for d in DIST]

    # ── dibujo fijo
    if animate:
        cd_w, cd_h = 300, 10
        sx, sy = CD_x-cd_w//2, CD_y-cd_h//2
        sim.AnimatePolygon([sx,sy,sx+cd_w,sy,sx+cd_w,sy+cd_h,sx,sy+cd_h],
                           fillcolor="orange", layer=0)
        sim.AnimateText("CD", x=CD_x-12, y=CD_y+cd_h//2+12,
                        textcolor="black", layer=0)
        for i in range(plants):
            x, y = COORD[i]
            sim.AnimateCircle(20, x=x, y=y, fillcolor="lightgray", layer=0)
            sim.AnimateText(f"P{i+1}", x=x-10, y=y+42,
                            textcolor="black", layer=0)

    # ── recursos y colas
    dock_cd   = sim.Resource("Dock_CD", capacity=1)
    loaded    = [sim.Store(f"Loaded_P{i+1}")   for i in range(plants)]
    empty_pl  = [sim.Store(f"Empty_P{i+1}")    for i in range(plants)]
    empty_cd  = [sim.Store(f"Empty_CD_P{i+1}") for i in range(plants)]
    loaded_cd = sim.Store("Loaded_CD")

    delivered = [0]*plants
    selected  = [0]*plants
    all_trucks : List["Truck"] = []

    # acumulador de tiempo ocupado del muelle
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

    # ── Loader
    loaders=[]
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
                log(f"Loader P{self.pid+1} ← {tr.name()}")
                d = load_t.sample()
                log(f"  carga {d:.2f} min")
                self.busy_until = env.now() + d
                yield self.hold(d)
                set_loaded(tr)
                log(f"Loader P{self.pid+1} → cargado {tr.name()}")
                yield self.to_store(loaded[self.pid], tr)
                wake()

    # ── Unloader
    class Unloader(sim.Component):
        def process(self):
            while True:
                tr = yield self.from_store(loaded_cd)
                log(f"Unloader ↙ {tr.name()}")
                d = unload_t.sample()
                log(f"  descarga {d:.2f} min")
                yield self.hold(d)
                delivered[tr.plant_id] += 1
                log(f"Unloader ↗ {tr.name()}   entregados P{tr.plant_id+1}={delivered[tr.plant_id]}")
                set_empty_cd(tr)
                yield self.to_store(empty_cd[tr.plant_id], tr)
                wake()

    # ── Truck
    class Truck(sim.Component):
        def setup(self, idx, home_pid, first_trailer):
            self.idx   = idx
            self.home  = home_pid
            self.carry = first_trailer
            self.trips = 0
            self.first_trip = True
            # métricas
            self.wait_pl = 0.0
            self.wait_cd = 0.0
            self.no_empty = 0
            self.no_full  = 0
            all_trucks.append(self)
            # sprite
            x, y = COORD[home_pid]
            self.sprite = sim.AnimateCircle(
                18, x=x, y=y-50, fillcolor="gray",
                text=f"T{idx+1}", textcolor="white", layer=5)
            first_trailer.sprite.x = x + CARRY_OFFSET[0]
            first_trailer.sprite.y = y-50 + CARRY_OFFSET[1]

        # helpers animación
        def _attach(self, tr):
            self.carry = tr
            tr.sprite.x = self.sprite.x(env.now()) + CARRY_OFFSET[0]
            tr.sprite.y = self.sprite.y(env.now()) + CARRY_OFFSET[1]
        def _detach(self): self.carry = None
        def _move(self, dst, dur, steps=40):
            x, y = self.sprite.x(env.now()), self.sprite.y(env.now())
            dx, dy, dt = (dst[0]-x)/steps, (dst[1]-y)/steps, dur/steps
            for _ in range(steps):
                x += dx;  y += dy
                self.sprite.x, self.sprite.y = x, y
                if self.carry:
                    self.carry.sprite.x = x + CARRY_OFFSET[0]
                    self.carry.sprite.y = y + CARRY_OFFSET[1]
                yield self.hold(dt)

        # elegir planta
        def choose(self):
            best = None
            min_deliv = min(delivered)
            for p in range(plants):
                if empty_cd[p].length() == 0:
                    continue
                eta   = ETA_EST[p] + max(0, loaders[p].rem() - ETA_EST[p])
                score = eta + beta * (delivered[p] - min_deliv)
                if best is None or score < best[1]:
                    best = (p, score)
            return None if best is None else best[0]

        def process(self):
            while True:
                # primer viaje
                if self.first_trip:
                    pid = self.home
                    yield self.hold(hook_t.sample())
                    yield from self._move(CD_PT[pid], DIST[pid]/speed_dist.sample())
                    start_dock = env.now()
                    yield self.request(dock_cd)
                    yield self.hold(hook_t.sample())
                    yield self.to_store(loaded_cd, self.carry)
                    self._detach()
                    self.release(dock_cd)
                    dock_busy_time[0] += env.now() - start_dock
                    self.first_trip = False; self.trips = 1
                    wake(); continue

                pid = self.choose()
                if pid is None:
                    t0 = env.now()
                    yield self.passivate()
                    dt = env.now() - t0
                    self.wait_cd += dt
                    if dt > 0: self.no_empty += 1
                    continue
                selected[pid] += 1

                # CD → planta
                tr = yield self.from_store(empty_cd[pid]); self._attach(tr)
                yield self.hold(hook_t.sample())
                yield from self._move(COORD[pid], DIST[pid]/speed_dist.sample())
                yield self.hold(hook_t.sample())
                yield self.to_store(empty_pl[pid], tr); self._detach()

                # esperar cargado
                t0 = env.now()
                tr = yield self.from_store(loaded[pid]); self._attach(tr)
                dt = env.now() - t0
                self.wait_pl += dt
                if dt > 0: self.no_full += 1

                # planta → CD
                yield self.hold(hook_t.sample())
                yield from self._move(CD_PT[pid], DIST[pid]/speed_dist.sample())
                start_dock = env.now()
                yield self.request(dock_cd); yield self.hold(hook_t.sample())
                yield self.to_store(loaded_cd, tr); self._detach(); self.release(dock_cd)
                dock_busy_time[0] += env.now() - start_dock
                self.trips += 1
                wake()

    # ── inventario inicial
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
            Truck(name=f"T{i+1}", idx=i, home_pid=i,
                  first_trailer=first_loaded))

    Unloader(name="UnloaderCD")
    env.run(till=hours*60)

    # salida compacta 
    return {
        "truck_trips":[t.trips for t in trucks],
        "wait_pl":[t.wait_pl for t in trucks],
        "wait_cd":[t.wait_cd for t in trucks],
        "no_empty":[t.no_empty for t in trucks],
        "no_full":[t.no_full for t in trucks],
        "delivered":delivered,
        "selected":selected,
        "queues":{
            "loaded":[s.length() for s in loaded],
            "empty_pl":[s.length() for s in empty_pl],
            "empty_cd":[s.length() for s in empty_cd]}
    }

# ------------------------------------------------------------------
#        GUARDAR RESULTADOS EN CSV
# ------------------------------------------------------------------
import pandas as pd

def monte_carlo_multi_beta(betas=(0,0.5,1,2), n_runs=100,
                           hours=8, out_csv="sim_results_betas.csv"):
    rows=[]
    for beta in betas:
        for it in range(n_runs):
            res = run_sim(hours=hours, beta=beta, seed=it, animate=False)
            horizon = hours*60
            # ── filas por camión ───────────────────────────────────────
            for t in range(3):
                util = 100*(horizon - (res["wait_pl"][t]+res["wait_cd"][t]))/horizon
                rows.append({
                    "entity":"truck",
                    "iter":it,
                    "beta":beta,
                    "truck":t+1,
                    "plant":t+1,
                    "trips":res["truck_trips"][t],
                    "delivered":res["delivered"][t],
                    "selected":res["selected"][t],
                    "wait_pl":round(res["wait_pl"][t],2),
                    "wait_cd":round(res["wait_cd"][t],2),
                    "no_empty":res["no_empty"][t],
                    "no_full":res["no_full"][t],
                    "util_%":round(util,2),
                    "queue_loaded":res["queues"]["loaded"][t],
                    "queue_empty_p":res["queues"]["empty_pl"][t],
                    "queue_empty_cd":res["queues"]["empty_cd"][t]
                })
            # ── filas por planta ───────────────────────────────────────
            for p in range(3):
                rows.append({
                    "entity":"plant",
                    "iter":it,
                    "beta":beta,
                    "truck":None,
                    "plant":p+1,
                    "trips":None,
                    "delivered":res["delivered"][p],
                    "selected":res["selected"][p],
                    "wait_pl":None,
                    "wait_cd":None,
                    "no_empty":None,
                    "no_full":None,
                    "util_%":None,
                    "queue_loaded":res["queues"]["loaded"][p],
                    "queue_empty_p":res["queues"]["empty_pl"][p],
                    "queue_empty_cd":res["queues"]["empty_cd"][p]
                })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✔ CSV generado → {out_csv}  ({len(rows)} filas)")

# ── ejecución principal ─────────────────────────────────────────
if __name__ == "__main__":
    monte_carlo_multi_beta()      
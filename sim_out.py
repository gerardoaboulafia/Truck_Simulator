# Escenario 3 × 3 × 1 — métricas ampliadas
# --------------------------------------------------------------------------------------
# • viajes por camión · entregados · esperas · %utilización · "sin vacío / sin lleno"
# --------------------------------------------------------------------------------------

import salabim as sim, random, numpy as np
from math import hypot
import pandas as pd

# ───────── visual ─────────
CARRY_OFFSET = (0, -12)
TRUCK_COLORS = ["blue", "green", "red"]
TRAILER_COLORS = ["lightblue", "lightgreen", "pink"]
TRAILER_ALPHA = 150

def set_loaded(tr):  tr.loaded = True;  tr.name(f"P{tr.plant_id+1}_loaded")
def set_empty_cd(tr): tr.loaded = False; tr.name(f"P{tr.plant_id+1}_emptyCD")

# ───────── simulación ────────
def run_sim(hours=8, seed=0, plants=3,
            animate=False, trace=False, debug=False):
    # 1 semillas
    random.seed(seed); np.random.seed(seed); sim.random_seed(seed)
    sim.yieldless(False)

    # 2 entorno
    env = sim.Environment(time_unit="minutes", width=1200, height=700,
                          animate=animate, trace=trace)
    env.random_seed(seed)

    # 3 distribuciones
    load_t   = sim.Uniform(8, 16)
    unload_t = sim.Uniform(4, 8)
    hook_t   = sim.Uniform(0.5, 1)
    speed_d  = sim.Bounded(sim.Normal(40, 3), lowerbound=15)

    # ── geometría
    CD_x, CD_y = env.width()//2, int(env.height()*0.25)
    half=150
    CD_PT={i:(CD_x+(-1+i)*half,CD_y) for i in range(plants)}
    P_X,P_YF=[350,650,900],[0.55,0.68,0.70]
    COORD={i:(P_X[i],int(env.height()*P_YF[i])) for i in range(plants)}
    def t_travel(pid):
        d=hypot(CD_PT[pid][0]-COORD[pid][0], CD_PT[pid][1]-COORD[pid][1])
        return d/speed_d.sample()

    # ── recursos
    dock_cd   = sim.Resource(capacity=1)
    loaded    = [sim.Store() for _ in range(plants)]
    empty_pl  = [sim.Store() for _ in range(plants)]
    empty_cd  = [sim.Store() for _ in range(plants)]
    loaded_cd = sim.Store()

    delivered=[0]*plants

    # helper trailer
    def trailer(pid, loaded_flag=False):
        tr=sim.Component(); tr.plant_id=pid; tr.loaded=loaded_flag
        tr.sprite=sim.AnimateCircle(10, x=0,y=0,
                                    fillcolor=(TRAILER_COLORS[pid],TRAILER_ALPHA),
                                    layer=2)
        return tr

    # Loader
    class Loader(sim.Component):
        def setup(self,pid): self.pid=pid
        def process(self):
            while True:
                tr=yield self.from_store(empty_pl[self.pid])
                yield self.hold(load_t.sample()); set_loaded(tr)
                yield self.to_store(loaded[self.pid], tr)

    # Unloader
    class Unloader(sim.Component):
        def process(self):
            while True:
                tr=yield self.from_store(loaded_cd)
                yield self.hold(unload_t.sample())
                delivered[tr.plant_id]+=1; set_empty_cd(tr)
                yield self.to_store(empty_cd[tr.plant_id], tr)

    # Truck
    class Truck(sim.Component):
        def setup(self,pid,first_tr):
            self.pid=pid; self.carry=first_tr
            self.trips=0; self.wait_pl=self.wait_cd=0
            self.no_empty=self.no_full=0
            x,y=COORD[pid]
            sim.AnimateCircle(18,x=x,y=y-50,fillcolor=TRUCK_COLORS[pid],
                              text=f"T{pid+1}",textcolor="white",layer=5)
        def _attach(self,tr): self.carry=tr
        def _detach(self):    self.carry=None
        def _move(self,dst,dur,steps=40):
            dt=dur/steps
            for _ in range(steps): yield self.hold(dt)
        def process(self):
            yield self.hold(hook_t.sample())
            yield from self._move(CD_PT[self.pid],t_travel(self.pid))
            yield self.request(dock_cd); yield self.hold(hook_t.sample())
            yield self.to_store(loaded_cd,self.carry); self._detach()
            self.release(dock_cd); self.trips=1
            while True:
                # CD→planta vacío
                t0=env.now(); tr=yield self.from_store(empty_cd[self.pid])
                self.wait_cd+=env.now()-t0
                if env.now()-t0>0: self.no_empty+=1
                self._attach(tr); yield self.hold(hook_t.sample())
                yield from self._move(COORD[self.pid],t_travel(self.pid))
                yield self.hold(hook_t.sample())
                yield self.to_store(empty_pl[self.pid],tr); self._detach()
                # planta→CD cargado
                t0=env.now(); tr=yield self.from_store(loaded[self.pid])
                self.wait_pl+=env.now()-t0
                if env.now()-t0>0: self.no_full+=1
                self._attach(tr); yield self.hold(hook_t.sample())
                yield from self._move(CD_PT[self.pid],t_travel(self.pid))
                yield self.request(dock_cd); yield self.hold(hook_t.sample())
                yield self.to_store(loaded_cd,tr); self._detach()
                self.release(dock_cd); self.trips+=1

    # instancias iniciales
    loaders=[Loader(pid=i) for i in range(plants)]
    trucks=[]
    for i in range(plants):
        empty_pl[i].add(trailer(i,False))
        empty_cd[i].add(trailer(i,False))
        trucks.append(Truck(pid=i, first_tr=trailer(i,True)))
    Unloader(); env.run(hours*60)

    return {
        "truck_trips":[t.trips for t in trucks],
        "wait_pl":[t.wait_pl for t in trucks],
        "wait_cd":[t.wait_cd for t in trucks],
        "no_empty":[t.no_empty for t in trucks],
        "no_full":[t.no_full for t in trucks],
        "delivered":delivered,
        "queues":{
            "loaded":[s.length() for s in loaded],
            "empty_pl":[s.length() for s in empty_pl],
            "empty_cd":[s.length() for s in empty_cd]}
    }

# ─────────── Monte-Carlo → CSV ───────────
def monte_carlo(n_runs=100, hours=8, out_csv="sim_results_h1.csv"):
    rows=[]
    for it in range(n_runs):
        res=run_sim(seed=it, hours=hours, animate=False)
        horizon=hours*60
        # filas por camión
        for t in range(3):
            util=100*(horizon-(res["wait_pl"][t]+res["wait_cd"][t]))/horizon
            rows.append({"entity":"truck","iter":it,
                         "truck":t+1,"plant":t+1,
                         "trips":res["truck_trips"][t],
                         "delivered":res["delivered"][t],
                         "wait_pl":res["wait_pl"][t],
                         "wait_cd":res["wait_cd"][t],
                         "no_empty":res["no_empty"][t],
                         "no_full":res["no_full"][t],
                         "util_%":round(util,2),
                         "queue_loaded":res["queues"]["loaded"][t],
                         "queue_empty_p":res["queues"]["empty_pl"][t],
                         "queue_empty_cd":res["queues"]["empty_cd"][t]})
        # filas por planta
        for p in range(3):
            rows.append({"entity":"plant","iter":it,
                         "truck":None,"plant":p+1,
                         "trips":None,
                         "delivered":res["delivered"][p],
                         "wait_pl":None,"wait_cd":None,
                         "no_empty":None,"no_full":None,"util_%":None,
                         "queue_loaded":res["queues"]["loaded"][p],
                         "queue_empty_p":res["queues"]["empty_pl"][p],
                         "queue_empty_cd":res["queues"]["empty_cd"][p]})
    pd.DataFrame(rows).to_csv(out_csv,index=False)
    print("✔ CSV listo:", out_csv, "| Filas:", len(rows))

if __name__=="__main__":
    monte_carlo()
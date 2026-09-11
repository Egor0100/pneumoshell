"""Генерация обучающей выборки с помощью классического решателя.

Входные параметры (признаки) одного расчётного случая:
    p0_1..p0_N — давления зарядки ярусов, Па;
    pc         — боковое давление (давление в воздушной подушке), Па;
    delta      — обжатие (подъём опорной поверхности относительно нижней
                 точки оболочки в отсчётном состоянии), м.
Выход (цель): координаты фиксированного набора материальных точек сечения
и интегральные характеристики (давления, длина контакта, натяжения).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .geometry import DESIGNS, ShellDesign
from .solver import ShellSolver


@dataclass
class ParamSpace:
    p0_min: float = 4000.0
    p0_max: float = 10000.0
    pc_min: float = 500.0
    pc_max: float = 2000.0
    delta_min: float = -0.05
    delta_max: float = 0.25

    def bounds(self, n_tiers):
        lo = [self.p0_min] * n_tiers + [self.pc_min, self.delta_min]
        hi = [self.p0_max] * n_tiers + [self.pc_max, self.delta_max]
        return np.array(lo), np.array(hi)

    def feature_names(self, n_tiers):
        return [f"p0_{i+1}" for i in range(n_tiers)] + ["pc", "delta"]


def sample_inputs(n_tiers, n, seed=0, space: ParamSpace = ParamSpace()):
    """Латинский гиперкуб (LHS) в пространстве входных параметров."""
    lo, hi = space.bounds(n_tiers)
    sampler = qmc.LatinHypercube(d=len(lo), seed=seed)
    return qmc.scale(sampler.random(n), lo, hi)


_WORKER = {}


def _init_worker(design_name):
    _WORKER["solver"] = ShellSolver(DESIGNS[design_name]())


def solve_case(X_row, solver: ShellSolver = None):
    S = solver or _WORKER["solver"]
    N = S.N
    p0 = X_row[:N]
    pc, delta = X_row[N], X_row[N + 1]
    t0 = time.perf_counter()
    try:
        sol = S.solve(p0, pc, delta)
    except Exception as exc:  # pragma: no cover - защитная ветка
        return dict(ok=False, error=str(exc))
    t = time.perf_counter() - t0
    out = dict(ok=bool(sol.converged and sol.valid), converged=sol.converged, valid=sol.valid,
               time=t, nfev=sol.n_fev)
    if not out["ok"]:
        return out
    segs = sol.segments
    out["points"] = S.material_points(segs).ravel()
    out["p"] = sol.p.copy()
    out["contact"] = sol.contact_length
    out["areas"] = sol.areas.copy()
    names = [k for k in segs if not k.startswith("M_")] + ["M_out"]
    out["T"] = np.array([segs[k].T for k in names])
    nc = S._n_core()
    x = sol.x
    out["xsol"] = np.concatenate([x[:nc + N], [sol.contact_length]])
    out["y_low"] = sol.info["yE"]
    return out


def generate(design_name: str, n: int, seed: int = 0, processes: int = 2, verbose=True):
    design: ShellDesign = DESIGNS[design_name]()
    X = sample_inputs(design.n_tiers, n, seed=seed)
    t0 = time.perf_counter()
    if processes > 1:
        with mp.Pool(processes, initializer=_init_worker, initargs=(design_name,)) as pool:
            res = []
            for k, r in enumerate(pool.imap(solve_case, X, chunksize=8)):
                res.append(r)
                if verbose and (k + 1) % 250 == 0:
                    print(f"  {design_name}: {k+1}/{n}  {time.perf_counter()-t0:.0f} s", flush=True)
    else:
        _init_worker(design_name)
        res = [solve_case(x) for x in X]
    ok = np.array([r["ok"] for r in res])
    data = dict(
        X=X[ok],
        Y=np.stack([r["points"] for r in res if r["ok"]]),
        P=np.stack([r["p"] for r in res if r["ok"]]),
        L=np.array([r["contact"] for r in res if r["ok"]]),
        A=np.stack([r["areas"] for r in res if r["ok"]]),
        T=np.stack([r["T"] for r in res if r["ok"]]),
        XS=np.stack([r["xsol"] for r in res if r["ok"]]),
        YLOW=np.array([r["y_low"] for r in res if r["ok"]]),
        time=np.array([r["time"] for r in res if r["ok"]]),
        nfev=np.array([r["nfev"] for r in res if r["ok"]]),
        X_failed=X[~ok],
        n_requested=n,
        wall_time=time.perf_counter() - t0,
    )
    if verbose:
        print(f"{design_name}: ok {ok.sum()}/{n}, wall {data['wall_time']:.0f} s, "
              f"mean solve {data['time'].mean():.3f} s", flush=True)
    return data

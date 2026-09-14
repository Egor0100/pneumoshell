"""Обучение, подбор гиперпараметров и оценка ML-моделей.

Результаты сохраняются в results/metrics_<design>.json, обученные модели —
в models/. Пример:  python scripts/train_evaluate.py --designs two_tier three_tier
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pneumoshell.geometry import DESIGNS  # noqa: E402
from pneumoshell.ml import (MODEL_TITLES, ShapeSurrogate, TargetCodec, load_dataset,  # noqa: E402
                            physics_metrics, shape_metrics)
from pneumoshell.solver import ShellSolver  # noqa: E402

MODELS = ["linear", "poly3", "knn", "rf", "gbm", "gpr", "mlp"]

MLP_GRID = [
    dict(hidden=(64, 64), activation="tanh", solver="lbfgs"),
    dict(hidden=(128, 128), activation="tanh", solver="lbfgs"),
    dict(hidden=(64, 64, 64), activation="tanh", solver="lbfgs"),
    dict(hidden=(128, 128, 128), activation="tanh", solver="adam", lr=2e-3),
    dict(hidden=(256, 256), activation="relu", solver="adam", lr=1e-3),
]


def log(*a):
    print(*a, flush=True)


def reference_height(design):
    S = ShellSolver(design)
    ref = S.solve_reference(np.full(design.n_tiers, 5000.0))
    return -ref.info["yE"]


def eval_surrogate(sur, split, design, H, with_physics=True, timing=True):
    # прогноз замеряется в однопоточном режиме (накладные расходы joblib искажают задержку)
    for obj in (sur.model, getattr(sur.model, "steps", [[None, None]])[-1][1]):
        if obj is not None and hasattr(obj, "n_jobs"):
            obj.n_jobs = None
    t0 = time.perf_counter()
    Y, P, L = sur.predict(split.X)
    t_batch = (time.perf_counter() - t0) / len(split.X)
    # задержка одиночного запроса
    ts = [0.0]
    for i in range(min(100, len(split.X)) if timing else 0):
        t1 = time.perf_counter()
        sur.predict(split.X[i:i + 1])
        ts.append(time.perf_counter() - t1)
    if timing:
        ts = ts[1:]
    m = shape_metrics(split.Y, Y, H)
    m["p_mape_pct"] = float(np.mean(np.abs(P - split.P) / split.P) * 100)
    m["contact_mae_mm"] = float(np.mean(np.abs(L - split.L)) * 1000)
    m["t_batch_ms"] = t_batch * 1000
    m["t_single_ms"] = float(np.median(ts) * 1000)
    if with_physics:
        m.update(physics_metrics(design, split.Y, Y))
    return m, (Y, P, L)


def pca_study(split_tr, split_te, ks=(2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)):
    out = []
    for k in ks:
        c = TargetCodec(k).fit(split_tr.Y, split_tr.P, split_tr.L)
        Yr = c.pca.inverse_transform(c.pca.transform(split_te.Y))
        E = np.linalg.norm((Yr - split_te.Y).reshape(len(Yr), -1, 2), axis=2)
        out.append(dict(k=k, explained=c.explained, rmse_mm=float(np.sqrt(np.mean(E ** 2)) * 1000),
                        max_mm=float(E.max() * 1000)))
    return out


def run_design(name, args):
    design = DESIGNS[name]()
    data = load_dataset(os.path.join(ROOT, "data", f"dataset_{name}.npz"), seed=0)
    tr, va, te = data["train"], data["val"], data["test"]
    H = reference_height(design)
    res = dict(design=name, n_train=len(tr.X), n_val=len(va.X), n_test=len(te.X), height_m=H,
               solver_time_mean_s=float(np.concatenate([tr.time, va.time, te.time]).mean()),
               solver_time_median_s=float(np.median(np.concatenate([tr.time, va.time, te.time]))),
               n_outputs=int(tr.Y.shape[1]))
    raw = np.load(os.path.join(ROOT, "data", f"dataset_{name}.npz"))
    res["n_requested"] = int(raw["n_requested"])
    res["n_failed"] = int(len(raw["X_failed"]))
    res["generation_wall_s"] = float(raw["wall_time"])
    res["contact_fraction"] = float(np.mean(raw["L"] > 0))
    log(f"== {name}: train {len(tr.X)} val {len(va.X)} test {len(te.X)}, H = {H:.3f} m")

    # 1. PCA
    res["pca"] = pca_study(tr, te)
    log("PCA:", [(r["k"], round(r["rmse_mm"], 3)) for r in res["pca"]])
    K = args.k

    # 2. подбор гиперпараметров MLP на валидационной выборке
    grid = []
    best = None
    cache = os.path.join(ROOT, "results", f"grid_cache_{name}.json")
    cached = json.load(open(cache)) if os.path.exists(cache) else None
    for cfg in (MLP_GRID if cached is None else []):
        sur = ShapeSurrogate.train(name, "mlp", tr, n_components=K, **cfg)
        m, _ = eval_surrogate(sur, va, design, H, with_physics=False)
        grid.append(dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                         val_rmse_mm=m["rmse_mm"], fit_s=sur.fit_time,
                         n_iter=int(sur.model.n_iter_)))
        log(f"  MLP {cfg}: val RMSE {m['rmse_mm']:.3f} mm, fit {sur.fit_time:.0f} s, iters {sur.model.n_iter_}")
        if best is None or m["rmse_mm"] < best[0]:
            best = (m["rmse_mm"], cfg)
    if cached is not None:
        grid = cached["grid"]
        g = min(grid, key=lambda q: q["val_rmse_mm"])
        best = (g["val_rmse_mm"], {k: (tuple(v) if isinstance(v, list) else v) for k, v in g["cfg"].items()})
        log("  MLP grid loaded from cache:", [(q["cfg"]["hidden"], q["val_rmse_mm"]) for q in grid])
    res["mlp_grid"] = grid
    best_cfg = best[1]
    res["mlp_best"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_cfg.items()}
    # абляция: покомпонентная стандартизация кода PCA вместо общего масштаба
    if cached is not None:
        res["ablation_standard_scaling_val_rmse_mm"] = cached["ablation"]
    else:
        sur = ShapeSurrogate.train(name, "mlp", tr, n_components=K, scaling="standard", **best_cfg)
        m, _ = eval_surrogate(sur, va, design, H, with_physics=False)
        res["ablation_standard_scaling_val_rmse_mm"] = m["rmse_mm"]
    m = dict(rmse_mm=res["ablation_standard_scaling_val_rmse_mm"])
    log(f"  ablation (per-component scaling): val RMSE {m['rmse_mm']:.3f} mm")

    # 3. сравнение моделей на тестовой выборке
    res["models"] = {}
    preds = {}
    for mn in MODELS:
        kw = best_cfg if mn == "mlp" else {}
        sur = ShapeSurrogate.train(name, mn, tr, n_components=K, **kw)
        m, pr = eval_surrogate(sur, te, design, H)
        m["fit_s"] = sur.fit_time
        m["title"] = MODEL_TITLES[mn]
        res["models"][mn] = m
        preds[mn] = pr
        log(f"  {mn:7s} RMSE {m['rmse_mm']:.3f} mm  max {m['max_mm']:.2f} mm  R2 {m['r2']:.6f} "
            f"p {m['p_mape_pct']:.3f}%  l {m['contact_mae_mm']:.2f} mm  area {m['area_mape_pct']:.3f}%  "
            f"fit {sur.fit_time:.1f}s  t1 {m['t_single_ms']:.3f} ms")
        if mn in ("mlp", "gpr"):
            sur.save(os.path.join(ROOT, "models", f"{name}_{mn}.pkl"))
    np.savez_compressed(os.path.join(ROOT, "results", f"test_pred_{name}.npz"),
                        X=te.X, Y=te.Y, P=te.P, L=te.L,
                        **{f"Y_{k}": v[0] for k, v in preds.items()},
                        **{f"P_{k}": v[1] for k, v in preds.items()},
                        **{f"L_{k}": v[2] for k, v in preds.items()})

    # 4. кривые обучения
    res["learning_curve"] = {}
    sizes = [150, 300, 600, 1200]
    for mn in ["poly3", "gbm", "gpr", "mlp"]:
        rows = []
        for n in sizes:
            sub = _subset(tr, n)
            kw = best_cfg if mn == "mlp" else {}
            sur = ShapeSurrogate.train(name, mn, sub, n_components=K, **kw)
            m, _ = eval_surrogate(sur, te, design, H, with_physics=False, timing=False)
            rows.append(dict(n=n, rmse_mm=m["rmse_mm"], max_mm=m["max_mm"]))
        # полный объём — уже обученная основная модель
        rows.append(dict(n=len(tr.X), rmse_mm=res["models"][mn]["rmse_mm"], max_mm=res["models"][mn]["max_mm"]))
        res["learning_curve"][mn] = rows
        log(f"  LC {mn}:", [(r["n"], round(r["rmse_mm"], 3)) for r in rows])

    # 5. гибридная схема: ML-прогноз неизвестных решателя -> метод Ньютона
    res["hybrid"] = hybrid_study(name, design, tr, te, best_cfg, n_cases=args.hybrid_cases)
    log("  hybrid:", res["hybrid"]["summary"])

    with open(os.path.join(ROOT, "results", f"metrics_{name}.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _subset(split, n):
    from pneumoshell.ml import Split
    return Split(*(getattr(split, f)[:n] for f in ("X", "Y", "P", "L", "A", "XS", "time")))


def hybrid_study(name, design, tr, te, cfg, n_cases=300):
    """ML-модель прогнозирует вектор неизвестных решателя (натяжения, углы,
    давления, длину контакта); прогноз используется как начальное приближение
    для метода Ньютона. Сравнивается с расчётом «с нуля» (продолжение по
    параметру)."""
    xs = StandardScaler().fit(tr.X)
    ys = StandardScaler().fit(tr.XS)
    from pneumoshell.ml import make_model
    mlp = make_model("mlp", tr.X.shape[1], **cfg)
    mlp.fit(xs.transform(tr.X), ys.transform(tr.XS))
    XSp = ys.inverse_transform(mlp.predict(xs.transform(te.X)))
    S = ShellSolver(design)
    N = design.n_tiers
    rows = []
    for i in range(min(n_cases, len(te.X))):
        p0, pc, dl = te.X[i, :N], te.X[i, N], te.X[i, N + 1]
        S._ref_cache = {k: v for k, v in S._ref_cache.items() if k == "base"}
        t0 = time.perf_counter()
        cold = S.solve(p0, pc, dl)
        t_cold = time.perf_counter() - t0
        S._ref_cache = {k: v for k, v in S._ref_cache.items() if k == "base"}
        t0 = time.perf_counter()
        warm = S.solve(p0, pc, dl, x_init=XSp[i])
        t_warm = time.perf_counter() - t0
        err = np.linalg.norm((S.material_points(warm.segments) - S.material_points(cold.segments)),
                             axis=1).max() if (warm.converged and cold.converged) else np.nan
        rows.append(dict(t_cold=t_cold, t_warm=t_warm, nfev_cold=cold.n_fev, nfev_warm=warm.n_fev,
                         warm_direct=int(warm.info.get("nsteps", 99) == 1), ok=bool(warm.converged),
                         diff_mm=float(err * 1000)))
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    summary = dict(
        n=len(rows),
        t_cold_mean_ms=float(arr["t_cold"].mean() * 1000),
        t_warm_mean_ms=float(arr["t_warm"].mean() * 1000),
        speedup=float(arr["t_cold"].mean() / arr["t_warm"].mean()),
        nfev_cold_mean=float(arr["nfev_cold"].mean()),
        nfev_warm_mean=float(arr["nfev_warm"].mean()),
        direct_success_pct=float(arr["warm_direct"].mean() * 100),
        converged_pct=float(arr["ok"].mean() * 100),
        max_diff_mm=float(np.nanmax(arr["diff_mm"])),
    )
    return dict(summary=summary, t_cold=arr["t_cold"].tolist(), t_warm=arr["t_warm"].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["two_tier", "three_tier"])
    ap.add_argument("--k", type=int, default=24, help="число главных компонент")
    ap.add_argument("--hybrid-cases", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    for name in args.designs:
        run_design(name, args)


if __name__ == "__main__":
    main()

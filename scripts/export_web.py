"""Экспорт обученных MLP-моделей в JSON для работы в браузере (интерактивная
демонстрация). Прямой проход MLP + обратное PCA-преобразование реализуются на
JavaScript без сторонних библиотек.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pneumoshell.geometry import DESIGNS  # noqa: E402
from pneumoshell.ml import ShapeSurrogate  # noqa: E402
from pneumoshell.solver import ShellSolver  # noqa: E402


def r(a, nd=6):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def b64(a):
    """Массив float32 в base64 (компактное хранение весов)."""
    import base64
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).decode()


def export(name):
    d = DESIGNS[name]()
    sur = ShapeSurrogate.load(os.path.join(ROOT, "models", f"{name}_mlp.pkl"))
    m = sur.model
    S = ShellSolver(d)
    N = d.n_tiers
    # нижняя точка отсчётного состояния как функция давлений зарядки (квадратичная аппроксимация)
    rng = np.random.default_rng(0)
    P0 = rng.uniform(4000, 10000, (150, N))
    ylow = np.array([S.solve_reference(p).info["yE"] for p in P0])
    F = _quad_features(P0 / 1000.0)
    coef, *_ = np.linalg.lstsq(F, ylow, rcond=None)
    fit_err = float(np.abs(F @ coef - ylow).max())
    ref = S.solve_reference(np.full(N, 6000.0))
    T = np.load(os.path.join(ROOT, "results", f"test_pred_{name}.npz"))
    idx = np.linspace(0, len(T["X"]) - 1, 24).astype(int)
    return dict(
        name=name, n_tiers=N, b=d.b,
        layout=S.output_layout(),
        x_mean=r(sur.x_scaler.mean_), x_scale=r(sur.x_scaler.scale_),
        c_scale=float(sur.codec.c_scale), w_pl=float(sur.codec.w_pl),
        pl_mean=r(sur.codec.pl_scaler.mean_), pl_scale=r(sur.codec.pl_scaler.scale_),
        pca_mean=b64(sur.codec.pca.mean_), pca_components=b64(sur.codec.pca.components_),
        weights=[dict(shape=list(w.shape), data=b64(w)) for w in m.coefs_],
        biases=[b64(b) for b in m.intercepts_],
        activation=m.activation, n_components=sur.codec.n_components,
        ylow_coef=r(coef, 8), ylow_fit_err=fit_err,
        ref_points=r(S.material_points(ref.segments), 5),
        test_cases=[dict(x=r(T["X"][i], 4), y_true=r(T["Y"][i], 5), p_true=r(T["P"][i], 1),
                         l_true=float(T["L"][i])) for i in idx],
    )


def _quad_features(P):
    cols = [np.ones(len(P))]
    n = P.shape[1]
    for i in range(n):
        cols.append(P[:, i])
    for i in range(n):
        for j in range(i, n):
            cols.append(P[:, i] * P[:, j])
    return np.column_stack(cols)


def main():
    out = {n: export(n) for n in ["two_tier", "three_tier"]}
    path = os.path.join(ROOT, "results", "web_models.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("saved", path, os.path.getsize(path) // 1024, "KB",
          {k: v["ylow_fit_err"] for k, v in out.items()})


if __name__ == "__main__":
    main()

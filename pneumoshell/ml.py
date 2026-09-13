"""ML-модели для определения деформированной формы пневмооболочки.

Схема суррогатной модели:
    x = (p0_1..p0_N, pc, delta)  --StandardScaler-->  регрессор  -->  z
    z = [главные компоненты формы (PCA), давления p_i, длина контакта l]
    форма = PCA^{-1}(z_формы)  ->  координаты материальных точек сечения
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .geometry import DESIGNS
from .solver import ShellSolver


# ----------------------------------------------------------------------
# данные
# ----------------------------------------------------------------------
@dataclass
class Split:
    X: np.ndarray
    Y: np.ndarray       # координаты точек (n, 2K)
    P: np.ndarray       # давления после деформации (n, N)
    L: np.ndarray       # длина зоны контакта (n,)
    A: np.ndarray       # площади ярусов (n, N)
    XS: np.ndarray      # вектор неизвестных решателя (n, m)
    time: np.ndarray    # время решения классическим методом, с


def load_dataset(path, seed=0, frac=(0.70, 0.15, 0.15)):
    d = np.load(path)
    n = len(d["X"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(frac[0] * n)
    n_va = int(frac[1] * n)
    parts = dict(train=idx[:n_tr], val=idx[n_tr:n_tr + n_va], test=idx[n_tr + n_va:])
    out = {}
    for k, ii in parts.items():
        out[k] = Split(d["X"][ii], d["Y"][ii], d["P"][ii], d["L"][ii], d["A"][ii],
                       d["XS"][ii], d["time"][ii])
    return out


# ----------------------------------------------------------------------
# кодирование цели
# ----------------------------------------------------------------------
class TargetCodec:
    """Кодирует форму через PCA и добавляет давления и длину контакта.

    Коэффициенты PCA делятся на общий масштаб (СКО первой компоненты), а не
    стандартизируются покомпонентно: тогда квадратичная функция потерь в
    пространстве кода пропорциональна квадрату ошибки положения точек
    (базис PCA ортонормирован), и модель не тратит ресурс на шумовые
    компоненты малой дисперсии. Давления и длина контакта стандартизируются
    и входят в потери с весом w_pl.
    """

    def __init__(self, n_components=24, w_pl=0.5, scaling="global"):
        self.n_components = n_components
        self.w_pl = w_pl
        self.scaling = scaling   # "global" — общий масштаб; "standard" — покомпонентно

    def fit(self, Y, P, L):
        self.pca = PCA(n_components=self.n_components, svd_solver="full").fit(Y)
        C = self.pca.transform(Y)
        if self.scaling == "standard":
            self.c_scale = C.std(axis=0)
        else:
            self.c_scale = float(C[:, 0].std())
        self.pl_scaler = StandardScaler().fit(np.column_stack([P / 1000.0, L]))
        return self

    def encode(self, Y, P, L):
        C = self.pca.transform(Y) / self.c_scale
        Q = self.pl_scaler.transform(np.column_stack([P / 1000.0, L])) * self.w_pl
        return np.column_stack([C, Q])

    def decode(self, Z):
        k = self.n_components
        Y = self.pca.inverse_transform(Z[:, :k] * self.c_scale)
        Q = self.pl_scaler.inverse_transform(Z[:, k:] / self.w_pl)
        P = Q[:, :-1] * 1000.0
        L = np.maximum(Q[:, -1], 0.0)
        return Y, P, L

    @property
    def explained(self):
        return float(np.sum(self.pca.explained_variance_ratio_))


# ----------------------------------------------------------------------
# модели
# ----------------------------------------------------------------------
def make_model(name, n_features, random_state=0, **kw):
    """Фабрика регрессоров. Все модели многомерные по выходу."""
    if name == "linear":
        return LinearRegression()
    if name == "poly3":
        return make_pipeline(PolynomialFeatures(3), Ridge(alpha=kw.get("alpha", 1e-4)))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=kw.get("k", 6), weights="distance")
    if name == "rf":
        return RandomForestRegressor(n_estimators=kw.get("n_estimators", 300), min_samples_leaf=1,
                                     n_jobs=2, random_state=random_state)
    if name == "gbm":
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=kw.get("max_iter", 600), learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1e-3, random_state=random_state), n_jobs=2)
    if name == "gpr":
        kern = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(n_features), (1e-2, 1e2)) \
            + WhiteKernel(1e-5, (1e-9, 1e-1))
        return GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=1,
                                        random_state=random_state)
    if name == "mlp":
        solver = kw.get("solver", "lbfgs")
        common = dict(hidden_layer_sizes=tuple(kw.get("hidden", (128, 128))),
                      activation=kw.get("activation", "tanh"), alpha=kw.get("l2", 1e-6),
                      random_state=random_state)
        if solver == "lbfgs":
            return MLPRegressor(solver="lbfgs", max_iter=kw.get("max_iter", 20000),
                                max_fun=kw.get("max_fun", 60000), tol=1e-12, **common)
        return MLPRegressor(solver="adam", learning_rate_init=kw.get("lr", 2e-3),
                            batch_size=kw.get("batch", 64), max_iter=kw.get("max_iter", 6000),
                            tol=1e-9, n_iter_no_change=kw.get("patience", 150),
                            early_stopping=True, validation_fraction=0.1, **common)
    raise ValueError(name)


MODEL_TITLES = {
    "linear": "Линейная регрессия",
    "poly3": "Полиномиальная регрессия (3-я степень, Ridge)",
    "knn": "k ближайших соседей (k = 6)",
    "rf": "Случайный лес (300 деревьев)",
    "gbm": "Градиентный бустинг (HistGB)",
    "gpr": "Гауссовский процесс (ARD-RBF)",
    "mlp": "Многослойный перцептрон (MLP)",
}


@dataclass
class ShapeSurrogate:
    """Обученная суррогатная модель: признаки -> форма + давления + контакт."""

    design_name: str
    model_name: str
    x_scaler: StandardScaler
    codec: TargetCodec
    model: object
    fit_time: float = 0.0
    meta: Dict = field(default_factory=dict)

    @classmethod
    def train(cls, design_name, model_name, split: Split, n_components=24, gpr_max=2000,
              scaling="global", **kw):
        t0 = time.perf_counter()
        xs = StandardScaler().fit(split.X)
        codec = TargetCodec(n_components, scaling=scaling).fit(split.Y, split.P, split.L)
        Xtr = xs.transform(split.X)
        Ztr = codec.encode(split.Y, split.P, split.L)
        model = make_model(model_name, Xtr.shape[1], **kw)
        if model_name == "gpr" and len(Xtr) > gpr_max:
            # подбор гиперпараметров ядра на подвыборке, затем фиксированное ядро на всех данных
            sub = np.random.default_rng(0).choice(len(Xtr), gpr_max, replace=False)
            model.fit(Xtr[sub], Ztr[sub])
            model = GaussianProcessRegressor(model.kernel_, optimizer=None).fit(Xtr, Ztr)
        else:
            model.fit(Xtr, Ztr)
        return cls(design_name, model_name, xs, codec, model, time.perf_counter() - t0,
                   dict(n_train=len(Xtr), n_components=n_components, pca_explained=codec.explained))

    def predict(self, X):
        X = np.atleast_2d(X)
        Z = self.model.predict(self.x_scaler.transform(X))
        if Z.ndim == 1:
            Z = Z[:, None]
        return self.codec.decode(Z)

    def predict_shape(self, p0, pc, delta):
        """Удобный интерфейс для одного расчётного случая. Возвращает словарь
        с координатами точек по мембранам, давлениями и длиной контакта."""
        x = np.concatenate([np.atleast_1d(p0), [pc, delta]])[None, :]
        Y, P, L = self.predict(x)
        pts = Y[0].reshape(-1, 2)
        layout = ShellSolver(DESIGNS[self.design_name]()).output_layout()
        parts, k = {}, 0
        for nm, n in layout:
            parts[nm] = pts[k:k + n]
            k += n
        return dict(points=pts, membranes=parts, p=P[0], contact_length=float(L[0]))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "ShapeSurrogate":
        with open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------
# метрики
# ----------------------------------------------------------------------
def polygon_area(P):
    return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


def shape_metrics(Y_true, Y_pred, height):
    """Метрики качества формы по координатам материальных точек (в метрах)."""
    n = len(Y_true)
    E = np.linalg.norm((Y_pred - Y_true).reshape(n, -1, 2), axis=2)   # (n, K)
    rmse = float(np.sqrt(np.mean(E ** 2)))
    mae = float(np.mean(E))
    per_sample_max = E.max(axis=1)
    ss_res = np.sum((Y_pred - Y_true) ** 2)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0)) ** 2)
    return dict(
        rmse_mm=rmse * 1000, mae_mm=mae * 1000,
        max_mm=float(E.max()) * 1000,
        p95_max_mm=float(np.percentile(per_sample_max, 95)) * 1000,
        rel_rmse_pct=rmse / height * 100,
        r2=float(1 - ss_res / ss_tot),
    )


def physics_metrics(design, Y_true, Y_pred):
    """Физическая согласованность прогноза: ошибка площадей ярусов и
    нарушение нерастяжимости (относительное изменение длины ломаной)."""
    S = ShellSolver(design)
    lay = S.output_layout()
    n = len(Y_true)
    area_err, len_err = [], []
    for i in range(n):
        Pt = Y_true[i].reshape(-1, 2)
        Pp = Y_pred[i].reshape(-1, 2)
        pt_parts, pp_parts, k = {}, {}, 0
        for nm, m in lay:
            pt_parts[nm], pp_parts[nm] = Pt[k:k + m], Pp[k:k + m]
            k += m
        for parts_t, parts_p in [(pt_parts, pp_parts)]:
            At = _tier_areas(design, parts_t)
            Ap = _tier_areas(design, parts_p)
            area_err.append(np.abs(Ap - At) / At)
            for nm in parts_t:
                lt = np.sum(np.linalg.norm(np.diff(parts_t[nm], axis=0), axis=1))
                lp = np.sum(np.linalg.norm(np.diff(parts_p[nm], axis=0), axis=1))
                len_err.append(abs(lp - lt) / lt)
    area_err = np.array(area_err)
    return dict(area_mape_pct=float(area_err.mean() * 100),
                area_max_pct=float(area_err.max() * 100),
                length_mape_pct=float(np.mean(len_err) * 100))


def _tier_areas(design, parts):
    """Площади ярусов по дискретным точкам (многоугольники)."""
    N = design.n_tiers
    areas = []
    for i in range(N):
        chain = []
        if i < N - 1:
            chain += [parts[f"O{i+1}"], parts[f"W{i+1}"], parts[f"I{i+1}"]]  # I хранится снизу вверх
        else:
            chain += [parts["M"]]
        if i > 0:
            chain.append(parts[f"W{i}"][::-1])
        if i == 0:
            pass  # верх первого яруса — отрезок корпуса B -> A замыкает многоугольник
        poly = np.vstack(chain)
        areas.append(abs(polygon_area(poly)))
    return np.array(areas)

"""Классический (физический) решатель равновесной формы многоярусной пневмооболочки.

Постановка (плоская задача для поперечного сечения, по аналогии с моделью
Ф.С. Пеплина для баллонетного ограждения):

* материал оболочки нерастяжим и невесом, изгибная жёсткость отсутствует;
* на каждую мембрану действует постоянный перепад давлений, поэтому она
  имеет форму дуги окружности, натяжение T постоянно вдоль мембраны,
  кривизна kappa = dp / T (формула Лапласа);
* ярусы (баллоны) замкнуты и соединены шарнирно; в шарнире сходятся три
  мембраны (две стенки и перегородка), и векторная сумма их натяжений равна нулю;
* давление в замкнутом ярусе меняется по адиабате
      p_i = (p_a + p_i0) * (S_i0 / S_i)^n - p_a;
* с внутренней стороны («со стороны подушки») на оболочку действует боковое
  давление p_c, с наружной — атмосферное;
* обжатие моделируется контактом нижнего яруса с жёсткой горизонтальной
  опорой, трение отсутствует; зона контакта — прямолинейный участок длины l,
  к которому дуги примыкают по касательной.

Неизвестные (метод стрельбы с распространением по шарнирам):
    [T_O1, th_O1, T_I1, th_I1, (T_Wi, th_Wi) i=1..N-1, p_1..p_N, (l)]
где l — длина зоны контакта (только в режиме контакта).

Контакт учитывается методом активного множества: сначала решается задача
без опоры; если нижняя точка оболочки оказывается ниже опоры, решается задача
с контактом (неизвестная l, уравнение y_E = y_g) продолжением по положению
опоры от момента касания.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .geometry import ShellDesign, arc_end, arc_points, arc_area_term

T_SCALE = 1000.0   # натяжения в кН/м
P_SCALE = 1000.0   # давления в кПа


def newton_solve(fun, x0, tol=1e-10, maxit=40, h_rel=1e-7):
    """Демпфированный метод Ньютона с центральным конечно-разностным якобианом.

    Шаг dx находится из J dx = -F(x); длина шага подбирается дроблением
    (backtracking) до выполнения условия убывания нормы невязки.
    Возвращает (x, success, число вычислений F, число итераций, норма невязки).
    """
    x = np.array(x0, dtype=float)
    n = x.size
    r = fun(x)
    nfev = 1
    if not np.all(np.isfinite(r)):
        return x, False, nfev, 0, np.inf
    rn = np.linalg.norm(r)
    for it in range(maxit):
        if rn < tol:
            return x, True, nfev, it, rn
        J = np.empty((r.size, n))
        for j in range(n):
            h = h_rel * max(1.0, abs(x[j]))
            e = np.zeros(n)
            e[j] = h
            J[:, j] = (fun(x + e) - fun(x - e)) / (2 * h)
        nfev += 2 * n
        if not np.all(np.isfinite(J)):
            return x, False, nfev, it, rn
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        alpha = 1.0
        while True:
            xn = x + alpha * dx
            rn_new_vec = fun(xn)
            nfev += 1
            rn_new = np.linalg.norm(rn_new_vec) if np.all(np.isfinite(rn_new_vec)) else np.inf
            if rn_new < (1.0 - 1e-4 * alpha) * rn or alpha < 1.0 / 128:
                break
            alpha *= 0.5
        if not np.isfinite(rn_new) or (alpha < 1.0 / 128 and rn_new >= rn):
            return x, False, nfev, it + 1, rn
        x, r, rn = xn, rn_new_vec, rn_new
    return x, rn < tol, nfev, maxit, rn


@dataclass
class Segment:
    name: str
    x0: float
    y0: float
    th0: float
    kappa: float
    s: float
    T: float

    def end(self):
        return arc_end(self.x0, self.y0, self.th0, self.kappa, self.s)

    def area_term(self):
        x1, y1, _ = self.end()
        return arc_area_term(self.x0, self.y0, x1, y1, self.kappa, self.s)

    def points(self, n):
        return arc_points(self.x0, self.y0, self.th0, self.kappa, self.s, n)


@dataclass
class Solution:
    design_name: str
    converged: bool
    valid: bool
    message: str
    p: np.ndarray                 # давления в ярусах, Па
    pc: float
    delta: Optional[float]
    y_ground: Optional[float]
    contact_length: float
    gap: float
    areas: np.ndarray             # площади сечений ярусов, м^2
    segments: Dict[str, Segment]
    x: np.ndarray                 # вектор неизвестных (масштабированный)
    residual_norm: float
    n_fev: int = 0
    time_s: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def tensions(self):
        return {k: v.T for k, v in self.segments.items() if not k.startswith("M_")} | {
            "M": self.segments["M_out"].T}


class ShellSolver:
    """Решатель равновесия замкнутой многоярусной пневмооболочки."""

    def __init__(self, design: ShellDesign):
        self.d = design
        self.N = design.n_tiers
        self._ref_cache: Dict[tuple, Solution] = {}

    # ------------------------------------------------------------------
    # сборка формы по вектору неизвестных
    # ------------------------------------------------------------------
    def _assemble(self, core, p, pc, l):
        """Строит все мембраны по неизвестным «ядра» и возвращает
        (segments, residuals_geom_equil, areas, extra)."""
        d, N = self.d, self.N
        segs: Dict[str, Segment] = {}
        res = []
        xD, yD = 0.0, 0.0
        xC, yC = d.b, 0.0
        T_O, th_O = core[0] * T_SCALE, core[1]
        T_I, th_I = core[2] * T_SCALE, core[3]
        T_M = th_M = None
        for i in range(N - 1):
            # наружная стенка яруса i+1 (обход вниз, ярус слева)
            kO = p[i] / T_O
            sO = Segment(f"O{i+1}", xD, yD, th_O, kO, d.L_out[i], T_O)
            xD1, yD1, thOe = sO.end()
            # внутренняя стенка (обход вниз, ярус справа, подушка слева)
            kI = (pc - p[i]) / T_I
            sI = Segment(f"I{i+1}", xC, yC, th_I, kI, d.L_in[i], T_I)
            xC1, yC1, thIe = sI.end()
            # перегородка между ярусами i+1 и i+2 (обход слева направо)
            T_W, th_W = core[4 + 2 * i] * T_SCALE, core[5 + 2 * i]
            kW = (p[i] - p[i + 1]) / T_W
            sW = Segment(f"W{i+1}", xD1, yD1, th_W, kW, d.L_w[i], T_W)
            xW1, yW1, thWe = sW.end()
            res += [xW1 - xC1, yW1 - yC1]
            segs[sO.name], segs[sI.name], segs[sW.name] = sO, sI, sW
            # равновесие шарнира D_i -> следующая наружная мембрана
            vx = T_O * np.cos(thOe) - T_W * np.cos(th_W)
            vy = T_O * np.sin(thOe) - T_W * np.sin(th_W)
            # угол выбирается на ветви, непрерывной с углом приходящей мембраны
            T_next, th_next = np.hypot(vx, vy), _unwrap_near(np.arctan2(vy, vx), thOe)
            if i < N - 2:
                T_O, th_O = T_next, th_next
                ux = T_I * np.cos(thIe) + T_W * np.cos(thWe)
                uy = T_I * np.sin(thIe) + T_W * np.sin(thWe)
                T_I, th_I = np.hypot(ux, uy), _unwrap_near(np.arctan2(uy, ux), thIe)
            else:
                T_M, th_M = T_next, th_next
                eq_I = (T_I * np.cos(thIe) + T_W * np.cos(thWe),
                        T_I * np.sin(thIe) + T_W * np.sin(thWe))
            xD, yD, xC, yC = xD1, yD1, xC1, yC1

        # нижняя мембрана: наружная дуга -> контактный участок -> внутренняя дуга
        pN = p[N - 1]
        kMo = pN / T_M
        s_o = max(0.0, -th_M / kMo)
        sMo = Segment("M_out", xD, yD, th_M, kMo, s_o, T_M)
        xE, yE, _ = sMo.end()
        sMc = Segment("M_con", xE, yE, 0.0, 0.0, l, T_M)
        xK, yK, _ = sMc.end()
        kMi = (pN - pc) / T_M
        s_i = d.L_m - s_o - l
        sMi = Segment("M_in", xK, yK, 0.0, kMi, s_i, T_M)
        xF, yF, thFe = sMi.end()
        segs["M_out"], segs["M_con"], segs["M_in"] = sMo, sMc, sMi
        res += [xF - xC, yF - yC]
        # равновесие шарнира C_{N-1}: T_I t_I + T_W t_W + T_M t_M = 0
        res += [(eq_I[0] + T_M * np.cos(thFe)) / T_SCALE,
                (eq_I[1] + T_M * np.sin(thFe)) / T_SCALE]

        areas = self._areas(segs)
        extra = dict(yE=yE, xE=xE, s_o=s_o, s_i=s_i, T_M=T_M)
        return segs, np.array(res), areas, extra

    def _areas(self, segs):
        N = self.N
        A = np.zeros(N)
        for i in range(N):
            a = 0.0
            if i < N - 1:
                a += segs[f"O{i+1}"].area_term() + segs[f"W{i+1}"].area_term() \
                    - segs[f"I{i+1}"].area_term()
            else:
                a += segs["M_out"].area_term() + segs["M_con"].area_term() \
                    + segs["M_in"].area_term()
            if i > 0:
                a -= segs[f"W{i}"].area_term()
            A[i] = a
        return A

    # ------------------------------------------------------------------
    # системы уравнений
    # ------------------------------------------------------------------
    def _n_core(self):
        return 4 + 2 * (self.N - 1)

    def _residual_reference(self, x, p0):
        segs, r, _, _ = self._assemble(x, p0, 0.0, 0.0)
        return r

    def _residual_loaded(self, x, p0, S0, pc, y_ground):
        """Невязки нагруженного состояния. y_ground=None — режим без контакта,
        иначе — режим контакта с неизвестной длиной зоны контакта l = x[-1]."""
        nc, N = self._n_core(), self.N
        core = x[:nc]
        p = x[nc:nc + N] * P_SCALE
        l = x[nc + N] if y_ground is not None else 0.0
        segs, r, S, ex = self._assemble(core, p, pc, l)
        pa, n = self.d.p_atm, self.d.n_poly
        with np.errstate(invalid="ignore", divide="ignore"):
            p_ad = (pa + p0) * np.power(np.maximum(S0 / np.maximum(S, 1e-9), 1e-9), n) - pa
        r_ad = (p - p_ad) / P_SCALE
        out = [r, r_ad]
        if y_ground is not None:
            out.append([ex["yE"] - y_ground])
        return np.concatenate(out)

    # ------------------------------------------------------------------
    def initial_guess(self):
        """Начальное приближение для отсчётного состояния (подобрано по эскизу)."""
        N = self.N
        x = [1.7, -2.3, 1.7, -0.84]
        for i in range(N - 1):
            x += [1.8, 0.0]
        return np.array(x, dtype=float)

    P_BASE = 5000.0

    def _continuation(self, fun_of_lam, x0, h0=0.5, hmin=1.0 / 2048, maxit=12):
        """Продолжение по параметру lam: 0 -> 1 с адаптивным шагом
        (схема «предиктор — корректор»).

        fun_of_lam(lam) возвращает функцию невязки F(x). Предиктор —
        секущая экстраполяция по двум последним решениям, корректор —
        демпфированный метод Ньютона. При неудаче шаг дробится.
        """
        lam, h, x = 0.0, h0, np.array(x0, dtype=float)
        lam_prev, x_prev = None, None
        nfev, nsteps = 0, 0
        while lam < 1.0 - 1e-12:
            lam_try = min(1.0, lam + h)
            if x_prev is not None:
                xg = x + (x - x_prev) * (lam_try - lam) / (lam - lam_prev)
            else:
                xg = x
            xn, ok, nf, nit, rn = newton_solve(fun_of_lam(lam_try), xg, maxit=maxit)
            nfev += nf
            nsteps += 1
            if ok:
                lam_prev, x_prev = lam, x
                x, lam = xn, lam_try
                h = min(h * (2.0 if nit <= 4 else 1.2), 0.5)
            else:
                h *= 0.5
                if h < hmin:
                    return x, lam, nfev, nsteps, False
        return x, lam, nfev, nsteps, True

    def _base_reference(self):
        """Отсчётная форма при равных давлениях зарядки (не зависит от уровня
        давления: натяжения пропорциональны давлению, геометрия одна и та же)."""
        if "base" not in self._ref_cache:
            pb = np.full(self.N, self.P_BASE)
            xb, ok, _, _, rn = newton_solve(lambda xx: self._residual_reference(xx, pb),
                                            self.initial_guess(), maxit=100)
            if not ok:
                raise RuntimeError("base reference state did not converge")
            self._ref_cache["base"] = xb
        return self._ref_cache["base"]

    def solve_reference(self, p0, x0=None) -> Solution:
        """Свободная форма при давлениях зарядки p0 (без подушки и опоры).

        Решение строится продолжением по давлениям от состояния с равными
        давлениями, для которого известно хорошее начальное приближение.
        """
        p0 = np.asarray(p0, dtype=float)
        key = tuple(np.round(p0, 6))
        if key in self._ref_cache and x0 is None:
            return self._ref_cache[key]
        t0 = time.perf_counter()
        xb = self._base_reference().copy()
        pm = float(np.mean(p0))
        scale = pm / self.P_BASE
        nc = self._n_core()
        # натяжения (чётные индексы ядра, кроме углов) масштабируются уровнем давления
        tens_idx = [0, 2] + [4 + 2 * i for i in range(self.N - 1)]
        xb[tens_idx] *= scale
        if x0 is not None:
            x, ok, nfev, _, _ = newton_solve(lambda xx: self._residual_reference(xx, p0), x0)
        else:
            pbar = np.full(self.N, pm)
            x, lam, nfev, nst, ok = self._continuation(
                lambda lam: (lambda xx: self._residual_reference(xx, pbar + lam * (p0 - pbar))), xb)
        r = self._residual_reference(x, p0)
        rn = float(np.linalg.norm(r))
        segs, _, S, ex = self._assemble(x, p0, 0.0, 0.0)
        valid = self._check_valid(segs, ex)
        out = Solution(self.d.name, bool(ok and rn < 1e-9), valid, "ok" if ok else "failed",
                       p0.copy(), 0.0, None, None, 0.0, np.inf, S, segs, x, rn,
                       nfev, time.perf_counter() - t0, dict(yE=ex["yE"], xE=ex["xE"]))
        if x0 is None:
            if len(self._ref_cache) > 20000:
                self._ref_cache = {"base": self._ref_cache["base"]}
            self._ref_cache[key] = out
        return out

    def solve(self, p0, pc, delta=None, x_init=None) -> Solution:
        """Равновесная форма под действием бокового давления pc и обжатия delta.

        delta — подъём опорной поверхности относительно нижней точки
        оболочки в отсчётном состоянии (delta > 0 — обжатие, delta <= 0 —
        зазор). delta=None — опора отсутствует.
        x_init — внешнее начальное приближение (например, полученное из
        прогноза ML-модели): вектор неизвестных в режиме, соответствующем
        длине его последней компоненты; при неудаче используется продолжение.
        """
        t0 = time.perf_counter()
        p0 = np.asarray(p0, dtype=float)
        ref = self.solve_reference(p0)
        if not ref.converged:
            return self._fail(ref, p0, pc, delta, "reference failed", t0)
        S0 = ref.areas
        y_ref_low = ref.info["yE"]
        y_ground = None if delta is None else y_ref_low + delta
        nc, N = self._n_core(), self.N
        nfev, nsteps = 0, 0
        mode, x, ok = None, None, False

        # 1) попытка «тёплого старта» по внешнему начальному приближению
        if x_init is not None:
            x_init = np.asarray(x_init, dtype=float)
            contact = (y_ground is not None) and x_init.size == nc + N + 1 and x_init[-1] > 1e-4
            yg = y_ground if contact else None
            xi = x_init if contact else x_init[:nc + N]
            xn, ok1, nf, nit, _ = newton_solve(
                lambda xx: self._residual_loaded(xx, p0, S0, pc, yg), xi, maxit=25)
            nfev += nf
            nsteps += 1
            if ok1:
                l_ok = (not contact) or xn[-1] >= 0.0
                if not contact and y_ground is not None:
                    segs, _, _, ex = self._assemble(xn[:nc], xn[nc:nc + N] * P_SCALE, pc, 0.0)
                    l_ok = ex["yE"] >= y_ground - 1e-10
                if l_ok:
                    x, ok, mode = xn, True, ("contact" if contact else "free")

        if not ok:
            # 2) деформация боковым давлением без опоры (продолжение по pc)
            xs = np.concatenate([ref.x[:nc], p0 / P_SCALE])
            x, lam, nf, nst, ok = self._continuation(
                lambda lam: (lambda xx: self._residual_loaded(xx, p0, S0, lam * pc, None)), xs)
            nfev += nf
            nsteps += nst
            mode = "free"
            if ok and y_ground is not None:
                segs, _, _, ex = self._assemble(x[:nc], x[nc:nc + N] * P_SCALE, pc, 0.0)
                yE_free = ex["yE"]
                if yE_free < y_ground:
                    # 3) обжатие: опора поднимается от точки касания до y_ground
                    xs = np.concatenate([x, [0.0]])
                    x, lam, nf, nst, ok = self._continuation(
                        lambda mu: (lambda xx: self._residual_loaded(
                            xx, p0, S0, pc, yE_free + mu * (y_ground - yE_free))), xs)
                    nfev += nf
                    nsteps += nst
                    mode = "contact"

        yg_mode = y_ground if mode == "contact" else None
        r = self._residual_loaded(x, p0, S0, pc, yg_mode)
        rn = float(np.linalg.norm(r))
        core = x[:nc]
        p = x[nc:nc + N] * P_SCALE
        l = float(x[nc + N]) if mode == "contact" else 0.0
        segs, _, S, ex = self._assemble(core, p, pc, l)
        conv = bool(ok and rn < 1e-9 and l >= -1e-9)
        valid = conv and self._check_valid(segs, ex)
        gap = (ex["yE"] - y_ground) if y_ground is not None else np.inf
        return Solution(self.d.name, conv, valid, "ok" if conv else "continuation failed",
                        p, pc, delta, y_ground, max(l, 0.0), gap, S, segs, x, rn, nfev,
                        time.perf_counter() - t0,
                        dict(yE=ex["yE"], xE=ex["xE"], S0=S0, nsteps=nsteps,
                             y_ref_low=y_ref_low, mode=mode))

    def _fail(self, ref, p0, pc, delta, msg, t0):
        return Solution(self.d.name, False, False, msg, p0, pc, delta, None, 0.0, np.inf,
                        ref.areas, ref.segments, ref.x, np.inf, 0, time.perf_counter() - t0)

    def _check_valid(self, segs, ex):
        """Физическая допустимость: положительные натяжения и длины,
        отсутствие самопересечений контура."""
        for s in segs.values():
            if not np.isfinite(s.T) or s.T <= 0:
                return False
            if s.s < -1e-12:
                return False
        if ex["s_i"] <= 0:
            return False
        pts = self.contour(segs, n_per_m=60)
        # оболочка не должна проникать в корпус: точка A — нижняя кромка борта,
        # корпус занимает четверть плоскости x >= 0, y >= 0
        inside = (pts[:, 0] > 1e-6) & (pts[:, 1] > 1e-6)
        if np.any(inside):
            return False
        return not _polyline_self_intersects(pts)

    # ------------------------------------------------------------------
    # дискретизация формы
    # ------------------------------------------------------------------
    def membrane_order(self):
        """Порядок мембран во внешнем контуре A -> ... -> B."""
        N = self.N
        outer = [f"O{i+1}" for i in range(N - 1)]
        inner = [f"I{i+1}" for i in reversed(range(N - 1))]
        return outer, inner

    def contour(self, segs, n_per_m=40):
        """Плотная ломаная внешнего контура (для графиков и проверок)."""
        outer, inner = self.membrane_order()
        pts = []
        for nm in outer:
            s = segs[nm]
            pts.append(s.points(max(3, int(s.s * n_per_m)))[:-1])
        for nm in ("M_out", "M_con", "M_in"):
            s = segs[nm]
            if s.s > 1e-9:
                pts.append(s.points(max(3, int(s.s * n_per_m)))[:-1])
        for nm in inner:
            s = segs[nm]
            pts.append(s.points(max(3, int(s.s * n_per_m)))[::-1][:-1])
        pts.append(np.array([[self.d.b, 0.0]]))
        return np.vstack(pts)

    def material_points(self, segs):
        """Координаты фиксированного набора материальных точек (выход ML-модели).

        Поскольку материал нерастяжим, длина дуги от начала мембраны является
        материальной (лагранжевой) координатой. Для каждой мембраны берётся
        фиксированное число равноотстоящих по длине точек.
        """
        d, N = self.d, self.N
        out = []
        for i in range(N - 1):
            out.append(segs[f"O{i+1}"].points(d.n_pts(d.L_out[i])))
        out.append(self._bottom_points(segs, d.n_pts(d.L_m)))
        for i in reversed(range(N - 1)):
            out.append(segs[f"I{i+1}"].points(d.n_pts(d.L_in[i]))[::-1])
        for i in range(N - 1):
            out.append(segs[f"W{i+1}"].points(d.n_pts(d.L_w[i])))
        return np.vstack(out)

    def _bottom_points(self, segs, n):
        so, sc, si = segs["M_out"], segs["M_con"], segs["M_in"]
        t = np.linspace(0.0, so.s + sc.s + si.s, n)
        P = np.empty((n, 2))
        for k, tk in enumerate(t):
            if tk <= so.s:
                x, y, _ = arc_end(so.x0, so.y0, so.th0, so.kappa, tk)
            elif tk <= so.s + sc.s:
                x, y = sc.x0 + (tk - so.s), sc.y0
            else:
                x, y, _ = arc_end(si.x0, si.y0, si.th0, si.kappa, tk - so.s - sc.s)
            P[k] = (x, y)
        return P

    def output_layout(self):
        """Имена и число точек каждого участка выходного вектора."""
        d, N = self.d, self.N
        lay = []
        for i in range(N - 1):
            lay.append((f"O{i+1}", d.n_pts(d.L_out[i])))
        lay.append(("M", d.n_pts(d.L_m)))
        for i in reversed(range(N - 1)):
            lay.append((f"I{i+1}", d.n_pts(d.L_in[i])))
        for i in range(N - 1):
            lay.append((f"W{i+1}", d.n_pts(d.L_w[i])))
        return lay


def _unwrap_near(th, ref):
    """Выбор представителя угла th (mod 2pi), ближайшего к опорному углу ref."""
    return ref + (th - ref + np.pi) % (2.0 * np.pi) - np.pi


def _polyline_self_intersects(P, tol=1e-9):
    """Проверка самопересечения ломаной (O(n^2), n ~ 200)."""
    n = len(P) - 1
    a = P[:-1]
    b = P[1:]
    for i in range(n):
        p1, p2 = a[i], b[i]
        # проверяем только несмежные отрезки
        j0 = i + 2
        if j0 >= n:
            continue
        q1, q2 = a[j0:], b[j0:]
        d1 = _cross(q2 - q1, p1 - q1)
        d2 = _cross(q2 - q1, p2 - q1)
        d3 = _cross(p2 - p1, q1 - p1)
        d4 = _cross(p2 - p1, q2 - p1)
        hit = (d1 * d2 < -tol) & (d3 * d4 < -tol)
        if np.any(hit):
            return True
    return False


def _cross(u, v):
    u = np.atleast_2d(u)
    v = np.atleast_2d(v)
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

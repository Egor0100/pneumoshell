"""Построение рисунков для отчёта и расчёт апробационных сценариев.

Запуск после train_evaluate.py:  python scripts/make_figures.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from pneumoshell.geometry import DESIGNS  # noqa: E402
from pneumoshell.ml import ShapeSurrogate  # noqa: E402
from pneumoshell.solver import ShellSolver  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", grid="#d9d8d3", gray="#9a9993")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
TITLE = {"two_tier": "двухъярусная", "three_tier": "трёхъярусная"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 220, "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})


def plain_log(ax, axis, ticks):
    """Логарифмическая ось с обычными числовыми подписями (без 10^n)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fmt = FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ","))
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_major_formatter(fmt)
    a.set_minor_locator(NullLocator())


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("saved", name)


def split_parts(design, pts):
    lay = ShellSolver(design).output_layout()
    parts, k = {}, 0
    for nm, n in lay:
        parts[nm] = pts[k:k + n]
        k += n
    return parts


def draw_shape(ax, design, pts, color, ls="-", lw=1.8, label=None, wall_color=None):
    parts = split_parts(design, pts.reshape(-1, 2))
    outer = [parts[f"O{i+1}"] for i in range(design.n_tiers - 1)] + [parts["M"]] + \
            [parts[f"I{i+1}"] for i in reversed(range(design.n_tiers - 1))]
    P = np.vstack(outer)
    ax.plot(P[:, 0], P[:, 1], ls, color=color, lw=lw, label=label)
    for i in range(design.n_tiers - 1):
        W = parts[f"W{i+1}"]
        ax.plot(W[:, 0], W[:, 1], ls, color=wall_color or color, lw=lw * 0.8)


def hull(ax, x0=-0.9, x1=0.9):
    ax.fill_between([0, x1], [0, 0], [0.08, 0.08], color="#cfcac0", lw=0, zorder=0)
    ax.plot([0, x1], [0, 0], color=C["ink2"], lw=1.2)
    ax.plot([0, 0], [0, 0.08], color=C["ink2"], lw=1.2)


# ----------------------------------------------------------------------
def fig_scheme():
    d = DESIGNS["two_tier"]()
    S = ShellSolver(d)
    p0 = np.array([6000.0, 6000.0])
    ref = S.solve_reference(p0)
    sol = S.solve(p0, 1200.0, 0.12)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6))
    for ax, s, ttl in [(axs[0], ref, "а) отсчётное состояние"), (axs[1], sol, "б) обжатие и боковое давление")]:
        hull(ax)
        segs = s.segments
        c = S.contour(segs, 200)
        ax.plot(c[:, 0], c[:, 1], color=C["blue"], lw=2)
        w = segs["W1"].points(40)
        ax.plot(w[:, 0], w[:, 1], color=C["orange"], lw=2)
        D = (segs["W1"].x0, segs["W1"].y0)
        xC, yC, _ = segs["W1"].end()
        for (x, y, t, dx, dy) in [(0, 0, "A", -0.07, 0.02), (d.b, 0, "B", 0.03, 0.02),
                                  (D[0], D[1], "D", -0.08, 0.0), (xC, yC, "C", 0.03, 0.0)]:
            ax.plot(x, y, "o", color=C["ink"], ms=4)
            ax.text(x + dx, y + dy, t, fontsize=11, style="italic")
        E = (s.info["xE"], s.info["yE"])
        ax.plot(*E, "o", color=C["ink"], ms=4)
        ax.text(E[0] - 0.07, E[1] + 0.03, "E", fontsize=11, style="italic")
        # подписи давлений — у центров ярусов
        wm = segs["W1"].points(3)[1]
        ax.text(wm[0] - 0.03, wm[1] + 0.13, "$p_1$", fontsize=13)
        ax.text(wm[0] - 0.03, wm[1] - 0.30, "$p_2$", fontsize=13)
        ax.text(-0.55, -0.35, "$p_a$", fontsize=13, color=C["ink2"])
        ax.set_aspect("equal")
        ax.set_xlim(-0.65, 0.85)
        ax.set_ylim(-1.12, 0.12)
        ax.set_title(ttl)
        ax.set_xlabel("x, м")
    axs[0].set_ylabel("y, м")
    ref_c = S.contour(ref.segments, 200)
    axs[1].plot(ref_c[:, 0], ref_c[:, 1], "--", color=C["gray"], lw=1)
    axs[1].axhline(sol.y_ground, color=C["ink"], lw=1.2)
    axs[1].fill_between([-0.65, 0.85], [sol.y_ground] * 2, [sol.y_ground - 0.05] * 2, color="#e6e2da", lw=0)
    axs[1].text(0.55, -0.55, "$p_c$", fontsize=12)
    for yy in (-0.3, -0.5, -0.7):
        axs[1].add_patch(FancyArrowPatch((0.78, yy), (0.62, yy), arrowstyle="-|>", mutation_scale=10,
                                         color=C["ink2"], lw=1))
    xs = sol.segments["M_con"]
    axs[1].plot([xs.x0, xs.x0 + xs.s], [xs.y0, xs.y0], color=C["red"], lw=3.5)
    axs[1].text(xs.x0 + xs.s / 2 - 0.01, xs.y0 + 0.03, "$l$", fontsize=13, color=C["red"])
    axs[1].annotate("", xy=(0.72, sol.y_ground), xytext=(0.72, ref.info["yE"]),
                    arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1))
    axs[1].text(0.74, (sol.y_ground + ref.info["yE"]) / 2 - 0.02, "$\\delta$", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_scheme.png")


def fig_examples():
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.9))
    rng = np.random.default_rng(3)
    for r, name in enumerate(["two_tier", "three_tier"]):
        d = DESIGNS[name]()
        dat = np.load(os.path.join(ROOT, "data", f"dataset_{name}.npz"))
        S = ShellSolver(d)
        ref = S.solve_reference(np.full(d.n_tiers, 5000.0))
        idx = rng.choice(len(dat["X"]), 4, replace=False)
        for c, i in enumerate(idx):
            ax = axs[r, c]
            hull(ax)
            rc = S.contour(ref.segments, 150)
            ax.plot(rc[:, 0], rc[:, 1], "--", color=C["gray"], lw=0.8)
            draw_shape(ax, d, dat["Y"][i], C["blue"], wall_color=C["orange"], lw=1.6)
            x = dat["X"][i]
            N = d.n_tiers
            yl = ref.info["yE"] + x[N + 1]
            ax.axhline(yl, color=C["ink"], lw=1)
            ax.set_aspect("equal")
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-1.35, 0.1)
            ax.set_xticks([-0.5, 0, 0.5])
            ax.tick_params(labelsize=8)
            p = "/".join(f"{v/1000:.1f}" for v in x[:N])
            ax.set_title(f"$p_0$={p} кПа\n$p_c$={x[N]/1000:.2f} кПа, δ={x[N+1]*100:.0f} см", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_examples.png")


def fig_pca(M):
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    for j, name in enumerate(["two_tier", "three_tier"]):
        r = M[name]["pca"]
        ax.semilogy([q["k"] for q in r], [q["rmse_mm"] for q in r], "o-", color=SERIES[j], ms=5,
                    label=f"{TITLE[name]} оболочка")
    ax.axvline(24, color=C["ink2"], lw=1, ls=":")
    ax.text(24.5, ax.get_ylim()[1] * 0.3, "k = 24", color=C["ink2"])
    plain_log(ax, "y", [0.001, 0.01, 0.1, 1, 10])
    ax.set_xlabel("число главных компонент k")
    ax.set_ylabel("СКО реконструкции, мм")
    ax.legend()
    save(fig, "fig_pca.png")


def fig_models(M):
    order = ["linear", "poly3", "knn", "rf", "gbm", "gpr", "mlp"]
    labels = ["Линейная", "Полином. 3-й ст.", "kNN", "Случ. лес", "Град. бустинг", "Гаусс. процесс", "MLP"]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    y = np.arange(len(order))
    h = 0.38
    for j, name in enumerate(["two_tier", "three_tier"]):
        v = [M[name]["models"][m]["rmse_mm"] for m in order]
        ax.barh(y + (j - 0.5) * h, v, height=h - 0.04, color=SERIES[j], label=f"{TITLE[name]}")
        for yy, vv in zip(y + (j - 0.5) * h, v):
            ax.text(vv * 1.08, yy, f"{vv:.2f}", va="center", fontsize=8, color=C["ink2"])
    ax.set_xscale("log")
    plain_log(ax, "x", [0.3, 1, 3, 10, 30])
    ax.set_xlim(0.2, 40)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("СКО координат точек на тестовой выборке, мм (лог. шкала)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig_models.png")


def fig_pred_vs_true(best):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.6))
    for r, name in enumerate(["two_tier", "three_tier"]):
        d = DESIGNS[name]()
        T = np.load(os.path.join(RES, f"test_pred_{name}.npz"))
        Y, Yp = T["Y"], T[f"Y_{best}"]
        E = np.linalg.norm((Yp - Y).reshape(len(Y), -1, 2), axis=2).max(axis=1)
        order = np.argsort(E)
        picks = [order[len(order) // 2], order[int(0.95 * len(order))], order[-1]]
        tt = ["медианный случай", "95-й процентиль", "наихудший случай"]
        S = ShellSolver(d)
        ref = S.solve_reference(np.full(d.n_tiers, 5000.0))
        N = d.n_tiers
        for c, i in enumerate(picks):
            ax = axs[r, c]
            hull(ax)
            draw_shape(ax, d, Y[i], C["blue"], lw=2.2, label="решатель")
            draw_shape(ax, d, Yp[i], C["orange"], ls="--", lw=1.5, label="ML (MLP)")
            ax.axhline(ref.info["yE"] + T["X"][i, N + 1], color=C["ink"], lw=1)
            ax.set_aspect("equal")
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-1.35, 0.1)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{tt[c]}\nмакс. ошибка {E[i]*1000:.1f} мм", fontsize=9)
        if r == 0:
            axs[0, 0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_pred_vs_true.png")


def fig_learning(M):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    names = dict(poly3="Полином. 3-й ст.", gbm="Град. бустинг", gpr="Гаусс. процесс", mlp="MLP")
    for a, name in zip(axs, ["two_tier", "three_tier"]):
        for j, mn in enumerate(["poly3", "gbm", "gpr", "mlp"]):
            r = M[name]["learning_curve"][mn]
            a.loglog([q["n"] for q in r], [q["rmse_mm"] for q in r], "o-", color=SERIES[j], ms=5,
                     label=names[mn])
        a.set_title(f"{TITLE[name]} оболочка")
        a.set_xlabel("объём обучающей выборки")
        plain_log(a, "x", [150, 300, 600, 1200, 2800])
        plain_log(a, "y", [0.3, 0.5, 1, 2, 5, 10, 20])
    axs[0].set_ylabel("СКО на тесте, мм")
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_learning.png")


def fig_error_bins(best):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for j, name in enumerate(["two_tier", "three_tier"]):
        d = DESIGNS[name]()
        N = d.n_tiers
        T = np.load(os.path.join(RES, f"test_pred_{name}.npz"))
        E = np.sqrt(np.mean(np.sum(((T[f"Y_{best}"] - T["Y"]).reshape(len(T["Y"]), -1, 2)) ** 2, axis=2),
                            axis=1)) * 1000
        for a, col, lab, sc in [(axs[0], N + 1, "обжатие δ, см", 100), (axs[1], N, "боковое давление $p_c$, кПа", 1e-3)]:
            x = T["X"][:, col] * sc
            bins = np.linspace(x.min(), x.max(), 9)
            cen = 0.5 * (bins[1:] + bins[:-1])
            mean = [E[(x >= bins[k]) & (x < bins[k + 1] + 1e-12)].mean() for k in range(8)]
            a.plot(cen, mean, "o-", color=SERIES[j], ms=5, label=f"{TITLE[name]}")
            a.set_xlabel(lab)
    axs[0].set_ylabel("СКО формы (среднее по интервалу), мм")
    axs[0].axvline(0, color=C["ink2"], lw=1, ls=":")
    axs[0].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_error_bins.png")


def fig_error_along(best):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for a, name in zip(axs, ["two_tier", "three_tier"]):
        d = DESIGNS[name]()
        T = np.load(os.path.join(RES, f"test_pred_{name}.npz"))
        E = np.linalg.norm((T[f"Y_{best}"] - T["Y"]).reshape(len(T["Y"]), -1, 2), axis=2) * 1000
        lay = ShellSolver(d).output_layout()
        k = 0
        for j, (nm, n) in enumerate(lay):
            xs = np.arange(k, k + n)
            col = C["orange"] if nm.startswith("W") else (C["aqua"] if nm == "M" else C["blue"])
            a.plot(xs, E[:, k:k + n].mean(axis=0), color=col, lw=1.8)
            a.text(k + n / 2, 0.02, nm if nm != "M" else "M", ha="center", fontsize=8, color=C["ink2"],
                   transform=a.get_xaxis_transform())
            k += n
        a.set_title(f"{TITLE[name]} оболочка")
        a.set_xlabel("номер материальной точки")
    axs[0].set_ylabel("средняя ошибка точки, мм")
    fig.tight_layout()
    save(fig, "fig_error_along.png")


def sweeps(best):
    """Апробация: непрерывные сценарии нагружения, не входившие в выборку."""
    out = {}
    for name in ["two_tier", "three_tier"]:
        d = DESIGNS[name]()
        S = ShellSolver(d)
        sur = ShapeSurrogate.load(os.path.join(ROOT, "models", f"{name}_{best}.pkl"))
        N = d.n_tiers
        p0 = np.full(N, 6500.0)
        # сценарий 1: обжатие при pc = 1 кПа
        deltas = np.linspace(-0.05, 0.25, 31)
        r1 = dict(delta=deltas.tolist(), l_true=[], l_pred=[], p_true=[], p_pred=[], ylow_true=[],
                  ylow_pred=[], rmse=[], shapes={})
        for dl in deltas:
            sol = S.solve(p0, 1000.0, dl)
            Yt = S.material_points(sol.segments)
            Yp, Pp, Lp = sur.predict(np.concatenate([p0, [1000.0, dl]])[None])
            Yp = Yp[0].reshape(-1, 2)
            r1["l_true"].append(sol.contact_length)
            r1["l_pred"].append(float(Lp[0]))
            r1["p_true"].append(sol.p.tolist())
            r1["p_pred"].append(Pp[0].tolist())
            r1["ylow_true"].append(float(Yt[:, 1].min()))
            r1["ylow_pred"].append(float(Yp[:, 1].min()))
            r1["rmse"].append(float(np.sqrt(np.mean(np.sum((Yp - Yt) ** 2, axis=1)))))
            if any(abs(dl - v) < 1e-9 for v in (0.0, 0.1, 0.2)):
                r1["shapes"][f"{dl:.2f}"] = dict(true=Yt.tolist(), pred=Yp.tolist(), yg=sol.y_ground)
        # сценарий 2: боковое давление при delta = 0.08 м
        pcs = np.linspace(500, 2000, 16)
        r2 = dict(pc=pcs.tolist(), xlow_true=[], xlow_pred=[], rmse=[], l_true=[], l_pred=[])
        for pc in pcs:
            sol = S.solve(p0, pc, 0.08)
            Yt = S.material_points(sol.segments)
            Yp, Pp, Lp = sur.predict(np.concatenate([p0, [pc, 0.08]])[None])
            Yp = Yp[0].reshape(-1, 2)
            # горизонтальное смещение центра нижнего яруса (средняя точка мембраны M)
            parts_t = split_parts(d, Yt)
            parts_p = split_parts(d, Yp)
            r2["xlow_true"].append(float(parts_t["M"][:, 0].mean()))
            r2["xlow_pred"].append(float(parts_p["M"][:, 0].mean()))
            r2["rmse"].append(float(np.sqrt(np.mean(np.sum((Yp - Yt) ** 2, axis=1)))))
            r2["l_true"].append(sol.contact_length)
            r2["l_pred"].append(float(Lp[0]))
        out[name] = dict(delta_sweep=r1, pc_sweep=r2)
    with open(os.path.join(RES, "sweeps.json"), "w") as f:
        json.dump(out, f)
    return out


def fig_sweeps(SW):
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.4))
    for r, name in enumerate(["two_tier", "three_tier"]):
        s1, s2 = SW[name]["delta_sweep"], SW[name]["pc_sweep"]
        dl = np.array(s1["delta"]) * 100
        a = axs[r, 0]
        a.plot(dl, np.array(s1["l_true"]) * 100, color=C["blue"], label="решатель")
        a.plot(dl, np.array(s1["l_pred"]) * 100, "o", color=C["orange"], ms=3.5, label="MLP")
        a.set_xlabel("δ, см")
        a.set_ylabel("длина контакта l, см")
        a = axs[r, 1]
        pt = np.array(s1["p_true"]) / 1000
        pp = np.array(s1["p_pred"]) / 1000
        for i in range(pt.shape[1]):
            a.plot(dl, pt[:, i], color=SERIES[i], lw=1.8, label=f"$p_{i+1}$")
            a.plot(dl, pp[:, i], "o", color=SERIES[i], ms=3)
        a.set_xlabel("δ, см")
        a.set_ylabel("давление в ярусе, кПа")
        a.legend(fontsize=7.5, ncol=3, loc="upper left")
        a = axs[r, 2]
        pc = np.array(s2["pc"]) / 1000
        a.plot(pc, np.array(s2["xlow_true"]) * 100, color=C["blue"], label="решатель")
        a.plot(pc, np.array(s2["xlow_pred"]) * 100, "o", color=C["orange"], ms=3.5, label="MLP")
        a.set_xlabel("$p_c$, кПа")
        a.set_ylabel("$x$ центра нижнего яруса, см")
        axs[r, 0].set_title(f"{TITLE[name]}", fontsize=10, loc="left")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_sweeps.png")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4))
    for a, name in zip(axs, ["two_tier", "three_tier"]):
        d = DESIGNS[name]()
        hull(a)
        for j, (k, v) in enumerate(SW[name]["delta_sweep"]["shapes"].items()):
            draw_shape(a, d, np.array(v["true"]), SERIES[j], lw=2.0, label=f"δ = {float(k)*100:.0f} см")
            draw_shape(a, d, np.array(v["pred"]), C["ink"], ls=":", lw=1.2)
            a.axhline(v["yg"], color=SERIES[j], lw=0.9, ls="-")
        a.set_aspect("equal")
        a.set_xlim(-0.7, 0.75)
        a.set_ylim(-1.35, 0.1)
        a.set_title(f"{TITLE[name]} оболочка, $p_c$ = 1 кПа", fontsize=10)
        a.set_xlabel("x, м")
    axs[0].set_ylabel("y, м")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_sweep_shapes.png")


def fig_hybrid(M):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for a, name in zip(axs, ["two_tier", "three_tier"]):
        h = M[name]["hybrid"]
        tc = np.array(h["t_cold"]) * 1000
        tw = np.array(h["t_warm"]) * 1000
        bins = np.logspace(np.log10(min(tc.min(), tw.min()) * 0.8), np.log10(max(tc.max(), tw.max()) * 1.2), 30)
        a.hist(tc, bins=bins, color=C["blue"], alpha=0.85, label="продолжение по параметру")
        a.hist(tw, bins=bins, color=C["orange"], alpha=0.85, label="старт от прогноза MLP")
        a.set_xscale("log")
        plain_log(a, "x", [20, 50, 100, 200, 500])
        a.set_xlabel("время решения, мс")
        a.set_title(f"{TITLE[name]} оболочка")
    axs[0].set_ylabel("число расчётных случаев")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig_hybrid.png")


def fig_bifurcation():
    """Неединственность решения при малом боковом давлении и сильном обжатии."""
    d = DESIGNS["two_tier"]()
    S = ShellSolver(d)
    lay = S.output_layout()
    k0 = lay[0][1]
    nM = lay[1][1]
    p0 = np.array([6000.0, 6000.0])
    pcs = np.concatenate([np.linspace(0, 200, 41), np.linspace(210, 800, 30)])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    out = {}
    for j, dl in enumerate([0.05, 0.10, 0.15, 0.20]):
        xs = []
        for pc in pcs:
            sol = S.solve(p0, pc, dl)
            P = S.material_points(sol.segments)
            xs.append(P[k0:k0 + nM, 0].mean() * 100 if sol.valid else np.nan)
        axs[0].plot(pcs / 1000, xs, "-", color=SERIES[j], lw=1.8, label=f"δ = {dl*100:.0f} см")
        out[f"{dl:.2f}"] = xs
    axs[0].axvspan(0, 0.5, color="#f1efe9", zorder=0)
    axs[0].text(0.02, 0.04, "исключено\nиз области", transform=axs[0].transAxes, fontsize=8, color=C["ink2"])
    axs[0].set_xlabel("$p_c$, кПа")
    axs[0].set_ylabel("$x$ центра нижнего яруса, см")
    axs[0].legend(fontsize=8)
    # две формы по разные стороны скачка
    hull(axs[1])
    for pc, col, lab in [(20.0, C["violet"], "$p_c$ = 0,02 кПа"), (60.0, C["red"], "$p_c$ = 0,06 кПа")]:
        sol = S.solve(p0, pc, 0.20)
        draw_shape(axs[1], d, S.material_points(sol.segments), col, lw=1.8, label=lab)
    ref = S.solve_reference(p0)
    axs[1].axhline(ref.info["yE"] + 0.20, color=C["ink"], lw=1)
    axs[1].set_aspect("equal")
    axs[1].set_xlim(-0.6, 0.8)
    axs[1].set_ylim(-0.95, 0.1)
    axs[1].set_title("δ = 20 см", fontsize=10)
    axs[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_bifurcation.png")
    json.dump(dict(pc=pcs.tolist(), x=out), open(os.path.join(RES, "bifurcation.json"), "w"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.05, 2.6, "Параметры\n$p_{0i}$, $p_c$, δ\n(LHS-выборка)"),
        (2.65, 2.6, "Классический\nрешатель (дуги,\nшарниры, адиабата,\nконтакт)"),
        (5.25, 2.6, "Датасет:\nкоординаты точек,\n$p_i$, $l$"),
        (7.85, 2.6, "Кодирование цели:\nPCA + стандарти-\nзация"),
        (7.85, 0.3, "Регрессор\n(MLP, ГП,\nбустинг, ...)"),
        (5.25, 0.3, "Метрики:\nСКО, R², площадь,\nнерастяжимость"),
        (2.65, 0.3, "Прогноз формы\n< 1 мс; тёплый\nстарт решателя"),
        (0.05, 0.3, "Потребители:\nпроектирование,\nмодель динамики"),
    ]
    for (x, y, t) in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.3, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc="#eef4fc", ec=C["blue"], lw=1.2))
        ax.text(x + 1.1, y + 0.65, t, ha="center", va="center", fontsize=8.5)
    arrows = [((2.3, 3.25), (2.6, 3.25)), ((4.9, 3.25), (5.2, 3.25)), ((7.5, 3.25), (7.8, 3.25)),
              ((8.95, 2.55), (8.95, 1.65)), ((7.8, 0.95), (7.5, 0.95)), ((5.2, 0.95), (4.9, 0.95)),
              ((2.6, 0.95), (2.3, 0.95))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=C["ink2"], lw=1.2))
    ax.add_patch(FancyArrowPatch((3.75, 1.65), (3.75, 2.55), arrowstyle="-|>", mutation_scale=12,
                                 color=C["orange"], lw=1.2, ls="--"))
    ax.text(3.85, 1.95, "начальное приближение", fontsize=7.5, color=C["orange"])
    save(fig, "fig_pipeline.png")


def main():
    M = {n: json.load(open(os.path.join(RES, f"metrics_{n}.json"))) for n in ["two_tier", "three_tier"]}
    best = "mlp"
    fig_scheme()
    fig_examples()
    fig_pca(M)
    fig_models(M)
    fig_pred_vs_true(best)
    fig_learning(M)
    fig_error_bins(best)
    fig_error_along(best)
    SW = sweeps(best)
    fig_sweeps(SW)
    fig_hybrid(M)
    fig_pipeline()
    fig_bifurcation()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            globals()[f]()
    else:
        main()

"""Модульные и интеграционные тесты классического решателя."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pneumoshell.geometry import arc_area_term, arc_end, arc_points, design_three_tier, design_two_tier  # noqa
from pneumoshell.ml import polygon_area  # noqa
from pneumoshell.solver import ShellSolver  # noqa


def test_arc_end_full_circle():
    # полная окружность радиуса 0.5 возвращается в исходную точку
    x, y, th = arc_end(0.0, 0.0, 0.3, 2.0, np.pi)
    assert abs(x) < 1e-12 and abs(y) < 1e-12
    assert abs(th - (0.3 + 2 * np.pi)) < 1e-12


def test_arc_straight_limit():
    x, y, _ = arc_end(1.0, 2.0, np.pi / 4, 1e-12, 2.0)
    assert np.allclose([x, y], [1.0 + np.sqrt(2), 2.0 + np.sqrt(2)], atol=1e-10)


def test_arc_area_matches_polygon():
    # площадь кругового сегмента, замкнутого хордой
    rng = np.random.default_rng(0)
    for _ in range(20):
        k, s, th = rng.uniform(-4, 4), rng.uniform(0.1, 1.2), rng.uniform(-3, 3)
        pts = arc_points(0.2, -0.1, th, k, s, 4001)
        x1, y1 = pts[-1]
        exact = arc_area_term(0.2, -0.1, x1, y1, k, s) + 0.5 * (x1 * (-0.1) - 0.2 * y1)
        assert abs(exact - polygon_area(pts)) < 1e-6


DESIGN_FACTORIES = [design_two_tier, design_three_tier]


def test_reference_symmetry():
    for design in DESIGN_FACTORIES:
        _check_reference_symmetry(design)


def _check_reference_symmetry(design):
    """При равных давлениях и отсутствии нагрузки форма симметрична
    относительно вертикали x = b/2."""
    S = ShellSolver(design())
    ref = S.solve_reference(np.full(S.N, 6000.0))
    assert ref.converged and ref.valid
    P = S.contour(ref.segments, n_per_m=200)
    Pm = P.copy()
    Pm[:, 0] = S.d.b - Pm[:, 0]
    # каждая отражённая точка близка к контуру
    d = np.min(np.linalg.norm(Pm[:, None, :] - P[None, ::3, :], axis=2), axis=1)
    assert d.max() < 1e-2


def test_pressure_scaling_invariance():
    for design in DESIGN_FACTORIES:
        _check_pressure_scaling(design)


def _check_pressure_scaling(design):
    """Форма отсчётного состояния не зависит от общего уровня давления."""
    S = ShellSolver(design())
    a = S.solve_reference(np.full(S.N, 4000.0))
    b = S.solve_reference(np.full(S.N, 9000.0))
    assert np.allclose(S.material_points(a.segments), S.material_points(b.segments), atol=1e-8)


def test_loaded_equilibrium_and_adiabat():
    for design in DESIGN_FACTORIES:
        _check_loaded(design)


def _check_loaded(design):
    S = ShellSolver(design())
    p0 = np.array([6000.0, 7000.0, 5000.0][:S.N])
    sol = S.solve(p0, 1200.0, 0.15)
    assert sol.converged and sol.valid
    assert sol.residual_norm < 1e-9
    # адиабата: p = (pa + p0)(S0/S)^n - pa
    pa, n = S.d.p_atm, S.d.n_poly
    p_ad = (pa + p0) * (sol.info["S0"] / sol.areas) ** n - pa
    assert np.allclose(sol.p, p_ad, rtol=1e-6)
    # обжатие уменьшает объём и повышает давление
    assert np.all(sol.p > p0)
    # контакт: нижняя точка на опоре, длина контакта положительна
    assert sol.contact_length > 0 and abs(sol.gap) < 1e-9
    # нерастяжимость: длины мембран сохраняются
    L = {f"O{i+1}": S.d.L_out[i] for i in range(S.N - 1)}
    for k, v in L.items():
        assert abs(sol.segments[k].s - v) < 1e-12
    total_M = sum(sol.segments[k].s for k in ("M_out", "M_con", "M_in"))
    assert abs(total_M - S.d.L_m) < 1e-10


def test_no_contact_when_gap():
    S = ShellSolver(design_two_tier())
    sol = S.solve([6000.0, 6000.0], 500.0, -0.04)
    assert sol.converged and sol.contact_length == 0 and sol.gap > 0


def test_warm_start_reproduces_solution():
    S = ShellSolver(design_two_tier())
    a = S.solve([7000.0, 5000.0], 900.0, 0.12)
    xi = np.concatenate([a.x, [0.0]]) if a.info["mode"] == "free" else a.x
    b = S.solve([7000.0, 5000.0], 900.0, 0.12, x_init=xi * (1 + 1e-3))
    assert b.converged
    assert np.allclose(S.material_points(a.segments), S.material_points(b.segments), atol=1e-8)


if __name__ == "__main__":
    # запуск без pytest: python tests/test_solver.py
    import time
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    t0 = time.perf_counter()
    for name, fn in tests:
        fn()
        print(f"PASSED  {name}")
    print(f"{len(tests)} passed in {time.perf_counter() - t0:.2f} s")

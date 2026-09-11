"""Геометрия дуг окружностей и описание конструкции многоярусной пневмооболочки.

Поперечное сечение оболочки представляется набором нерастяжимых мембран.
Каждая мембрана под действием постоянного перепада давлений принимает форму
дуги окружности (формула Лапласа для безмоментной нити: T = dp * R), поэтому
её положение полностью задаётся начальной точкой, начальным углом касательной,
кривизной kappa = dp / T и длиной s.

Соглашение о знаках: kappa = (p_left - p_right) / T, где «слева» —
относительно направления обхода мембраны. Положительная кривизна означает
поворот касательной против часовой стрелки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def _sinc(u):
    """sin(u)/u с корректным пределом при u -> 0."""
    return np.sinc(np.asarray(u) / np.pi)


def arc_end(x0: float, y0: float, th0: float, kappa: float, s: float):
    """Конечная точка и угол касательной дуги (x0, y0, th0, kappa, s).

    Формула записана через хорду: хорда направлена под средним углом
    th0 + kappa*s/2 и имеет длину s * sinc(kappa*s/2). Это устойчиво
    и при kappa -> 0 (прямой отрезок).
    """
    half = 0.5 * kappa * s
    chord = s * float(_sinc(half))
    thm = th0 + half
    return x0 + chord * np.cos(thm), y0 + chord * np.sin(thm), th0 + kappa * s


def arc_points(x0, y0, th0, kappa, s, n):
    """n точек, равномерно распределённых по длине дуги (включая концы)."""
    t = np.linspace(0.0, s, n)
    half = 0.5 * kappa * t
    chord = t * _sinc(half)
    thm = th0 + half
    return np.column_stack([x0 + chord * np.cos(thm), y0 + chord * np.sin(thm)])


def arc_area_term(x0, y0, x1, y1, kappa, s):
    """Вклад дуги в ориентированную площадь замкнутого контура.

    Площадь = сумма по дугам [ (x0*y1 - x1*y0)/2 + сегмент ], где сегмент —
    площадь между хордой и дугой со знаком kappa:
        seg = (u - sin u) / (2 kappa^2),   u = kappa * s.
    При обходе против часовой стрелки результат положителен.
    """
    u = kappa * s
    if abs(u) < 1e-4:
        seg = kappa * s ** 3 / 12.0 - kappa ** 3 * s ** 5 / 240.0
    else:
        seg = (u - np.sin(u)) / (2.0 * kappa ** 2)
    return 0.5 * (x0 * y1 - x1 * y0) + seg


@dataclass
class ShellDesign:
    """Конструктивные параметры многоярусной пневмооболочки.

    Ось x направлена от наружной (атмосферной) стороны к внутренней стороне
    (стороне воздушной подушки), ось y — вверх; днище корпуса y = 0.
    Верхний ярус закреплён на корпусе в точках A = (0, 0) и B = (b, 0).

    Для ярусов i = 1..N-1 заданы длины наружной стенки L_out[i],
    внутренней стенки L_in[i] и перегородки (диафрагмы) L_w[i] между ярусами
    i и i+1. Нижний ярус N ограничен снизу одной мембраной длины L_m,
    которая может контактировать с опорной поверхностью.
    """

    name: str
    n_tiers: int
    b: float
    L_out: List[float]
    L_in: List[float]
    L_w: List[float]
    L_m: float
    # число точек дискретизации на погонный метр (для выходного вектора формы)
    pts_per_m: float = 40.0
    p_atm: float = 101325.0
    n_poly: float = 1.4
    description: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        k = self.n_tiers - 1
        assert len(self.L_out) == k and len(self.L_in) == k and len(self.L_w) == k

    # --- дискретизация формы ---------------------------------------------
    def n_pts(self, L: float) -> int:
        return max(8, int(round(L * self.pts_per_m)) + 1)

    @property
    def total_length(self) -> float:
        return sum(self.L_out) + sum(self.L_in) + sum(self.L_w) + self.L_m


def design_two_tier() -> ShellDesign:
    """Двухъярусная оболочка (высота в свободном состоянии ~0.9 м)."""
    return ShellDesign(
        name="two_tier",
        n_tiers=2,
        b=0.50,
        L_out=[0.42],
        L_in=[0.42],
        L_w=[0.64],
        L_m=1.75,
        description="Двухъярусная пневмооболочка: верхний баллон + нижний баллон",
    )


def design_three_tier() -> ShellDesign:
    """Трёхъярусная оболочка (высота в свободном состоянии ~1.1 м)."""
    return ShellDesign(
        name="three_tier",
        n_tiers=3,
        b=0.50,
        L_out=[0.36, 0.36],
        L_in=[0.36, 0.36],
        L_w=[0.60, 0.66],
        L_m=1.70,
        description="Трёхъярусная пневмооболочка: три баллона, соединённые шарнирно",
    )


DESIGNS = {"two_tier": design_two_tier, "three_tier": design_three_tier}

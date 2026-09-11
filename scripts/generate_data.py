"""Генерация датасетов для двух- и трёхъярусной оболочек.

Пример:  python scripts/generate_data.py --n 4000 --processes 2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pneumoshell.dataset import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["two_tier", "three_tier"])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for k, name in enumerate(a.designs):
        data = generate(name, a.n, seed=a.seed + k, processes=a.processes)
        path = os.path.join(a.out, f"dataset_{name}.npz")
        np.savez_compressed(path, **data)
        print("saved", path, data["X"].shape, data["Y"].shape)


if __name__ == "__main__":
    main()

"""Уточнение задержки одиночного прогноза для ансамблевых моделей: при
обучении они используют 2 потока (n_jobs=2), а на время прогноза одного случая
накладные расходы на распараллеливание искажают замер. Здесь модели
переобучаются и замеряются в однопоточном режиме."""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pneumoshell.ml import ShapeSurrogate, load_dataset  # noqa: E402

for name in ["two_tier", "three_tier"]:
    path = os.path.join(ROOT, "results", f"metrics_{name}.json")
    res = json.load(open(path))
    d = load_dataset(os.path.join(ROOT, "data", f"dataset_{name}.npz"), seed=0)
    for mn in ["rf", "gbm"]:
        sur = ShapeSurrogate.train(name, mn, d["train"], n_components=24)
        if mn == "rf":
            sur.model.n_jobs = 1
        else:
            sur.model.n_jobs = None
        te = d["test"].X
        ts = []
        for i in range(100):
            t1 = time.perf_counter()
            sur.predict(te[i:i + 1])
            ts.append(time.perf_counter() - t1)
        t0 = time.perf_counter()
        sur.predict(te)
        tb = (time.perf_counter() - t0) / len(te)
        res["models"][mn]["t_single_ms"] = float(np.median(ts) * 1000)
        res["models"][mn]["t_batch_ms"] = tb * 1000
        print(name, mn, res["models"][mn]["t_single_ms"], tb * 1000)
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)

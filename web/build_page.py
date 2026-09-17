"""Сборка интерактивной страницы: подстановка весов MLP и метрик в шаблон."""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tpl = open(os.path.join(ROOT, "web", "template.html"), encoding="utf8").read()
data = json.load(open(os.path.join(ROOT, "results", "web_models.json")))
metrics = {}
solver_ms = []
for n in ["two_tier", "three_tier"]:
    r = json.load(open(os.path.join(ROOT, "results", f"metrics_{n}.json")))
    metrics[n] = {k: dict(rmse_mm=v["rmse_mm"], r2=v["r2"], p_mape_pct=v["p_mape_pct"],
                          t_single_ms=v["t_single_ms"]) for k, v in r["models"].items()}
    solver_ms.append(r["hybrid"]["summary"]["t_cold_mean_ms"])
page = tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
page = page.replace("/*METRICS*/null", json.dumps(metrics, separators=(",", ":")))
page = page.replace("{{SOLVER_MS}}", f"в среднем {solver_ms[0]:.0f} мс (2 яруса) и {solver_ms[1]:.0f} мс (3 яруса)"
                    .replace(".", ","))
out = os.path.join(ROOT, "web", "pneumoshell_demo.html")
open(out, "w", encoding="utf8").write(page)
print(out, len(page) // 1024, "KB")

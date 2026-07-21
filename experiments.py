

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import os
import platform
import random
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import codegen
from codegen import MODEL_CONFIGS, PipelineConfig

@dataclass
class ExperimentSuite:
  name: str
  description: str = ""
  base: Dict[str, Any] = field(default_factory=dict)
  grid: Dict[str, List[Any]] = field(default_factory=dict)
  seeds: List[int] = field(default_factory=lambda: [42])

  def expand(self) -> List[Dict[str, Any]]:
    """Разворачивает grid × seeds в список словарей-переопределений."""
    keys = list(self.grid.keys())
    value_lists = [self.grid[k] for k in keys]
    combos = list(itertools.product(*value_lists)) if keys else [()]

    runs = []
    for combo in combos:
      grid_override = dict(zip(keys, combo))
      for seed in self.seeds:
        override = {**self.base, **grid_override, "seed": seed}
        runs.append({"grid": grid_override, "seed": seed, "override": override})
    return runs


SUITES: Dict[str, ExperimentSuite] = {
  "smoke": ExperimentSuite(
    name="smoke",
    description="Быстрая проверка работоспособности (синтетика, 1 прогон).",
    base=dict(
      data_source="synthetic", num_samples=80, epochs=3,
      wpd_mode="center", model_name="compact",
    ),
    grid={},
    seeds=[42],
  ),

  "model_comparison": ExperimentSuite(
    name="model_comparison",
    description="Сравнение архитектур CNN на синтетике при равных данных.",
    base=dict(
      data_source="synthetic", num_samples=500, epochs=20,
      wpd_mode="stride", wpd_sensor_stride=8,
    ),
    grid={"model_name": ["paper", "compact", "deep", "wide"]},
    seeds=[42, 1, 7],
  ),

  "wpd_ablation": ExperimentSuite(
    name="wpd_ablation",
    description="Влияние режима WPD и числа отобранных каналов на качество.",
    base=dict(
      data_source="synthetic", num_samples=500, epochs=15, model_name="compact",
    ),
    grid={
      "wpd_mode": ["center", "stride"],
      "top_k_channels": [3, 6],
    },
    seeds=[42, 1],
  ),

  "optimizer_lr": ExperimentSuite(
    name="optimizer_lr",
    description="Подбор оптимизатора и learning rate.",
    base=dict(
      data_source="synthetic", num_samples=500, epochs=20,
      wpd_mode="stride", model_name="compact",
    ),
    grid={
      "optimizer": ["adam", "adamw"],
      "lr": [1e-3, 3e-4],
    },
    seeds=[42, 1],
  ),

  "prefilter_sweep": ExperimentSuite(
    name="prefilter_sweep",
    description="Настройка предфильтра: размерность PCA, степень полинома, вес класса.",
    base=dict(
      data_source="synthetic", num_samples=500, epochs=10,
      wpd_mode="stride", model_name="compact",
    ),
    grid={
      "prefilter_k": [4, 8],
      "prefilter_degree": [2, 4],
      "prefilter_class_weight": [100.0, 800.0],
    },
    seeds=[42],
  ),

  "h5_models": ExperimentSuite(
    name="h5_models",
    description="Сравнение spectrogram-моделей на реальных H5-данных.",
    base=dict(
      data_source="h5", h5_max_samples=8000, epochs=15, batch_size=64,
    ),
    grid={"model_name": ["spectrogram", "spectrogram_deep"]},
    seeds=[42, 1],
  ),
}


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Утилиты для метрик/сериализации
# ---------------------------------------------------------------------------

def _run_name(grid_override: Dict[str, Any], seed: int) -> str:
  if not grid_override:
    parts = ["base"]
  else:
    parts = [f"{k}={v}" for k, v in grid_override.items()]
  return "__".join(parts) + f"__seed{seed}"


def flatten_metrics(results: Dict[str, object]) -> Dict[str, float]:
  """Достаёт числовые метрики из результата codegen.main в плоский словарь."""
  flat: Dict[str, float] = {}
  history = results.get("history")
  if history is not None:
    flat["best_val_acc"] = float(history.best_val_acc)
    flat["best_epoch"] = float(history.best_epoch)
    flat["epochs_run"] = float(len(history.val_accs))

  for prefix, key in (("cnn", "cnn_metrics"), ("casc", "cascade_metrics")):
    block = results.get(key)
    if isinstance(block, dict):
      for m, v in block.items():
        flat[f"{prefix}_{m}"] = float(v)

  if "prefilter_threshold" in results:
    flat["prefilter_threshold"] = float(results["prefilter_threshold"])
  return flat


def history_to_dict(results: Dict[str, object]) -> Dict[str, Any]:
  history = results.get("history")
  if history is None:
    return {}
  return {
    "best_val_acc": history.best_val_acc,
    "best_epoch": history.best_epoch,
    "train_losses": history.train_losses,
    "val_accs": history.val_accs,
  }

def run_single(
  override: Dict[str, Any],
  run_dir: str,
) -> Dict[str, Any]:
  os.makedirs(run_dir, exist_ok=True)
  cfg = PipelineConfig(**override)
  cfg.checkpoint_path = os.path.join(run_dir, "best_cnn.pth")

  set_seed(cfg.seed)

  with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
    json.dump(dataclasses.asdict(cfg), fh, ensure_ascii=False, indent=2)

  t0 = time.time()
  results = codegen.main(cfg)
  elapsed = time.time() - t0

  metrics = flatten_metrics(results)
  metrics["elapsed_sec"] = round(elapsed, 2)

  payload = {"metrics": metrics, "history": history_to_dict(results)}
  with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)

  return metrics


def run_suite(suite: ExperimentSuite, results_root: str) -> str:
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  suite_dir = os.path.join(results_root, f"{suite.name}__{timestamp}")
  runs_dir = os.path.join(suite_dir, "runs")
  os.makedirs(runs_dir, exist_ok=True)

  runs = suite.expand()
  print(f"\n{'=' * 70}")
  print(f"SUITE: {suite.name} — {suite.description}")
  print(f"Прогонов: {len(runs)} (точек сетки x сидов)")
  print(f"Каталог: {suite_dir}")
  print(f"{'=' * 70}")

  with open(os.path.join(suite_dir, "suite.json"), "w", encoding="utf-8") as fh:
    json.dump(
      {
        "suite": dataclasses.asdict(suite),
        "environment": {
          "python": sys.version.split()[0],
          "platform": platform.platform(),
          "torch": torch.__version__,
          "numpy": np.__version__,
          "cuda_available": torch.cuda.is_available(),
        },
        "timestamp": timestamp,
      },
      fh, ensure_ascii=False, indent=2,
    )

  all_rows: List[Dict[str, Any]] = []
  for i, run in enumerate(runs, start=1):
    name = _run_name(run["grid"], run["seed"])
    run_dir = os.path.join(runs_dir, name)
    print(f"\n[{i}/{len(runs)}] {name}")

    row: Dict[str, Any] = {**run["grid"], "seed": run["seed"], "run_name": name}
    try:
      metrics = run_single(run["override"], run_dir)
      row.update(metrics)
      row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 — фиксируем сбой прогона, не роняя suite
      row["status"] = f"error: {exc}"
      with open(os.path.join(run_dir, "error.txt"), "w", encoding="utf-8") as fh:
        fh.write(traceback.format_exc())
      print(f"  ОШИБКА: {exc}")
    all_rows.append(row)

  _write_runs_csv(os.path.join(suite_dir, "runs.csv"), all_rows)
  _write_summary_csv(os.path.join(suite_dir, "summary.csv"), all_rows, suite)

  print(f"\nГотово. Результаты: {suite_dir}")
  return suite_dir


def _write_runs_csv(path: str, rows: List[Dict[str, Any]]) -> None:
  if not rows:
    return
  fields: List[str] = []
  for row in rows:
    for k in row:
      if k not in fields:
        fields.append(k)
  with open(path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def _write_summary_csv(
  path: str,
  rows: List[Dict[str, Any]],
  suite: ExperimentSuite,
) -> None:
  """Агрегирует mean±std по сидам для каждой точки сетки."""
  ok_rows = [r for r in rows if r.get("status") == "ok"]
  if not ok_rows:
    return

  grid_keys = list(suite.grid.keys())
  metric_keys = sorted(
    {k for r in ok_rows for k, v in r.items()
     if isinstance(v, (int, float)) and k not in grid_keys and k != "seed"}
  )

  groups: Dict[tuple, List[Dict[str, Any]]] = {}
  for r in ok_rows:
    key = tuple(r.get(k) for k in grid_keys)
    groups.setdefault(key, []).append(r)

  summary_rows = []
  for key, group in groups.items():
    out: Dict[str, Any] = dict(zip(grid_keys, key))
    out["n_seeds"] = len(group)
    for m in metric_keys:
      vals = [r[m] for r in group if m in r]
      if not vals:
        continue
      out[f"{m}_mean"] = round(statistics.mean(vals), 4)
      out[f"{m}_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
    summary_rows.append(out)

  sort_key = "casc_e2e_recall_mean" if any(
    "casc_e2e_recall_mean" in r for r in summary_rows
  ) else "cnn_recall_mean"
  summary_rows.sort(key=lambda r: r.get(sort_key, 0.0), reverse=True)

  _write_runs_csv(path, summary_rows)

  print("\n--- Сводка (сортировка по", sort_key, ") ---")
  for r in summary_rows:
    grid_str = ", ".join(f"{k}={r[k]}" for k in grid_keys) or "base"
    key_metric = r.get(sort_key, float("nan"))
    print(f"  {grid_str:40s} {sort_key}={key_metric:.4f} (n={r['n_seeds']})")


def main() -> None:
  parser = argparse.ArgumentParser(description="Раннер экспериментов над codegen.py")
  parser.add_argument("--suite", help="Имя suite для запуска")
  parser.add_argument("--results-dir", default="results", help="Корень для результатов")
  parser.add_argument("--list", action="store_true", help="Показать доступные suites")
  args = parser.parse_args()

  if args.list or not args.suite:
    print("Доступные эксперименты:")
    for name, s in SUITES.items():
      n_runs = len(s.expand())
      print(f"  {name:20s} прогонов={n_runs:<3d} — {s.description}")
    print(f"\nМодели: {', '.join(MODEL_CONFIGS)}")
    return

  if args.suite not in SUITES:
    raise SystemExit(f"Неизвестный suite: {args.suite}. Доступны: {list(SUITES)}")

  run_suite(SUITES[args.suite], args.results_dir)


if __name__ == "__main__":
  main()

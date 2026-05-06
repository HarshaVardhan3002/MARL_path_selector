"""Phase 2 sweep launcher.

Reads `configs/phase2/sweep_*.yaml`, expands across seeds, runs 3
configs in parallel via Python multiprocessing (each config × seed in
its own process). Per PHASE_2_SPEC §1.4: 9 configs × 3 seeds = 27 runs,
9 batches of 3 parallel. Per-run output at
`runs/phase2/{config}/seed_{seed}/`; sweep-level log at
`runs/phase2/sweep.log`.

Usage:
    uv run python scripts/run_phase2_sweep.py
    uv run python scripts/run_phase2_sweep.py --only sweep_01_balanced
    uv run python scripts/run_phase2_sweep.py --seeds 42
    uv run python scripts/run_phase2_sweep.py --parallelism 2   # GPU pressure fallback
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import yaml

from scion_marl.config import Phase2Config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs" / "phase2"
RUNS_ROOT = REPO_ROOT / "runs" / "phase2"
DEFAULT_SEEDS = (42, 1337, 2024)


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase2_sweep")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def _load_config(yaml_path: Path, seed: int, episodes_per_run: int | None = None) -> Phase2Config:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    payload["seed"] = seed
    if episodes_per_run is not None:
        payload.setdefault("mappo", {})
        payload["mappo"]["episodes_per_run"] = episodes_per_run
    return Phase2Config.model_validate(payload)


def _run_one(args: tuple[str, int, int | None]) -> dict:
    """Worker: train one (config, seed). Imports torch lazily inside the worker."""
    yaml_path_str, seed, episodes_per_run = args
    # Defer imports so the parent process doesn't hold a torch handle that
    # breaks multiprocessing on Windows (spawn).
    from scion_marl.mappo import MappoTrainer
    from scion_marl.device import select_device

    yaml_path = Path(yaml_path_str)
    cfg = _load_config(yaml_path, seed=seed, episodes_per_run=episodes_per_run)
    run_dir = RUNS_ROOT / cfg.name / f"seed_{seed}"
    device_info = select_device()
    trainer = MappoTrainer(cfg, run_dir=run_dir, device_info=device_info)
    t0 = time.perf_counter()
    try:
        result = trainer.train()
        return {
            "name": cfg.name,
            "seed": seed,
            "ok": True,
            "elapsed_sec": result.elapsed_sec,
            "auto_restarted": result.auto_restarted,
            "final_episode": result.final_episode,
            "final_reward": result.final_metrics.get("episode_reward"),
            "final_entropy": result.final_metrics.get("policy_entropy"),
            "checkpoint": str(result.checkpoint_path),
            "device_backend": device_info.backend,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "name": cfg.name,
            "seed": seed,
            "ok": False,
            "elapsed_sec": elapsed,
            "error": repr(e),
            "device_backend": device_info.backend,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated config names; omit to run all.")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--parallelism", type=int, default=3,
                   help="Configs to run in parallel within a batch (drop to 2 on GPU pressure).")
    p.add_argument("--configs-dir", type=str, default=str(CONFIGS_DIR))
    p.add_argument("--episodes-per-run", type=int, default=None,
                   help="Override mappo.episodes_per_run from YAML (e.g., 200 for pilot, 1500 for full).")
    args = p.parse_args()

    sweep_log = RUNS_ROOT / "sweep.log"
    logger = _setup_logging(sweep_log)
    logger.info("=" * 60)
    logger.info("Phase 2 sweep starting")
    logger.info("=" * 60)

    cfg_dir = Path(args.configs_dir)
    if args.only:
        names = {n.strip() for n in args.only.split(",") if n.strip()}
        yaml_paths = sorted(cfg_dir / f"{n}.yaml" for n in names)
    else:
        yaml_paths = sorted(cfg_dir.glob("sweep_*.yaml"))
    seeds = tuple(int(s) for s in args.seeds.split(","))
    payloads: list[tuple[str, int, int | None]] = [
        (str(p), s, args.episodes_per_run) for p in yaml_paths for s in seeds
    ]
    logger.info(
        "configs: %d, seeds: %s, total runs: %d, episodes_per_run override: %s",
        len(yaml_paths), seeds, len(payloads),
        args.episodes_per_run if args.episodes_per_run else "(YAML default)",
    )

    if args.parallelism <= 0:
        raise SystemExit("parallelism must be >= 1")
    parallelism = args.parallelism

    t0 = time.perf_counter()
    results: list[dict] = []
    if parallelism == 1:
        for payload in payloads:
            r = _run_one(payload)
            _log_result(logger, r)
            results.append(r)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=parallelism) as pool:
            for r in pool.imap_unordered(_run_one, payloads, chunksize=1):
                _log_result(logger, r)
                results.append(r)

    elapsed = time.perf_counter() - t0
    summary_path = RUNS_ROOT / "sweep_summary.json"
    summary_path.write_text(json.dumps({
        "elapsed_sec": elapsed,
        "n_runs": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_failed": sum(1 for r in results if not r.get("ok")),
        "n_auto_restarted": sum(1 for r in results if r.get("auto_restarted")),
        "results": results,
    }, indent=2))
    logger.info("sweep complete: %.1f sec (%.2f h); summary -> %s",
                elapsed, elapsed / 3600.0, summary_path)


def _log_result(logger: logging.Logger, r: dict) -> None:
    if r.get("ok"):
        logger.info(
            "[%s/seed_%d] OK in %.1fs reward=%.2f entropy=%.3f auto_restart=%s device=%s",
            r["name"], r["seed"], r["elapsed_sec"],
            r.get("final_reward", float("nan")),
            r.get("final_entropy", float("nan")),
            r.get("auto_restarted", False),
            r.get("device_backend", "?"),
        )
    else:
        logger.error(
            "[%s/seed_%d] FAILED in %.1fs: %s",
            r["name"], r["seed"], r["elapsed_sec"], r.get("error", "unknown"),
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()

"""Extract empirical summary statistics from the public ScionPathML measurement
release (mpquic-on-scion-ipc, pinned in JUSTIFICATIONS.md J-007).

Inputs (relative to project root):
    external/scionpathml_extracted/AnalysisResults/Lab{1,2,3}/Data/raw_data.zip
    external/scionpathml_extracted/AnalysisResults/Lab{1,2,3}/Data/selected_paths.json

Output:
    data/scionpathml_summary.json

Computed statistics (per JUSTIFICATIONS.md J-006):
    - path_lifetime_seconds: pooled across labs and AS pairs
        - sample_size, mean, median, p25, p75, p95, p99,
        - empirical CDF (~200 quantile points)
    - rtt_ms: per (source_lab, destination_ia) pair
        - count, mean, median, IQR, p5, p95, full sample (downsampled to 5000 if larger)
    - jitter_ms: same shape as rtt_ms (mdev_rtt as proxy)
    - loss_pct: same shape
    - bandwidth_mbps: per (source_lab, destination_ia, tier, direction) tuple
    - path_discrepancy_rate: AS-pair-level proxy = |added - removed| / max(added, removed)
    - churn_events_per_hour: total change events / observation window in hours

Run via:
    uv run python calibration/extract_scionpathml_summary.py
"""
from __future__ import annotations

import json
import logging
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("scionpathml.extract")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_ROOT = PROJECT_ROOT / "external" / "scionpathml_extracted"
OUTPUT_PATH = PROJECT_ROOT / "data" / "scionpathml_summary.json"

LABS = ("Lab1", "Lab2", "Lab3")
TIMESTAMP_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")
EXCLUDED_DESTINATIONS = {"22-ffaa:1:11ee"}  # Taiwan AS — unavailable for entire campaign (J-007)


def _parse_ts(s: str) -> datetime | None:
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _empirical_cdf(values: np.ndarray, n_points: int = 200) -> dict[str, list[float]]:
    """Return (quantiles, values) suitable for KS computation downstream."""
    if values.size == 0:
        return {"quantiles": [], "values": []}
    sorted_vals = np.sort(values)
    qs = np.linspace(0.0, 1.0, n_points)
    indices = np.clip((qs * (sorted_vals.size - 1)).round().astype(int), 0, sorted_vals.size - 1)
    return {"quantiles": qs.tolist(), "values": sorted_vals[indices].tolist()}


def _summarize_marginal(values: np.ndarray, max_sample: int = 5000) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }
    out["iqr"] = out["p75"] - out["p25"]
    if values.size > max_sample:
        rng = np.random.default_rng(0)
        out["sample"] = rng.choice(values, size=max_sample, replace=False).tolist()
    else:
        out["sample"] = values.tolist()
    return out


def _iter_json_in_zip(zip_path: Path, prefix: str):
    """Yield (filename, parsed_json) for every JSON file whose name starts with prefix."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.rsplit("/", 1)[-1]
            if not name.startswith(prefix) or not name.endswith(".json"):
                continue
            try:
                with zf.open(info) as fh:
                    yield info.filename, json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning("Skipping unreadable %s: %s", info.filename, e)


def extract_path_lifetimes(zip_path: Path, lab: str) -> tuple[list[float], dict[str, dict[str, int]], list[datetime]]:
    """Replicates the upstream analyze_comparer.py logic against a zip stream.

    Returns:
        all_lifetimes_sec — flat list of observed (added → removed) lifetimes
        per_dest_counts   — {destination_ia: {added, removed, change_events}}
        change_event_ts   — every timestamp at which a change_status=='change_detected' was seen
    """
    all_lifetimes: list[float] = []
    change_event_ts: list[datetime] = []
    per_dest_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"added": 0, "removed": 0, "change_events": 0})

    by_destination: dict[str, list[dict]] = defaultdict(list)
    for _name, doc in _iter_json_in_zip(zip_path, "delta_"):
        dest = doc.get("destination")
        if dest and dest not in EXCLUDED_DESTINATIONS:
            by_destination[dest].append(doc)

    for dest, entries in by_destination.items():
        entries.sort(key=lambda d: d.get("timestamp", ""))
        last_seen: dict[str, datetime] = {}
        for entry in entries:
            ts = _parse_ts(entry.get("timestamp", ""))
            if ts is None:
                continue
            if entry.get("change_status") == "change_detected":
                per_dest_counts[dest]["change_events"] += 1
                change_event_ts.append(ts)
            for change in entry.get("changes") or []:
                ctype = change.get("change")
                seq = (change.get("sequence") or "").strip()
                if not seq:
                    continue
                if ctype == "added":
                    per_dest_counts[dest]["added"] += 1
                    last_seen[seq] = ts
                elif ctype == "removed":
                    per_dest_counts[dest]["removed"] += 1
                    if seq in last_seen:
                        all_lifetimes.append((ts - last_seen[seq]).total_seconds())
                        last_seen.pop(seq, None)

    logger.info("[%s] %d path lifetimes, %d destinations", lab, len(all_lifetimes), len(per_dest_counts))
    return all_lifetimes, dict(per_dest_counts), change_event_ts


def extract_prober_marginals(zip_path: Path, lab: str) -> dict[str, dict[str, np.ndarray]]:
    """Per-destination RTT / mdev / loss arrays from prober_*.json files."""
    by_dest: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"rtt": [], "mdev": [], "loss": []})
    n_files = 0
    for _name, doc in _iter_json_in_zip(zip_path, "prober_"):
        n_files += 1
        ia = doc.get("ia")
        if not ia or ia in EXCLUDED_DESTINATIONS:
            continue
        for probe in doc.get("probes") or []:
            stats = (probe.get("ping_result") or {}).get("statistics") or {}
            rtt = stats.get("avg_rtt")
            mdev = stats.get("mdev_rtt")
            loss = stats.get("packet_loss")
            if rtt is not None:
                by_dest[ia]["rtt"].append(float(rtt))
            if mdev is not None:
                by_dest[ia]["mdev"].append(float(mdev))
            if loss is not None:
                by_dest[ia]["loss"].append(float(loss))
    logger.info("[%s] prober: %d files, destinations=%s", lab, n_files, sorted(by_dest))
    return {dest: {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()} for dest, metrics in by_dest.items()}


_BW_DIR_KEYS = (("sc", "S->C results"), ("cs", "C->S results"))
_LOSS_DIR_KEYS = _BW_DIR_KEYS  # same JSON path, different field


def _parse_mbps(s: str | float | int | None) -> float | None:
    """Parse strings like '10000000 bps / 10.00 Mbps' or '85.31 Mbps' or numerics."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    if "Mbps" in s:
        # take the substring just before 'Mbps' and trim back to the number
        head = s.split("Mbps", 1)[0]
        token = head.strip().split()[-1]
        try:
            return float(token)
        except ValueError:
            return None
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return None


def _parse_loss_pct(s: str | float | int | None) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def extract_bandwidth_marginals(
    zip_path: Path, lab: str
) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[tuple[str, int, str], np.ndarray]]:
    """BW + loss rows keyed by (destination_ia, tier_mbps, direction).

    Filename prefixes 'BW_' (sequential) and 'BW-P_' (parallel/multipath). JSON shape:
        {timestamp, as, target_mbps, paths: [
            {fingerprint, sequence, result: {"S->C results": {achieved_bps, loss_rate, ...},
                                              "C->S results": {...}}}
        ]}
    `achieved_bps` is a string like '10000000 bps / 10.00 Mbps'; we parse the Mbps tail.
    """
    bw: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    loss: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    n_files = 0
    for prefix in ("BW_", "BW-P_"):
        for _name, doc in _iter_json_in_zip(zip_path, prefix):
            n_files += 1
            ia = doc.get("as") or doc.get("ia") or doc.get("destination")
            if not ia or ia in EXCLUDED_DESTINATIONS:
                continue
            tier_raw = doc.get("target_mbps") or doc.get("target_bw_mbps") or doc.get("tier")
            try:
                tier = int(tier_raw) if tier_raw is not None else None
            except (TypeError, ValueError):
                tier = None
            for path_entry in doc.get("paths") or []:
                result = path_entry.get("result") or {}
                if not isinstance(result, dict):
                    continue
                # tier may be on the per-path target if not at top level
                local_tier = tier
                if local_tier is None:
                    target_block = path_entry.get("target") or {}
                    try:
                        local_tier = int(target_block.get("tier_mbps")) if target_block.get("tier_mbps") is not None else None
                    except (TypeError, ValueError):
                        local_tier = None
                if local_tier is None:
                    continue
                for dir_short, dir_key in _BW_DIR_KEYS:
                    sub = result.get(dir_key) or {}
                    if not isinstance(sub, dict):
                        continue
                    achieved = _parse_mbps(sub.get("achieved_bps") or sub.get("achieved_bw_mbps"))
                    if achieved is not None:
                        bw[(ia, local_tier, dir_short)].append(achieved)
                    lr = _parse_loss_pct(sub.get("loss_rate"))
                    if lr is not None:
                        loss[(ia, local_tier, dir_short)].append(lr)
    logger.info("[%s] bandwidth: %d files, %d (dest,tier,dir) cells", lab, n_files, len(bw))
    return (
        {k: np.asarray(v, dtype=np.float64) for k, v in bw.items()},
        {k: np.asarray(v, dtype=np.float64) for k, v in loss.items()},
    )


def compute_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "version": 1,
        "source_repo": "github.com/Keshvadi/mpquic-on-scion-ipc",
        "source_repo_pinned_sha": "362ad60885dabeb2a067d3dae058c506ce5f092a",
        "excluded_destinations": sorted(EXCLUDED_DESTINATIONS),
        "labs_used": [],
        "path_lifetime_seconds": {},
        "rtt_ms": {},
        "jitter_ms": {},
        "loss_pct": {},
        "bandwidth_mbps": {},
        "bandwidth_loss_pct": {},
        "path_discrepancy_rate": {},
        "churn_events_per_hour": {},
    }

    pooled_lifetimes: list[float] = []
    pooled_change_events: list[datetime] = []
    rtt_by_pair: dict[str, list[float]] = defaultdict(list)
    mdev_by_pair: dict[str, list[float]] = defaultdict(list)
    loss_by_pair: dict[str, list[float]] = defaultdict(list)
    bw_by_pair: dict[str, list[float]] = defaultdict(list)
    bw_loss_by_pair: dict[str, list[float]] = defaultdict(list)
    discrepancy_by_dest: dict[str, list[float]] = defaultdict(list)

    for lab in LABS:
        zip_path = EXTRACTED_ROOT / "AnalysisResults" / lab / "Data" / "raw_data.zip"
        if not zip_path.exists():
            logger.warning("Skipping %s: %s missing (run scripts/refresh_external.sh)", lab, zip_path)
            continue
        summary["labs_used"].append(lab)

        lifetimes, dest_counts, change_ts = extract_path_lifetimes(zip_path, lab)
        pooled_lifetimes.extend(lifetimes)
        pooled_change_events.extend(change_ts)

        for dest, counts in dest_counts.items():
            added, removed = counts["added"], counts["removed"]
            denom = max(added, removed, 1)
            discrepancy_by_dest[dest].append(abs(added - removed) / denom)

        prober = extract_prober_marginals(zip_path, lab)
        for dest, metrics in prober.items():
            key = f"{lab}->{dest}"
            rtt_by_pair[key].extend(metrics["rtt"].tolist())
            mdev_by_pair[key].extend(metrics["mdev"].tolist())
            loss_by_pair[key].extend(metrics["loss"].tolist())

        bw, bw_loss = extract_bandwidth_marginals(zip_path, lab)
        for (dest, tier, direction), arr in bw.items():
            key = f"{lab}->{dest}|tier={tier}|dir={direction}"
            bw_by_pair[key].extend(arr.tolist())
        for (dest, tier, direction), arr in bw_loss.items():
            key = f"{lab}->{dest}|tier={tier}|dir={direction}"
            bw_loss_by_pair[key].extend(arr.tolist())

    if pooled_lifetimes:
        arr = np.asarray(pooled_lifetimes, dtype=np.float64)
        summary["path_lifetime_seconds"] = _summarize_marginal(arr, max_sample=10000)
        summary["path_lifetime_seconds"]["empirical_cdf"] = _empirical_cdf(arr)

    for key, vals in rtt_by_pair.items():
        summary["rtt_ms"][key] = _summarize_marginal(np.asarray(vals, dtype=np.float64))
    for key, vals in mdev_by_pair.items():
        summary["jitter_ms"][key] = _summarize_marginal(np.asarray(vals, dtype=np.float64))
    for key, vals in loss_by_pair.items():
        summary["loss_pct"][key] = _summarize_marginal(np.asarray(vals, dtype=np.float64))
    for key, vals in bw_by_pair.items():
        summary["bandwidth_mbps"][key] = _summarize_marginal(np.asarray(vals, dtype=np.float64))
    for key, vals in bw_loss_by_pair.items():
        summary["bandwidth_loss_pct"][key] = _summarize_marginal(np.asarray(vals, dtype=np.float64))

    for dest, rates in discrepancy_by_dest.items():
        summary["path_discrepancy_rate"][dest] = {
            "per_lab_rates": rates,
            "mean": float(np.mean(rates)) if rates else None,
        }

    if pooled_change_events:
        ts_sorted = sorted(pooled_change_events)
        window_hours = max((ts_sorted[-1] - ts_sorted[0]).total_seconds() / 3600.0, 1.0)
        summary["churn_events_per_hour"] = {
            "total_events": len(pooled_change_events),
            "window_hours": window_hours,
            "events_per_hour": len(pooled_change_events) / window_hours,
        }

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    summary = compute_summary()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Wrote %s (%d bytes)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size)


if __name__ == "__main__":
    main()

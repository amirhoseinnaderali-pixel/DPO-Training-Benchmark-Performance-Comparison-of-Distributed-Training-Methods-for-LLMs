import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def flatten_result(result):
    peak = result.get("peak_memory", {})
    train = result.get("train_metrics", {})
    ev = result.get("eval_metrics", {})
    env = result.get("environment", {})
    gpus = env.get("gpus", [])
    return {
        "run_id": result.get("run_id"),
        "method": result.get("method"),
        "seed": result.get("seed"),
        "train_examples": result.get("train_examples"),
        "eval_examples": result.get("eval_examples"),
        "world_size": result.get("world_size"),
        "effective_batch_size": result.get("effective_batch_size"),
        "optimizer_steps": result.get("optimizer_steps"),
        "learning_rate": result.get("learning_rate"),
        "train_loop_seconds": result.get("train_loop_seconds"),
        "trainer_runtime_seconds": train.get("train_runtime"),
        "examples_per_second": train.get("train_samples_per_second"),
        "steps_per_second": train.get("train_steps_per_second"),
        "train_loss": train.get("train_loss"),
        "eval_loss": ev.get("eval_loss"),
        "peak_allocated_gb_rank0": (peak.get("allocated_gb") or [None])[0],
        "peak_reserved_gb_rank0": (peak.get("reserved_gb") or [None])[0],
        "peak_allocated_gb_all_ranks_max": max(peak.get("allocated_gb") or [None]),
        "peak_reserved_gb_all_ranks_max": max(peak.get("reserved_gb") or [None]),
        "host_rss_gb": result.get("host_rss_gb"),
        "gpu_names": "; ".join(g.get("name", "") for g in gpus),
        "torch": env.get("torch"),
        "cuda_runtime": env.get("cuda_runtime"),
        "git_sha": env.get("git_sha"),
        "config_hash": result.get("config_hash"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(flatten_result(json.load(f)))

    if not rows:
        raise SystemExit("No raw result JSON files found.")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)

    figdir = Path("figures")
    figdir.mkdir(exist_ok=True)

    plots = [
        ("examples_per_second", "Examples / second", "throughput.png"),
        ("train_loop_seconds", "Train-loop seconds", "train_time.png"),
        ("peak_allocated_gb_all_ranks_max", "Peak allocated GPU GB", "memory.png"),
    ]
    for metric, ylabel, filename in plots:
        if metric not in frame or frame[metric].isna().all():
            continue
        grouped = frame.groupby("method", as_index=False)[metric].mean()
        plt.figure(figsize=(7, 4))
        plt.bar(grouped["method"], grouped[metric])
        plt.ylabel(ylabel)
        plt.xlabel("Method")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=180)
        plt.close()


if __name__ == "__main__":
    main()

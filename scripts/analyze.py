import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/raw")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        raise SystemExit("No raw result JSON files found.")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)

    if "method" not in frame:
        raise SystemExit("Raw results do not contain a 'method' field.")

    figdir = Path("figures")
    figdir.mkdir(exist_ok=True)

    for metric, ylabel, filename in [
        ("examples_per_second", "Examples / second", "throughput.png"),
        ("train_loop_seconds", "Train-loop seconds", "train_time.png"),
        ("peak_allocated_gb", "Peak allocated GPU GB", "memory.png"),
    ]:
        if metric not in frame:
            continue
        plt.figure()
        plt.bar(frame["method"], frame[metric])
        plt.ylabel(ylabel)
        plt.xlabel("Method")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=180)
        plt.close()


if __name__ == "__main__":
    main()

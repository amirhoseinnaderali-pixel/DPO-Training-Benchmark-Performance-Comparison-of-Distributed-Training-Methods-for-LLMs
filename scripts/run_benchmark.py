import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def command_for(method, cfg):
    workers = str(cfg["world_size"])
    script = [sys.executable, "src/train_worker.py", "--method", method]

    if method == "ddp":
        return ["torchrun", "--standalone", f"--nproc_per_node={workers}", *script]
    if method == "fsdp":
        return ["torchrun", "--standalone", f"--nproc_per_node={workers}", *script]
    return ["deepspeed", f"--num_gpus={workers}", *script]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled.yaml")
    parser.add_argument("--method", action="append")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    methods = args.method or cfg["methods"]
    for method in methods:
        print(f"\n=== RUNNING {method.upper()} ===")
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        completed = subprocess.run(
            command_for(method, cfg),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"{method} failed with exit code {completed.returncode}.")


if __name__ == "__main__":
    main()

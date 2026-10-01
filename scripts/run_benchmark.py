import argparse
import os
import subprocess
import sys

import yaml


def command_for(method, cfg, config_path, run_id):
    workers = str(cfg["world_size"])
    script = [
        sys.executable,
        "src/train_worker.py",
        "--method",
        method,
        "--config",
        config_path,
        "--run-id",
        run_id,
    ]

    if method in {"ddp", "fsdp"}:
        return ["torchrun", "--standalone", f"--nproc_per_node={workers}", *script]
    return ["deepspeed", f"--num_gpus={workers}", *script]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled.yaml")
    parser.add_argument("--method", action="append")
    parser.add_argument("--run-id", default="run1")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    methods = args.method or cfg["methods"]
    for method in methods:
        print(f"\n=== RUNNING {method.upper()} ({args.run_id}) ===")
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        completed = subprocess.run(
            command_for(method, cfg, args.config, args.run_id),
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"{method} failed with exit code {completed.returncode}."
            )


if __name__ == "__main__":
    main()

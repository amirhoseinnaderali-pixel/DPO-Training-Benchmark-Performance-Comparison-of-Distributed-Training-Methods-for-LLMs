import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import psutil
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def config_hash(config):
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def environment_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_sha": git_sha(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            info["gpus"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
            })
    return info


def peak_memory():
    if not torch.cuda.is_available():
        return {"allocated_gb": [], "reserved_gb": []}
    torch.cuda.synchronize()
    return {
        "allocated_gb": [
            round(torch.cuda.max_memory_allocated(i) / 2**30, 4)
            for i in range(torch.cuda.device_count())
        ],
        "reserved_gb": [
            round(torch.cuda.max_memory_reserved(i) / 2**30, 4)
            for i in range(torch.cuda.device_count())
        ],
    }


def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def host_rss_gb():
    return round(psutil.Process().memory_info().rss / 2**30, 4)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class SynchronizedTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - self.start_time

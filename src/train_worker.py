import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

from benchmark_utils import (
    config_hash,
    environment_info,
    host_rss_gb,
    load_config,
    peak_memory,
    reset_memory,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["ddp", "fsdp", "zero2", "zero3"], required=True)
    p.add_argument("--config", default="configs/controlled.yaml")
    p.add_argument("--run-id", default="run1")
    return p.parse_args()


def prepare_dataset(cfg):
    raw = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])

    def convert(example):
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        if "Assistant:" not in chosen or "Assistant:" not in rejected:
            raise ValueError("Unexpected HH-RLHF example format.")
        pc, rc = chosen.rsplit("Assistant:", 1), rejected.rsplit("Assistant:", 1)
        return {
            "prompt": pc[0] + "Assistant:",
            "chosen": pc[1].lstrip(),
            "rejected": rc[1].lstrip(),
        }

    data = raw.map(convert, remove_columns=raw.column_names)
    split = data.train_test_split(
        test_size=1 - cfg["train_fraction"], seed=cfg["seed"]
    )
    return split["train"], split["test"]


def build_trainer(cfg, method, train_ds, eval_ds):
    args = dict(
        output_dir="results/generated",
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        beta=cfg["beta"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        fp16=cfg["fp16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )

    if method == "fsdp":
        args["fsdp"] = "full_shard auto_wrap"
        args["fsdp_config"] = {
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]
        }
    elif method in {"zero2", "zero3"}:
        args["deepspeed"] = f"configs/ds_{method}.json"

    dpo_args = DPOConfig(**args)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    common = dict(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft,
    )

    try:
        return DPOTrainer(**common, processing_class=tokenizer)
    except TypeError:
        return DPOTrainer(**common, tokenizer=tokenizer)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        reset_memory()

    train_ds, eval_ds = prepare_dataset(cfg)
    trainer = build_trainer(cfg, args.method, train_ds, eval_ds)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    train_output = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    eval_metrics = trainer.evaluate()

    result = {
        "method": args.method,
        "seed": cfg["seed"],
        "model": cfg["model_name"],
        "dataset": cfg["dataset_name"],
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "effective_batch_size": (
            cfg["per_device_train_batch_size"]
            * cfg["gradient_accumulation_steps"]
            * int(os.environ.get("WORLD_SIZE", "1"))
        ),
        "optimizer_steps": train_output.global_step,
        "learning_rate": cfg["learning_rate"],
        "train_loop_seconds": train_seconds,
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "peak_memory": peak_memory(),
        "host_rss_gb": host_rss_gb(),
        "environment": environment_info(),
        "config_hash": config_hash(cfg),
        "run_id": args.run_id,
    }

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        out = Path("results/raw") / f"{args.run_id}_{args.method}_seed{cfg['seed']}.json"
        save_json(out, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

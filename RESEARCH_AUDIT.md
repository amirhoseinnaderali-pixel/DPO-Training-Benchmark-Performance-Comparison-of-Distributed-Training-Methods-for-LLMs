# Research Audit — DPO Distributed-Training Benchmark

## Scope

This audit analyzes the original `Dpo.ipynb` and its saved outputs. No new training run is claimed here.

## Scientific verdict

The original repository is a working benchmark demo, but not yet a controlled experiment. Five configurations completed once. However, the runs differ in process count, effective batch/optimizer-step count, gradient checkpointing, and CPU offload. The headline wall-clock timing also diverges substantially from Trainer `train_runtime` for ZeRO-2.

The original README therefore overstates what the evidence establishes.

## Historical measurements

| Method | Steps | Trainer runtime (s) | Wall time (s) | Trainer samples/s | Table samples/s | Peak rank-0 allocated GB | Mean loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard | 112 | 336.37 | 336.79 | 2.676 | 2.67 | 2.48 | 0.69304 |
| FSDP | 56 | 513.45 | 517.47 | 1.753 | 1.74 | 2.91 | 0.69296 |
| ZeRO-2 | 56 | 201.06 | 242.60 | 4.476 | 3.71 | 3.62 | 0.69300 |
| ZeRO-2 Optimized | 56 | 200.49 | 208.76 | 4.489 | 4.31 | 3.62 | 0.69317 |
| ZeRO-3 | 56 | 1086.09 | 1097.17 | 0.829 | 0.82 | 4.20 | 0.69303 |

These are **historical single-run measurements**, not reproduced results.

## Main validity issues

1. Standard used one process with `device_map="auto"`; the other methods used two processes.
2. Standard used effective batch 8 / 112 optimizer steps; the others used effective batch 16 / 56 steps.
3. ZeRO-3 also changed micro-batch size, gradient checkpointing, and parameter CPU offload.
4. ZeRO-2 included CPU optimizer offload; this was not isolated as a factor.
5. Wall-clock timing and Trainer runtime disagree, especially for the first ZeRO-2 run.
6. Peak memory only covered rank 0 allocated GPU memory.
7. No fixed seed or persisted data split was stored.
8. No held-out DPO evaluation was performed.
9. The loss remained approximately ln(2), so the historical run provides little evidence about learning behavior.
10. Environment versions were not recoverable from the notebook.
11. The original notebook contains a destructive package-uninstall cell and a compatibility monkey-patch.
12. The original results were not persisted as raw machine-readable artifacts in the repository.

## Research decisions for the new study

### Primary question

> Under the same DPO workload, how does the distributed training strategy affect throughput, wall-clock training time, peak memory, and DPO training behavior on a fixed 2-GPU system?

### Primary comparison

The new primary comparison will use the same world size (2 processes) and the same effective batch, optimization steps, model, data split, DPO hyperparameters, precision, and LoRA configuration.

Primary methods:

- DDP baseline
- FSDP
- DeepSpeed ZeRO-2
- DeepSpeed ZeRO-3

Offload configurations are treated as explicit variants rather than silently bundled into the strategy.

The historical single-process Standard run remains historical evidence and is not mixed into the new primary ranking.

### Training regime

The historical learning rate/step budget is retained for the first reproducibility pass rather than silently changing the scientific setup. If learning is still effectively flat, a **separate explicitly labelled training-regime experiment** will test a larger optimization budget.

### Metrics

Every new run should record:

- train-loop time
- end-to-end time
- examples/sec
- optimizer steps/sec
- peak allocated GPU memory per rank
- peak reserved GPU memory per rank
- host RSS where offload is enabled
- final train loss
- DPO reward accuracy
- DPO reward margin
- environment metadata
- git SHA
- configuration hash
- seed

## Evidence policy

Historical numbers are labelled historical. New numbers are labelled reproduced only after an actual run. Explanations of performance differences are hypotheses unless directly measured.

## Next step

Run the new controlled benchmark on the original dual-T4 environment. No new result should be written into the README before the run produces raw JSON.

# Reproducibility

## Current status

The original environment is **not fully reproducible from the repository**. The notebook does not contain a complete dependency lockfile and its first cell removes packages before installing only NumPy.

The new benchmark is designed to record the environment automatically.

## Target environment

The historical run reports:

- Python 3.12.x
- CUDA toolkit 12.5
- NCCL 2.22.3
- dual NVIDIA T4 GPUs
- TinyLlama 1.1B
- Anthropic HH-RLHF

Some library versions are not recoverable from the saved notebook and are therefore intentionally not guessed here.

## Reproduction protocol

1. Create a clean environment.
2. Install dependencies from the project's dependency file.
3. Verify exactly two CUDA devices are visible.
4. Run the smoke test.
5. Run the controlled benchmark.
6. Preserve every raw JSON result.
7. Generate the summary and figures from raw results.

The benchmark records package versions, GPU information, CUDA information, git SHA, seed, and configuration hash with every run.

## Important

Do not use the historical notebook's package-uninstall cell as the environment setup procedure.

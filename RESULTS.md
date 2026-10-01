# Results

No new controlled benchmark has been executed yet.

The repository contains historical single-run measurements from the original notebook. Those measurements are preserved in the audit and must not be presented as reproduced results.

## Required result schema

Each run should produce JSON containing:

- method
- seed
- model
- dataset
- train_examples
- eval_examples
- world_size
- effective_batch_size
- optimizer_steps
- learning_rate
- precision
- train_loop_seconds
- end_to_end_seconds
- examples_per_second
- steps_per_second
- peak_allocated_gb_by_rank
- peak_reserved_gb_by_rank
- host_rss_gb
- final_train_loss
- final_reward_accuracy
- final_reward_margin
- environment
- git_sha
- config_hash

Populate this file only from actual benchmark output.

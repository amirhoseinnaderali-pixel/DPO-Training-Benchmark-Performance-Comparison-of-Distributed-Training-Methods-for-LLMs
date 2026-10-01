from src.benchmark_utils import config_hash, load_config


def test_controlled_config():
    cfg = load_config("configs/controlled.yaml")
    assert cfg["world_size"] == 2
    assert cfg["seed"] == 42
    assert cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"] == 8
    assert len(config_hash(cfg)) == 16

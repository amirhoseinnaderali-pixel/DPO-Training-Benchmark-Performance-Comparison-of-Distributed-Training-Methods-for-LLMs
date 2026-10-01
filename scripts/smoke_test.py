from src.benchmark_utils import config_hash, environment_info, load_config


def main():
    cfg = load_config("configs/controlled.yaml")
    print("Config hash:", config_hash(cfg))
    print("Environment:", environment_info())
    print("Smoke test passed: configuration and measurement utilities import successfully.")


if __name__ == "__main__":
    main()

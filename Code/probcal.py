import sys
from run import run_experiment


def main():

    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python main.py <config.toml>")

    run_experiment(sys.argv[1])


if __name__ == "__main__":
    main()
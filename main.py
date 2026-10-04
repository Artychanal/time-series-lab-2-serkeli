from pathlib import Path

from analysis import run_analysis


def main() -> None:
    root = Path(__file__).resolve().parent
    run_analysis(root / "data", root / "results")


if __name__ == "__main__":
    main()

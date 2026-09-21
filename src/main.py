import sys

from unsupervised_pipeline import main as run_unsupervised


def main() -> int:
    return run_unsupervised()


if __name__ == "__main__":
    sys.exit(main())

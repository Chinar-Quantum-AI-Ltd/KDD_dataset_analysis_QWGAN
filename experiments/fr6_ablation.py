"""Minimal FR-6 ablation orchestration script (dry-run).

This is a placeholder runner that shows how to run multiple seeds and arms. It uses the example implementation in fr6_example.
"""
from experiments.fr6_example import run_example


def main():
    for seed in [42, 123, 2026]:
        run_example(seed)


if __name__ == '__main__':
    main()

import subprocess
import sys
from pathlib import Path


IMAGE = Path("images/baboon.jpg")
OUTPUT_DIR = Path("outputs/demo")
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
MANIFEST = OUTPUT_DIR / "puzzle.manifest.json"

SEED = "42"
SIZE = "64"
GENERATIONS = "20"
POPULATION = "100"
MUTATION_RATE = "0.05"
SNAPSHOT_INTERVAL = "5"


def run_step(title, command):
    print(f"\n=== {title}")
    subprocess.run(command, check=True)


def main():
    if not IMAGE.exists():
        raise SystemExit("Run this script from the repository root.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cli = [sys.executable, "-m", "gaps.cli"]

    puzzle = OUTPUT_DIR / "puzzle.jpg"
    baseline = OUTPUT_DIR / "baseline.jpg"
    solution = OUTPUT_DIR / "solution.jpg"
    history = OUTPUT_DIR / "history.csv"
    fitness_plot = OUTPUT_DIR / "fitness.png"
    comparison = OUTPUT_DIR / "comparison.jpg"

    run_step(
        "Create puzzle",
        cli
        + [
            "create",
            str(IMAGE),
            str(puzzle),
            "--size",
            SIZE,
            "--seed",
            SEED,
            "--manifest",
            str(MANIFEST),
        ],
    )

    run_step(
        "Create random baseline",
        cli
        + [
            "baseline",
            str(puzzle),
            str(baseline),
            "--size",
            SIZE,
            "--seed",
            SEED,
        ],
    )

    run_step(
        "Run GA solver",
        cli
        + [
            "run",
            str(puzzle),
            str(solution),
            "--size",
            SIZE,
            "--generations",
            GENERATIONS,
            "--population",
            POPULATION,
            "--seed",
            SEED,
            "--mutation-rate",
            MUTATION_RATE,
            "--history",
            str(history),
            "--fitness-plot",
            str(fitness_plot),
            "--original",
            str(IMAGE),
            "--manifest",
            str(MANIFEST),
            "--comparison",
            str(comparison),
            "--snapshots-dir",
            str(SNAPSHOTS_DIR),
            "--snapshot-interval",
            SNAPSHOT_INTERVAL,
        ],
    )

    artifacts = [
        puzzle,
        MANIFEST,
        baseline,
        solution,
        history,
        fitness_plot,
        comparison,
    ]
    artifacts.extend(sorted(SNAPSHOTS_DIR.glob("generation_*.jpg")))

    print("\n=== Generated artifacts")
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    main()

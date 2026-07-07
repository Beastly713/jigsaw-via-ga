import csv
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path


IMAGE = Path("images/baboon.jpg")
OUTPUT_DIR = Path("outputs/experiments")
PUZZLE = OUTPUT_DIR / "puzzle.jpg"
MANIFEST = OUTPUT_DIR / "puzzle.manifest.json"
RESULTS = OUTPUT_DIR / "results.csv"

SEED = 42
SIZE = 64
GENERATIONS = [10, 20]
POPULATIONS = [50]
MUTATION_RATES = [0.0, 0.05, 0.1]

FIELDNAMES = [
    "run_id",
    "seed",
    "generations",
    "population",
    "mutation_rate",
    "generations_completed",
    "best_fitness",
    "termination_reason",
    "runtime_seconds",
    "piece_position_accuracy",
    "adjacency_accuracy",
    "manifest_path",
    "history_path",
    "solution_path",
]


def run_command(command, capture_output=False):
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )


def summary_value(stdout, label):
    pattern = rf"^\s*{re.escape(label)}:\s*(.+?)\s*$"
    match = re.search(pattern, stdout, flags=re.MULTILINE)
    if match is None:
        return ""

    return match.group(1)


def percent_value(stdout, label):
    value = summary_value(stdout, label)
    if value.endswith("%"):
        return value[:-1]

    return value


def main():
    if not IMAGE.exists():
        raise SystemExit("Run this script from the repository root.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cli = [sys.executable, "-m", "gaps.cli"]

    print("Creating base puzzle", flush=True)
    run_command(
        cli
        + [
            "create",
            str(IMAGE),
            str(PUZZLE),
            "--size",
            str(SIZE),
            "--seed",
            str(SEED),
            "--manifest",
            str(MANIFEST),
        ]
    )

    rows = []
    grid = itertools.product(GENERATIONS, POPULATIONS, MUTATION_RATES)
    for run_id, (generations, population, mutation_rate) in enumerate(grid, start=1):
        history_path = OUTPUT_DIR / f"run_{run_id}_history.csv"
        solution_path = OUTPUT_DIR / f"run_{run_id}_solution.jpg"

        print(
            "Running experiment "
            f"{run_id}: generations={generations}, "
            f"population={population}, mutation_rate={mutation_rate}",
            flush=True,
        )
        command = cli + [
            "run",
            str(PUZZLE),
            str(solution_path),
            "--size",
            str(SIZE),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--mutation-rate",
            str(mutation_rate),
            "--seed",
            str(SEED),
            "--history",
            str(history_path),
            "--original",
            str(IMAGE),
            "--manifest",
            str(MANIFEST),
        ]

        start = time.perf_counter()
        result = run_command(command, capture_output=True)
        runtime_seconds = time.perf_counter() - start

        rows.append(
            {
                "run_id": run_id,
                "seed": SEED,
                "generations": generations,
                "population": population,
                "mutation_rate": mutation_rate,
                "generations_completed": summary_value(
                    result.stdout, "Generations completed"
                ),
                "best_fitness": summary_value(result.stdout, "Best fitness"),
                "termination_reason": summary_value(
                    result.stdout, "Termination reason"
                ),
                "runtime_seconds": f"{runtime_seconds:.2f}",
                "piece_position_accuracy": percent_value(
                    result.stdout, "Piece-position accuracy"
                ),
                "adjacency_accuracy": percent_value(
                    result.stdout, "Adjacency accuracy"
                ),
                "manifest_path": MANIFEST,
                "history_path": history_path,
                "solution_path": solution_path,
            }
        )

    with RESULTS.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()

import argparse
import shutil
from pathlib import Path


OUTPUT_DIRECTORIES = [
    Path("outputs"),
]

ROOT_GENERATED_FILES = [
    # Common manual/smoke-test outputs created during development
    Path("puzzle.jpg"),
    Path("solution.jpg"),
    Path("baseline.jpg"),
    Path("puzzle.manifest.json"),
    Path("demo_puzzle.manifest.json"),
    Path("baseline.manifest.json"),
    Path("solution.manifest.json"),

    Path("puzzle_plot.jpg"),
    Path("solution_plot.jpg"),

    Path("puzzle_fitness_plot.jpg"),
    Path("solution_fitness_plot.jpg"),

    Path("puzzle_baseline.jpg"),

    Path("puzzle_metrics.jpg"),
    Path("solution_metrics.jpg"),

    Path("puzzle_comparison.jpg"),
    Path("solution_comparison.jpg"),

    Path("puzzle_snapshots.jpg"),
    Path("solution_snapshots.jpg"),
]


def remove_path(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False

    if dry_run:
        print(f"[dry-run] would remove {path}")
        return True

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    print(f"removed {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove generated demo, experiment, and smoke-test artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    args = parser.parse_args()

    if not Path("gaps").is_dir() or not Path("scripts").is_dir():
        raise SystemExit("Run this script from the repository root.")

    removed_anything = False

    for directory in OUTPUT_DIRECTORIES:
        removed_anything = remove_path(directory, args.dry_run) or removed_anything

    for file_path in ROOT_GENERATED_FILES:
        removed_anything = remove_path(file_path, args.dry_run) or removed_anything

    if not removed_anything:
        print("No generated artifacts found.")


if __name__ == "__main__":
    main()

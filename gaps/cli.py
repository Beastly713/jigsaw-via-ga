import csv
import json
import random
import time
from pathlib import Path
from typing import Optional

import click
import cv2 as cv
import numpy as np

from gaps import utils
from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.size_detector import SizeDetector

DEFAULT_GENERATIONS: int = 20
DEFAULT_POPULATION: int = 200

MIN_PIECE_SIZE: int = 32
MAX_PIECE_SIZE: int = 128


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
    }
)
def cli() -> None:
    """Solve or create puzzles with square pieces."""


def _validate_piece_size(_context: click.Context, _param: str, value: int) -> int:
    if value < MIN_PIECE_SIZE:
        raise click.BadParameter(f"Minimum piece size is {MIN_PIECE_SIZE} pixels")

    if value > MAX_PIECE_SIZE:
        raise click.BadParameter(f"Maximum piece size is {MAX_PIECE_SIZE} pixels")

    return value


def _validate_positive_integer(_context: click.Context, _param: str, value: int) -> int:
    if value <= 0:
        raise click.BadParameter("Should be a positive integer.")

    return value


def _validate_mutation_rate(_context: click.Context, _param: str, value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise click.BadParameter("Should be between 0.0 and 1.0.")

    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_image(path: str) -> np.ndarray:
    image = cv.imread(path)
    if image is None:
        raise click.ClickException(f"Could not read image file: {path}")

    return image


def _write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv.imwrite(str(output_path), image):
        raise click.ClickException(f"Could not write image file: {output_path}")


def _default_manifest_path(puzzle_path: str | Path) -> Path:
    path = Path(puzzle_path)
    return path.with_suffix(".manifest.json")


def _write_manifest(path: str | Path, manifest: dict) -> None:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


def _read_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except OSError as exc:
        raise click.ClickException(
            f"Could not read puzzle manifest: {manifest_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Invalid puzzle manifest JSON: {manifest_path}"
        ) from exc

    if not isinstance(manifest, dict):
        raise click.ClickException(f"Invalid puzzle manifest format: {manifest_path}")

    return manifest


def _validate_manifest(
    manifest: dict, piece_size: int, rows: int, columns: int
) -> dict:
    required_fields = {
        "version",
        "piece_size",
        "rows",
        "columns",
        "puzzle_to_original",
    }
    missing_fields = sorted(required_fields - set(manifest))
    if missing_fields:
        raise click.ClickException(
            "Puzzle manifest is missing required fields: "
            + ", ".join(missing_fields)
        )

    if manifest["piece_size"] != piece_size:
        raise click.ClickException(
            "Puzzle manifest piece size does not match requested piece size"
        )

    if manifest["rows"] != rows or manifest["columns"] != columns:
        raise click.ClickException(
            "Puzzle manifest grid does not match puzzle image dimensions"
        )

    mapping = manifest["puzzle_to_original"]
    if not isinstance(mapping, list):
        raise click.ClickException("Puzzle manifest mapping must be a list")

    total_pieces = rows * columns
    if len(mapping) != total_pieces:
        raise click.ClickException(
            "Puzzle manifest mapping length does not match puzzle grid"
        )

    try:
        normalized_mapping = [int(value) for value in mapping]
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            "Puzzle manifest mapping must contain integer piece ids"
        ) from exc

    if sorted(normalized_mapping) != list(range(total_pieces)):
        raise click.ClickException(
            "Puzzle manifest mapping must be a permutation of puzzle piece ids"
        )

    manifest = manifest.copy()
    manifest["puzzle_to_original"] = normalized_mapping
    return manifest


def _crop_image_to_piece_grid(
    image: np.ndarray, piece_size: int
) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]

    cropped_width = (width // piece_size) * piece_size
    cropped_height = (height // piece_size) * piece_size

    if cropped_width == 0 or cropped_height == 0:
        raise click.ClickException(
            f"Piece size {piece_size} is too large for image dimensions "
            f"{width}x{height}"
        )

    if cropped_width == width and cropped_height == height:
        return image, False

    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2

    cropped_image = image[top : top + cropped_height, left : left + cropped_width]
    return cropped_image.copy(), True


def _validate_image_dimensions(image: np.ndarray, piece_size: int) -> None:
    height, width = image.shape[:2]
    if height % piece_size != 0 or width % piece_size != 0:
        raise click.ClickException(
            f"Piece size {piece_size} does not evenly divide "
            f"image dimensions {width}x{height}"
        )


def _format_fitness(fitness: Optional[float]) -> str:
    if fitness is None:
        return "N/A"

    return f"{fitness:.6f}"


def _write_fitness_history(path: str, history) -> None:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["generation", "best_fitness", "average_fitness", "worst_fitness"]
    with output_path.open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_fitness_plot(path: str, history) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    generations = [row["generation"] for row in history]
    best = [row["best_fitness"] for row in history]
    average = [row["average_fitness"] for row in history]

    plt.figure()
    plt.plot(generations, best, label="Best fitness")
    plt.plot(generations, average, label="Average fitness")
    plt.title("GA Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _compute_solution_metrics_from_manifest(individual, manifest: dict) -> dict:
    rows = manifest["rows"]
    columns = manifest["columns"]
    puzzle_to_original = manifest["puzzle_to_original"]

    total_pieces = rows * columns
    if len(individual.pieces) != total_pieces:
        raise click.ClickException(
            "Solved individual size does not match puzzle manifest"
        )

    solved_original_ids = [puzzle_to_original[piece.id] for piece in individual.pieces]

    correct_positions = sum(
        original_id == position
        for position, original_id in enumerate(solved_original_ids)
    )

    correct_adjacencies = 0
    total_adjacencies = rows * (columns - 1) + (rows - 1) * columns

    for row in range(rows):
        for column in range(columns - 1):
            left = solved_original_ids[row * columns + column]
            right = solved_original_ids[row * columns + column + 1]

            if left % columns != columns - 1 and right == left + 1:
                correct_adjacencies += 1

    for row in range(rows - 1):
        for column in range(columns):
            top = solved_original_ids[row * columns + column]
            bottom = solved_original_ids[(row + 1) * columns + column]

            if top < total_pieces - columns and bottom == top + columns:
                correct_adjacencies += 1

    return {
        "metric_method": "manifest",
        "piece_position_accuracy": correct_positions / total_pieces,
        "adjacency_accuracy": (
            correct_adjacencies / total_adjacencies
            if total_adjacencies > 0
            else 0.0
        ),
        "correct_positions": correct_positions,
        "total_pieces": total_pieces,
        "correct_adjacencies": correct_adjacencies,
        "total_adjacencies": total_adjacencies,
    }


def _write_comparison_image(path, images):
    target_height, target_width = images[0].shape[:2]
    normalized_images = []
    for image in images:
        if image.shape[:2] != (target_height, target_width):
            image = cv.resize(image, (target_width, target_height))
        normalized_images.append(image)

    comparison_image = np.hstack(normalized_images)
    _write_image(path, comparison_image)


def _make_snapshot_callback(snapshots_dir, snapshot_interval):
    output_dir = Path(snapshots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_callback(generation, fittest):
        if generation % snapshot_interval != 0:
            return

        output_path = output_dir / f"generation_{generation:04d}.jpg"
        _write_image(output_path, fittest.to_image())

    return snapshot_callback


@click.command()
@click.argument("puzzle", type=click.Path(exists=True, readable=True))
@click.argument("solution", type=click.Path(dir_okay=False, writable=True))
@click.option(
    "-s",
    "--size",
    type=int,
    callback=_validate_piece_size,
    help="Size of single square puzzle piece in pixels. Autodetected if not specified.",
)
@click.option(
    "-g",
    "--generations",
    type=int,
    show_default=True,
    default=DEFAULT_GENERATIONS,
    callback=_validate_positive_integer,
    help="The number of generations for genetic algorithm.",
)
@click.option(
    "-p",
    "--population",
    type=int,
    show_default=True,
    default=DEFAULT_POPULATION,
    callback=_validate_positive_integer,
    help="The size of the initial population for genetic algorithm.",
)
@click.option(
    "-d",
    "--debug",
    type=bool,
    is_flag=True,
    default=False,
    help="If enabled, shows the best individual after each generation.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed for deterministic puzzle solving.",
)
@click.option(
    "--mutation-rate",
    type=float,
    show_default=True,
    default=0.0,
    callback=_validate_mutation_rate,
    help="Probability of applying swap mutation to each child.",
)
@click.option(
    "--history",
    type=click.Path(dir_okay=False, writable=True),
    help="Write GA fitness history to a CSV file.",
)
@click.option(
    "--fitness-plot",
    type=click.Path(dir_okay=False, writable=True),
    help="Write GA fitness history plot to an image file.",
)
@click.option(
    "--original",
    type=click.Path(exists=True, readable=True),
    help="Original image used to compute solution-quality metrics.",
)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, exists=True, readable=True),
    help=(
        "Puzzle manifest JSON created by gaps create. Defaults to "
        "puzzle.manifest.json next to the puzzle if present."
    ),
)
@click.option(
    "--comparison",
    type=click.Path(dir_okay=False, writable=True),
    help="Write side-by-side comparison image to a file.",
)
@click.option(
    "--snapshots-dir",
    type=click.Path(file_okay=False, writable=True),
    help="Directory for generation snapshot images.",
)
@click.option(
    "--snapshot-interval",
    type=int,
    show_default=True,
    default=1,
    callback=_validate_positive_integer,
    help="Save a generation snapshot every N completed generations.",
)
def run(
    puzzle: str,
    solution: str,
    size: int,
    generations: int,
    population: int,
    debug: bool,
    seed: int,
    mutation_rate: float,
    history: str,
    fitness_plot: str,
    original: str,
    manifest: str,
    comparison: str,
    snapshots_dir: str,
    snapshot_interval: int,
) -> None:
    """Run puzzle solver.

    \b
    PUZZLE is the input puzzle image with square pieces.
    SOLUTION is the output image file for solved puzzle.

    Examples:

    $ gaps run puzzle.jpg solution.jpg --size=32 --generations=100 --population=1000

    """

    if seed is not None:
        _set_seed(seed)

    input_puzzle = _read_image(puzzle)

    if size is None:
        detector = SizeDetector(input_puzzle)
        size = detector.detect()

    _validate_image_dimensions(input_puzzle, size)

    height, width = input_puzzle.shape[:2]
    puzzle_rows = height // size
    puzzle_columns = width // size
    pieces = puzzle_rows * puzzle_columns

    manifest_path = None
    manifest_data = None

    if manifest is not None:
        manifest_path = Path(manifest)
    else:
        default_manifest_path = _default_manifest_path(puzzle)
        if default_manifest_path.exists():
            manifest_path = default_manifest_path

    if manifest_path is not None:
        manifest_data = _validate_manifest(
            _read_manifest(manifest_path),
            size,
            puzzle_rows,
            puzzle_columns,
        )

    original_image = None
    if original is not None:
        original_image = _read_image(original)

        if original_image.shape != input_puzzle.shape:
            original_height, original_width = original_image.shape[:2]
            original_image, was_cropped = _crop_image_to_piece_grid(
                original_image, size
            )

            if was_cropped:
                cropped_height, cropped_width = original_image.shape[:2]
                click.echo(
                    "Cropped original image from "
                    f"{original_width}x{original_height} to "
                    f"{cropped_width}x{cropped_height} "
                    f"to match puzzle grid"
                )

        _validate_image_dimensions(original_image, size)

        if original_image.shape != input_puzzle.shape:
            raise click.ClickException(
                "Original image shape does not match puzzle image shape"
            )

    click.echo(f"Population: {population}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Piece size: {size}")
    click.echo(f"Mutation rate: {mutation_rate}")
    if seed is not None:
        click.echo(f"Seed: {seed}")

    snapshot_callback = None
    if snapshots_dir is not None:
        snapshot_callback = _make_snapshot_callback(snapshots_dir, snapshot_interval)

    ga = GeneticAlgorithm(
        image=input_puzzle,
        piece_size=size,
        population_size=population,
        generations=generations,
        mutation_rate=mutation_rate,
    )
    start_time = time.perf_counter()
    result = ga.start_evolution(debug, generation_callback=snapshot_callback)
    runtime = time.perf_counter() - start_time
    output_image = result.to_image()
    metrics = None
    if manifest_data is not None:
        metrics = _compute_solution_metrics_from_manifest(result, manifest_data)
    comparison_images = []
    if original_image is not None:
        comparison_images.append(original_image)
    comparison_images.append(input_puzzle)
    comparison_images.append(output_image)

    _write_image(solution, output_image)
    if history is not None:
        _write_fitness_history(history, ga.fitness_history)
    if fitness_plot is not None:
        _write_fitness_plot(fitness_plot, ga.fitness_history)
    if comparison is not None:
        _write_comparison_image(comparison, comparison_images)

    click.echo("\nPuzzle solved")
    click.echo("Summary:")
    click.echo(f"  Pieces: {pieces}")
    click.echo(f"  Piece size: {size}")
    click.echo(f"  Population size: {population}")
    click.echo(f"  Generations requested: {generations}")
    click.echo(f"  Generations completed: {ga.generations_completed}")
    click.echo(f"  Best fitness: {_format_fitness(ga.best_fitness)}")
    click.echo(f"  Termination reason: {ga.termination_reason}")
    click.echo(f"  Runtime: {runtime:.2f}s")
    click.echo(f"  Mutation rate: {mutation_rate}")
    if seed is not None:
        click.echo(f"  Seed: {seed}")
    click.echo(f"  Output: {solution}")
    if history is not None:
        click.echo(f"  History: {history}")
    if fitness_plot is not None:
        click.echo(f"  Fitness plot: {fitness_plot}")
    if comparison is not None:
        click.echo(f"  Comparison: {comparison}")
    if snapshots_dir is not None:
        click.echo(f"  Snapshots: {snapshots_dir}")
    if metrics is not None:
        click.echo(f"  Metric method: {metrics['metric_method']}")
        click.echo(
            "  Piece-position accuracy: "
            f"{metrics['piece_position_accuracy'] * 100:.2f}%"
        )
        click.echo(
            "  Correct positions: "
            f"{metrics['correct_positions']}/{metrics['total_pieces']}"
        )
        click.echo(
            f"  Adjacency accuracy: {metrics['adjacency_accuracy'] * 100:.2f}%"
        )
        click.echo(
            "  Correct adjacencies: "
            f"{metrics['correct_adjacencies']}/{metrics['total_adjacencies']}"
        )
        if manifest_path is not None:
            click.echo(f"  Manifest: {manifest_path}")
    else:
        click.echo("  Solution-quality metrics: unavailable (no puzzle manifest found)")


@click.command()
@click.argument("image", type=click.Path(exists=True, readable=True))
@click.argument("puzzle", type=click.Path(writable=True))
@click.option(
    "-s",
    "--size",
    type=int,
    show_default=True,
    default=MAX_PIECE_SIZE,
    callback=_validate_piece_size,
    help="Size of single square puzzle piece in pixels.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed for deterministic puzzle creation.",
)
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, writable=True),
    help=(
        "Write puzzle manifest JSON. Defaults to puzzle.manifest.json next "
        "to the puzzle."
    ),
)
def create(image: str, puzzle: str, size: int, seed: int, manifest: str) -> None:
    """Create jigsaw puzzle with square pieces.

    \b
    IMAGE is the input image file to create puzzle.
    PUZZLE is the output puzzle image with square pieces.

    Examples:

    $ gaps create image.jpg puzzle.jpg --size=32

    """

    if seed is not None:
        _set_seed(seed)

    input_image = _read_image(image)
    original_height, original_width = input_image.shape[:2]

    input_image, was_cropped = _crop_image_to_piece_grid(input_image, size)

    if was_cropped:
        cropped_height, cropped_width = input_image.shape[:2]
        click.echo(
            "Cropped image from "
            f"{original_width}x{original_height} to "
            f"{cropped_width}x{cropped_height} "
            f"to fit piece size {size}"
        )

    _validate_image_dimensions(input_image, size)
    pieces, rows, columns = utils.flatten_image(input_image, size)

    puzzle_to_original = np.arange(len(pieces))
    np.random.shuffle(puzzle_to_original)
    puzzle_to_original = puzzle_to_original.tolist()

    shuffled_pieces = [pieces[index] for index in puzzle_to_original]
    output_image = utils.assemble_image(shuffled_pieces, rows, columns)

    _write_image(puzzle, output_image)
    manifest_path = (
        Path(manifest) if manifest is not None else _default_manifest_path(puzzle)
    )
    source_height, source_width = input_image.shape[:2]

    _write_manifest(
        manifest_path,
        {
            "version": 1,
            "source_image": image,
            "puzzle_image": puzzle,
            "piece_size": size,
            "rows": rows,
            "columns": columns,
            "cropped_width": source_width,
            "cropped_height": source_height,
            "puzzle_to_original": puzzle_to_original,
        },
    )

    if seed is not None:
        click.echo(f"Seed: {seed}")
    click.echo(f"\nCreated puzzle with {len(pieces)} pieces")
    click.echo(f"Manifest: {manifest_path}")


@click.command()
@click.argument("puzzle", type=click.Path(exists=True, readable=True))
@click.argument("baseline", type=click.Path(dir_okay=False, writable=True))
@click.option(
    "-s",
    "--size",
    type=int,
    callback=_validate_piece_size,
    help="Size of single square puzzle piece in pixels. Autodetected if not specified.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed for deterministic baseline creation.",
)
def baseline(puzzle: str, baseline: str, size: int, seed: int) -> None:
    """Create random baseline puzzle arrangement.

    \b
    PUZZLE is the input puzzle image with square pieces.
    BASELINE is the output random baseline image.

    Examples:

    $ gaps baseline puzzle.jpg baseline.jpg --size=32

    """

    if seed is not None:
        _set_seed(seed)

    input_puzzle = _read_image(puzzle)

    if size is None:
        detector = SizeDetector(input_puzzle)
        size = detector.detect()

    _validate_image_dimensions(input_puzzle, size)
    pieces, rows, columns = utils.flatten_image(input_puzzle, size)

    np.random.shuffle(pieces)
    output_image = utils.assemble_image(pieces, rows, columns)

    _write_image(baseline, output_image)

    click.echo("\nRandom baseline created")
    click.echo("Summary:")
    click.echo(f"  Pieces: {len(pieces)}")
    click.echo(f"  Piece size: {size}")
    if seed is not None:
        click.echo(f"  Seed: {seed}")
    click.echo(f"  Output: {baseline}")


cli.add_command(run, name="run")
cli.add_command(create, name="create")
cli.add_command(baseline, name="baseline")

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter

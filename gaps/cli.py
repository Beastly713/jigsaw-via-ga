import csv
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


def _compute_solution_metrics(original_image, solved_image, piece_size):
    original_pieces, rows, columns = utils.flatten_image(original_image, piece_size)
    solved_pieces, _, _ = utils.flatten_image(solved_image, piece_size)

    correct_positions = sum(
        np.array_equal(original_piece, solved_piece)
        for original_piece, solved_piece in zip(original_pieces, solved_pieces)
    )
    total_pieces = len(original_pieces)

    original_piece_indexes = {
        piece.tobytes(): index for index, piece in enumerate(original_pieces)
    }
    solved_piece_indexes = [
        original_piece_indexes.get(piece.tobytes()) for piece in solved_pieces
    ]

    correct_adjacencies = 0
    total_adjacencies = rows * (columns - 1) + (rows - 1) * columns

    for row in range(rows):
        for column in range(columns - 1):
            left = solved_piece_indexes[row * columns + column]
            right = solved_piece_indexes[row * columns + column + 1]
            if left is not None and right is not None and right == left + 1:
                correct_adjacencies += 1

    for row in range(rows - 1):
        for column in range(columns):
            top = solved_piece_indexes[row * columns + column]
            bottom = solved_piece_indexes[(row + 1) * columns + column]
            if top is not None and bottom is not None and bottom == top + columns:
                correct_adjacencies += 1

    return {
        "piece_position_accuracy": correct_positions / total_pieces,
        "adjacency_accuracy": (
            correct_adjacencies / total_adjacencies
            if total_adjacencies > 0
            else 0.0
        ),
    }


def _write_comparison_image(path, images):
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    target_height, target_width = images[0].shape[:2]
    normalized_images = []
    for image in images:
        if image.shape[:2] != (target_height, target_width):
            image = cv.resize(image, (target_width, target_height))
        normalized_images.append(image)

    comparison_image = np.hstack(normalized_images)
    cv.imwrite(str(output_path), comparison_image)


def _make_snapshot_callback(snapshots_dir, snapshot_interval):
    output_dir = Path(snapshots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_callback(generation, fittest):
        if generation % snapshot_interval != 0:
            return

        output_path = output_dir / f"generation_{generation:04d}.jpg"
        cv.imwrite(str(output_path), fittest.to_image())

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
    original_image = None
    if original is not None:
        original_image = _read_image(original)
        _validate_image_dimensions(original_image, size)
        if original_image.shape != input_puzzle.shape:
            raise click.ClickException(
                "Original image shape does not match puzzle image shape"
            )

    height, width = input_puzzle.shape[:2]
    pieces = (height // size) * (width // size)

    click.echo(f"Population: {population}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Piece size: {size}")
    click.echo(f"Mutation rate: {mutation_rate}")
    if seed is not None:
        click.echo(f"Seed: {seed}")

    snapshot_callback = None
    if snapshots_dir is not None:
        snapshot_callback = _make_snapshot_callback(
            snapshots_dir, snapshot_interval
        )

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
    if original_image is not None:
        metrics = _compute_solution_metrics(original_image, output_image, size)
    comparison_images = []
    if original_image is not None:
        comparison_images.append(original_image)
    comparison_images.append(input_puzzle)
    comparison_images.append(output_image)

    cv.imwrite(solution, output_image)
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
        click.echo(
            "  Piece-position accuracy: "
            f"{metrics['piece_position_accuracy'] * 100:.2f}%"
        )
        click.echo(
            f"  Adjacency accuracy: {metrics['adjacency_accuracy'] * 100:.2f}%"
        )


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
def create(image: str, puzzle: str, size: int, seed: int) -> None:
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
    _validate_image_dimensions(input_image, size)
    pieces, rows, columns = utils.flatten_image(input_image, size)

    # Randomize pieces in order to make puzzle
    np.random.shuffle(pieces)

    # Create puzzle by stacking pieces
    output_image = utils.assemble_image(pieces, rows, columns)

    cv.imwrite(puzzle, output_image)

    if seed is not None:
        click.echo(f"Seed: {seed}")
    click.echo(f"\nCreated puzzle with {len(pieces)} pieces")


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

    cv.imwrite(baseline, output_image)

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

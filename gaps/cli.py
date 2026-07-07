import random

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
def run(
    puzzle: str,
    solution: str,
    size: int,
    generations: int,
    population: int,
    debug: bool,
    seed: int,
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

    click.echo(f"Population: {population}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Piece size: {size}")
    if seed is not None:
        click.echo(f"Seed: {seed}")

    ga = GeneticAlgorithm(
        image=input_puzzle,
        piece_size=size,
        population_size=population,
        generations=generations,
    )
    result = ga.start_evolution(debug)
    output_image = result.to_image()

    cv.imwrite(solution, output_image)

    click.echo("Puzzle solved")


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


cli.add_command(run, name="run")
cli.add_command(create, name="create")

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter

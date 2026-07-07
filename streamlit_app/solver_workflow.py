import random
import time
from contextlib import redirect_stdout
from io import BytesIO, StringIO

import cv2 as cv
import numpy as np
from PIL import Image

from gaps import utils
from gaps.cli import (
    _compute_solution_metrics_from_manifest,
    _crop_image_to_piece_grid,
)
from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.image_analysis import ImageAnalysis

MAX_PIECES = 150


def pil_to_bgr(uploaded_file) -> np.ndarray:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
    return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def reset_image_analysis_cache() -> None:
    ImageAnalysis.dissimilarity_measures.clear()
    ImageAnalysis.best_match_table.clear()


def crop_to_piece_grid(image: np.ndarray, piece_size: int) -> tuple[np.ndarray, bool]:
    return _crop_image_to_piece_grid(image, piece_size)


def create_puzzle_and_manifest(
    image: np.ndarray,
    piece_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cropped_image, _ = crop_to_piece_grid(image, piece_size)
    pieces, rows, columns = utils.flatten_image(cropped_image, piece_size)

    puzzle_to_original = np.arange(len(pieces))
    np.random.shuffle(puzzle_to_original)
    puzzle_to_original = puzzle_to_original.tolist()

    shuffled_pieces = [pieces[index] for index in puzzle_to_original]
    puzzle_image = utils.assemble_image(shuffled_pieces, rows, columns)

    manifest = {
        "version": 1,
        "piece_size": piece_size,
        "rows": rows,
        "columns": columns,
        "puzzle_to_original": puzzle_to_original,
    }

    return cropped_image, puzzle_image, manifest


def comparison_image(
    original: np.ndarray,
    puzzle: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    return np.hstack([original, puzzle, solution])


def image_to_png_bytes(image: np.ndarray) -> bytes:
    rgb_image = bgr_to_rgb(image)
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def format_fitness(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def run_solver_workflow(
    image: np.ndarray,
    piece_size: int,
    generations: int,
    population: int,
    mutation_rate: float,
    seed: int,
) -> dict:
    set_seed(seed)

    start_time = time.perf_counter()

    cropped_image, puzzle_image, manifest = create_puzzle_and_manifest(
        image,
        piece_size,
    )

    reset_image_analysis_cache()

    ga = GeneticAlgorithm(
        image=puzzle_image,
        piece_size=piece_size,
        population_size=population,
        generations=generations,
        mutation_rate=mutation_rate,
    )

    stdout_buffer = StringIO()
    with redirect_stdout(stdout_buffer):
        result = ga.start_evolution(verbose=False)

    runtime = time.perf_counter() - start_time
    solution_image = result.to_image()
    metrics = _compute_solution_metrics_from_manifest(result, manifest)
    combined_image = comparison_image(cropped_image, puzzle_image, solution_image)

    return {
        "cropped_image": cropped_image,
        "puzzle_image": puzzle_image,
        "solution_image": solution_image,
        "comparison_image": combined_image,
        "manifest": manifest,
        "metrics": metrics,
        "best_fitness": ga.best_fitness,
        "generations_completed": ga.generations_completed,
        "termination_reason": ga.termination_reason,
        "runtime": runtime,
        "stdout": stdout_buffer.getvalue(),
        "piece_size": piece_size,
        "population": population,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "seed": int(seed),
    }

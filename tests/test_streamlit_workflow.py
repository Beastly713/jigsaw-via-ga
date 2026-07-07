from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from streamlit_app.solver_workflow import (
    MAX_PIECES,
    bgr_to_rgb,
    comparison_image,
    create_puzzle_and_manifest,
    crop_to_piece_grid,
    format_fitness,
    image_to_png_bytes,
    pil_to_bgr,
    reset_image_analysis_cache,
    run_solver_workflow,
)
from gaps.image_analysis import ImageAnalysis


def _sample_bgr_image(width=64, height=64):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = [255, 0, 0]
    image[:, width // 2 :] = [0, 255, 0]
    return image


def test_create_puzzle_and_manifest_returns_valid_mapping():
    image = _sample_bgr_image(width=64, height=64)

    cropped, puzzle, manifest = create_puzzle_and_manifest(image, piece_size=32)

    assert cropped.shape == image.shape
    assert puzzle.shape == image.shape
    assert manifest["version"] == 1
    assert manifest["piece_size"] == 32
    assert manifest["rows"] == 2
    assert manifest["columns"] == 2
    assert sorted(manifest["puzzle_to_original"]) == [0, 1, 2, 3]


def test_comparison_image_stacks_three_images_horizontally():
    image = _sample_bgr_image(width=64, height=64)

    combined = comparison_image(image, image, image)

    assert combined.shape == (64, 192, 3)


def test_image_to_png_bytes_returns_png_data():
    image = _sample_bgr_image(width=64, height=64)

    png_bytes = image_to_png_bytes(image)

    assert png_bytes.startswith(b"\x89PNG")


def test_pil_to_bgr_and_bgr_to_rgb_round_trip_shapes():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[:, :] = [10, 20, 30]
    pil_image = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    bgr = pil_to_bgr(buffer)
    converted_rgb = bgr_to_rgb(bgr)

    assert bgr.shape == (32, 32, 3)
    assert converted_rgb.shape == (32, 32, 3)
    assert converted_rgb[0, 0].tolist() == [10, 20, 30]


def test_crop_to_piece_grid_crops_to_valid_grid():
    image = _sample_bgr_image(width=70, height=65)

    cropped, was_cropped = crop_to_piece_grid(image, piece_size=32)

    assert was_cropped is True
    assert cropped.shape == (64, 64, 3)


def test_reset_image_analysis_cache_clears_global_tables():
    ImageAnalysis.dissimilarity_measures[(1, 2)] = {"LR": 1.0}
    ImageAnalysis.best_match_table[1] = {"T": [], "R": [], "D": [], "L": []}

    reset_image_analysis_cache()

    assert ImageAnalysis.dissimilarity_measures == {}
    assert ImageAnalysis.best_match_table == {}


def test_format_fitness_handles_none_and_number():
    assert format_fitness(None) == "n/a"
    assert format_fitness(1.23456789) == "1.234568"


def test_run_solver_workflow_returns_result_dictionary():
    image = _sample_bgr_image(width=64, height=64)

    result = run_solver_workflow(
        image=image,
        piece_size=32,
        generations=2,
        population=20,
        mutation_rate=0.05,
        seed=42,
    )

    expected_keys = {
        "cropped_image",
        "puzzle_image",
        "solution_image",
        "comparison_image",
        "manifest",
        "metrics",
        "best_fitness",
        "generations_completed",
        "termination_reason",
        "runtime",
        "stdout",
        "piece_size",
        "population",
        "generations",
        "mutation_rate",
        "seed",
    }

    assert set(result) == expected_keys
    assert result["cropped_image"].shape == (64, 64, 3)
    assert result["puzzle_image"].shape == (64, 64, 3)
    assert result["solution_image"].shape == (64, 64, 3)
    assert result["comparison_image"].shape == (64, 192, 3)
    assert result["manifest"]["rows"] == 2
    assert result["manifest"]["columns"] == 2
    assert result["metrics"]["metric_method"] == "manifest"
    assert result["metrics"]["total_pieces"] == 4
    assert result["piece_size"] == 32
    assert result["population"] == 20
    assert result["generations"] == 2
    assert result["mutation_rate"] == pytest.approx(0.05)
    assert result["seed"] == 42
    assert result["generations_completed"] >= 1
    assert result["termination_reason"] in {"max_generations", "stagnation"}

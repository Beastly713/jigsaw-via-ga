from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from streamlit_app.solver_workflow import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION,
    FITNESS_HISTORY_FIELDS,
    HIGH_PIECE_COUNT_WARNING,
    MAX_PIECE_SIZE,
    MIN_PIECE_SIZE,
    bgr_to_rgb,
    collect_snapshot,
    comparison_image,
    create_puzzle_and_manifest,
    crop_to_piece_grid,
    estimate_run_score,
    fitness_history_chart_data,
    fitness_history_to_csv_bytes,
    format_fitness,
    image_to_png_bytes,
    make_snapshot_callback,
    pil_to_bgr,
    reset_image_analysis_cache,
    run_solver_workflow,
    should_warn_heavy_configuration,
    should_warn_piece_count,
    snapshot_caption,
    snapshot_filename,
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


def test_local_ui_constants_match_cli_defaults_and_piece_size_limits():
    assert MIN_PIECE_SIZE == 32
    assert MAX_PIECE_SIZE == 128
    assert DEFAULT_GENERATIONS == 20
    assert DEFAULT_POPULATION == 200


def test_warning_helpers_warn_without_blocking():
    assert should_warn_piece_count(HIGH_PIECE_COUNT_WARNING) is False
    assert should_warn_piece_count(HIGH_PIECE_COUNT_WARNING + 1) is True

    assert (
        estimate_run_score(total_pieces=100, generations=20, population=200)
        == 400_000
    )

    assert (
        should_warn_heavy_configuration(
            total_pieces=100,
            generations=20,
            population=200,
        )
        is False
    )

    assert (
        should_warn_heavy_configuration(
            total_pieces=201,
            generations=50,
            population=500,
        )
        is True
    )


def test_snapshot_callback_collects_at_interval():
    snapshots = []

    class DummyFittest:
        fitness = 1.25

        def to_image(self):
            return np.zeros((32, 32, 3), dtype=np.uint8)

    callback = make_snapshot_callback(
        snapshots=snapshots,
        snapshot_interval=2,
    )

    callback(1, DummyFittest())
    callback(2, DummyFittest())
    callback(3, DummyFittest())
    callback(4, DummyFittest())

    assert len(snapshots) == 2
    assert snapshots[0]["generation"] == 2
    assert snapshots[0]["fitness"] == 1.25
    assert snapshots[0]["image"].shape == (32, 32, 3)
    assert snapshots[1]["generation"] == 4
    assert snapshot_filename(snapshots[0]) == "generation_0002.png"
    assert snapshot_caption(snapshots[0]) == "Generation 2 | Fitness: 1.250000"

    collect_snapshot(snapshots, 6, DummyFittest())

    assert snapshots[2]["generation"] == 6
    assert snapshots[2]["fitness"] == 1.25


def test_snapshot_filename_and_caption_formatting():
    snapshot = {
        "generation": 7,
        "fitness": 1.23456789,
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
    }

    assert snapshot_filename(snapshot) == "generation_0007.png"
    assert snapshot_caption(snapshot) == "Generation 7 | Fitness: 1.234568"


def test_snapshot_caption_handles_missing_fitness():
    snapshot = {
        "generation": 3,
        "fitness": None,
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
    }

    assert snapshot_caption(snapshot) == "Generation 3 | Fitness: n/a"


def test_fitness_history_to_csv_bytes_includes_expected_columns():
    history = [
        {
            "generation": 1,
            "best_fitness": 10.0,
            "average_fitness": 5.0,
            "worst_fitness": 1.0,
        },
        {
            "generation": 2,
            "best_fitness": 12.0,
            "average_fitness": 7.5,
            "worst_fitness": 2.0,
        },
    ]

    csv_bytes = fitness_history_to_csv_bytes(history)
    csv_text = csv_bytes.decode("utf-8")

    assert csv_text.splitlines()[0] == ",".join(FITNESS_HISTORY_FIELDS)
    assert "1,10.0,5.0,1.0" in csv_text
    assert "2,12.0,7.5,2.0" in csv_text


def test_fitness_history_chart_data_returns_column_oriented_data():
    history = [
        {
            "generation": 1,
            "best_fitness": 10.0,
            "average_fitness": 5.0,
            "worst_fitness": 1.0,
        },
        {
            "generation": 2,
            "best_fitness": 12.0,
            "average_fitness": 7.5,
            "worst_fitness": 2.0,
        },
    ]

    chart_data = fitness_history_chart_data(history)

    assert chart_data == {
        "generation": [1, 2],
        "best_fitness": [10.0, 12.0],
        "average_fitness": [5.0, 7.5],
        "worst_fitness": [1.0, 2.0],
    }


def test_run_solver_workflow_rejects_invalid_snapshot_interval():
    image = _sample_bgr_image(width=64, height=64)

    with pytest.raises(ValueError, match="snapshot_interval"):
        run_solver_workflow(
            image=image,
            piece_size=32,
            generations=2,
            population=20,
            mutation_rate=0.05,
            seed=42,
            snapshot_interval=0,
        )


def test_run_solver_workflow_returns_result_dictionary():
    image = _sample_bgr_image(width=64, height=64)

    result = run_solver_workflow(
        image=image,
        piece_size=32,
        generations=2,
        population=20,
        mutation_rate=0.05,
        seed=42,
        snapshot_interval=1,
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
        "fitness_history",
        "snapshots",
        "snapshot_interval",
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
    assert result["snapshot_interval"] == 1
    assert result["generations_completed"] >= 1
    assert result["termination_reason"] in {"max_generations", "stagnation"}
    assert len(result["fitness_history"]) == result["generations_completed"]
    assert len(result["snapshots"]) == result["generations_completed"]
    assert result["fitness_history"][0]["generation"] == 1
    assert set(result["fitness_history"][0]) == {
        "generation",
        "best_fitness",
        "average_fitness",
        "worst_fitness",
    }
    assert result["snapshots"][0]["generation"] == 1
    assert result["snapshots"][0]["image"].shape == (64, 64, 3)
    csv_bytes = fitness_history_to_csv_bytes(result["fitness_history"])
    assert csv_bytes.startswith(
        b"generation,best_fitness,average_fitness,worst_fitness"
    )

import json
from pathlib import Path
from types import SimpleNamespace

import click
import cv2 as cv
import numpy as np
import pytest
from click.testing import CliRunner

from gaps.cli import (
    _compute_solution_metrics_from_manifest,
    _default_manifest_path,
    _validate_manifest,
    cli,
)


def _individual_with_piece_ids(piece_ids):
    return SimpleNamespace(
        pieces=[SimpleNamespace(id=piece_id) for piece_id in piece_ids]
    )


def test_default_manifest_path_uses_manifest_suffix():
    assert _default_manifest_path("puzzle.jpg") == Path("puzzle.manifest.json")
    assert _default_manifest_path("outputs/demo/puzzle.jpg") == Path(
        "outputs/demo/puzzle.manifest.json"
    )


def test_validate_manifest_accepts_valid_mapping_and_normalizes_ids():
    manifest = {
        "version": 1,
        "piece_size": 32,
        "rows": 2,
        "columns": 2,
        "puzzle_to_original": ["0", "1", "2", "3"],
    }

    validated = _validate_manifest(manifest, piece_size=32, rows=2, columns=2)

    assert validated["puzzle_to_original"] == [0, 1, 2, 3]


def test_validate_manifest_rejects_duplicate_mapping():
    manifest = {
        "version": 1,
        "piece_size": 32,
        "rows": 2,
        "columns": 2,
        "puzzle_to_original": [0, 1, 1, 3],
    }

    with pytest.raises(click.ClickException, match="permutation"):
        _validate_manifest(manifest, piece_size=32, rows=2, columns=2)


def test_manifest_metrics_use_mapping_not_raw_piece_ids():
    manifest = {
        "version": 1,
        "piece_size": 32,
        "rows": 2,
        "columns": 3,
        "puzzle_to_original": [2, 0, 1, 5, 3, 4],
    }

    # These puzzle-piece ids map back to original ids [0, 1, 2, 3, 4, 5].
    individual = _individual_with_piece_ids([1, 2, 0, 4, 5, 3])

    metrics = _compute_solution_metrics_from_manifest(individual, manifest)

    assert metrics["metric_method"] == "manifest"
    assert metrics["correct_positions"] == 6
    assert metrics["total_pieces"] == 6
    assert metrics["piece_position_accuracy"] == pytest.approx(1.0)
    assert metrics["correct_adjacencies"] == 7
    assert metrics["total_adjacencies"] == 7
    assert metrics["adjacency_accuracy"] == pytest.approx(1.0)


def test_shifted_solution_has_zero_direct_but_nonzero_neighbor_accuracy():
    rows = 3
    columns = 4
    total_pieces = rows * columns
    manifest = {
        "version": 1,
        "piece_size": 32,
        "rows": rows,
        "columns": columns,
        "puzzle_to_original": list(range(total_pieces)),
    }

    # Each row is shifted left by one position:
    # original rows would be:
    #   0  1  2  3
    #   4  5  6  7
    #   8  9 10 11
    #
    # solved rows are:
    #   1  2  3  0
    #   5  6  7  4
    #   9 10 11  8
    #
    # Absolute locations are all wrong, but many local neighbor relations remain correct.
    individual = _individual_with_piece_ids(
        [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8]
    )

    metrics = _compute_solution_metrics_from_manifest(individual, manifest)

    assert metrics["correct_positions"] == 0
    assert metrics["piece_position_accuracy"] == pytest.approx(0.0)
    assert metrics["correct_adjacencies"] == 14
    assert metrics["total_adjacencies"] == 17
    assert metrics["adjacency_accuracy"] == pytest.approx(14 / 17)


def test_create_writes_default_manifest():
    runner = CliRunner()

    with runner.isolated_filesystem():
        image = np.zeros((65, 70, 3), dtype=np.uint8)
        image[:, :35] = [255, 0, 0]
        image[:, 35:] = [0, 255, 0]

        assert cv.imwrite("source.jpg", image)

        result = runner.invoke(
            cli,
            [
                "create",
                "source.jpg",
                "puzzle.jpg",
                "--size",
                "32",
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0, result.output

        puzzle_path = Path("puzzle.jpg")
        manifest_path = Path("puzzle.manifest.json")

        assert puzzle_path.exists()
        assert manifest_path.exists()
        assert "Manifest: puzzle.manifest.json" in result.output

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["version"] == 1
        assert manifest["source_image"] == "source.jpg"
        assert manifest["puzzle_image"] == "puzzle.jpg"
        assert manifest["piece_size"] == 32
        assert manifest["rows"] == 2
        assert manifest["columns"] == 2
        assert manifest["cropped_width"] == 64
        assert manifest["cropped_height"] == 64
        assert sorted(manifest["puzzle_to_original"]) == [0, 1, 2, 3]

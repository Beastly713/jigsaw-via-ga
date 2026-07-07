import json
import random
import sys
import time
from contextlib import redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaps import utils
from gaps.cli import (
    _compute_solution_metrics_from_manifest,
    _crop_image_to_piece_grid,
)
from gaps.genetic_algorithm import GeneticAlgorithm
from gaps.image_analysis import ImageAnalysis

st.set_page_config(
    page_title="GA Jigsaw Puzzle Solver",
    page_icon="🧩",
    layout="wide",
)

MAX_PIECES = 150


def _pil_to_bgr(uploaded_file) -> np.ndarray:
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)
    return cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _reset_image_analysis_cache() -> None:
    ImageAnalysis.dissimilarity_measures.clear()
    ImageAnalysis.best_match_table.clear()


def _create_puzzle_and_manifest(
    image: np.ndarray,
    piece_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cropped_image, _ = _crop_image_to_piece_grid(image, piece_size)
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


def _comparison_image(
    original: np.ndarray,
    puzzle: np.ndarray,
    solution: np.ndarray,
) -> np.ndarray:
    return np.hstack([original, puzzle, solution])


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    rgb_image = _bgr_to_rgb(image)
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _format_fitness(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


st.title("🧩 Genetic Algorithm Jigsaw Puzzle Solver")

st.write(
    "Upload an image, turn it into a square-piece jigsaw puzzle, "
    "solve it with a Genetic Algorithm, and evaluate the result with "
    "manifest-based direct and neighbor comparison metrics."
)

st.sidebar.header("Solver Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
)

piece_size = st.sidebar.selectbox(
    "Piece size",
    options=[32, 64, 96, 128],
    index=1,
)

generations = st.sidebar.slider(
    "Generations",
    min_value=2,
    max_value=30,
    value=10,
    step=1,
)

population = st.sidebar.slider(
    "Population size",
    min_value=20,
    max_value=150,
    value=50,
    step=10,
)

mutation_rate = st.sidebar.slider(
    "Mutation rate",
    min_value=0.0,
    max_value=0.30,
    value=0.05,
    step=0.01,
)

seed = st.sidebar.number_input(
    "Seed",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

st.sidebar.caption(
    "Large images, small piece sizes, high generations, and high population "
    "sizes can be slow on free deployment resources."
)

run_button = st.sidebar.button("Run solver", type="primary")

if uploaded_file is None:
    st.info("Upload a JPG or PNG image from the sidebar to start.")
    st.stop()

try:
    input_image = _pil_to_bgr(uploaded_file)
    preview_cropped_image, _ = _crop_image_to_piece_grid(input_image, piece_size)
except Exception as exc:
    st.error(f"Could not prepare image: {exc}")
    st.stop()

original_height, original_width = input_image.shape[:2]
cropped_height, cropped_width = preview_cropped_image.shape[:2]
rows = cropped_height // piece_size
columns = cropped_width // piece_size
total_pieces = rows * columns

st.subheader("Input Preview")
st.write(
    f"Original size: {original_width}×{original_height} px | "
    f"Cropped size: {cropped_width}×{cropped_height} px | "
    f"Grid: {rows}×{columns} | Pieces: {total_pieces}"
)

st.image(
    _bgr_to_rgb(preview_cropped_image),
    caption="Cropped original image used for puzzle generation",
    use_container_width=True,
)

if total_pieces > MAX_PIECES:
    st.error(
        f"This setup creates {total_pieces} pieces, which is above the deployment "
        f"limit of {MAX_PIECES}. Increase the piece size or upload a smaller image."
    )
    st.stop()

if run_button:
    _set_seed(int(seed))

    with st.spinner("Creating puzzle and running Genetic Algorithm..."):
        start_time = time.perf_counter()

        cropped_image, puzzle_image, manifest = _create_puzzle_and_manifest(
            input_image,
            piece_size,
        )

        _reset_image_analysis_cache()

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
        comparison_image = _comparison_image(
            cropped_image,
            puzzle_image,
            solution_image,
        )

    st.session_state["solver_result"] = {
        "cropped_image": cropped_image,
        "puzzle_image": puzzle_image,
        "solution_image": solution_image,
        "comparison_image": comparison_image,
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

    st.success("Solver finished.")

if "solver_result" in st.session_state:
    data = st.session_state["solver_result"]
    metrics = data["metrics"]

    st.subheader("Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            _bgr_to_rgb(data["cropped_image"]),
            caption="Cropped Original",
            use_container_width=True,
        )

    with col2:
        st.image(
            _bgr_to_rgb(data["puzzle_image"]),
            caption="Shuffled Puzzle",
            use_container_width=True,
        )

    with col3:
        st.image(
            _bgr_to_rgb(data["solution_image"]),
            caption="GA Solution",
            use_container_width=True,
        )

    st.image(
        _bgr_to_rgb(data["comparison_image"]),
        caption="Original | Puzzle | Solution",
        use_container_width=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric(
        "Piece-position accuracy",
        f"{metrics['piece_position_accuracy'] * 100:.2f}%",
    )

    metric_col2.metric(
        "Adjacency accuracy",
        f"{metrics['adjacency_accuracy'] * 100:.2f}%",
    )

    metric_col3.metric(
        "Best fitness",
        _format_fitness(data["best_fitness"]),
    )

    metric_col4.metric(
        "Runtime",
        f"{data['runtime']:.2f}s",
    )

    st.write(
        f"Correct positions: {metrics['correct_positions']}/"
        f"{metrics['total_pieces']}"
    )
    st.write(
        f"Correct adjacencies: {metrics['correct_adjacencies']}/"
        f"{metrics['total_adjacencies']}"
    )
    st.write(f"Generations completed: {data['generations_completed']}")
    st.write(f"Termination reason: {data['termination_reason']}")
    st.write(f"Metric method: {metrics['metric_method']}")

    st.subheader("Downloads")

    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:
        st.download_button(
            "Download solution PNG",
            data=_image_to_png_bytes(data["solution_image"]),
            file_name="ga_solution.png",
            mime="image/png",
        )

    with download_col2:
        st.download_button(
            "Download comparison PNG",
            data=_image_to_png_bytes(data["comparison_image"]),
            file_name="ga_comparison.png",
            mime="image/png",
        )

    with download_col3:
        st.download_button(
            "Download manifest JSON",
            data=json.dumps(data["manifest"], indent=2).encode("utf-8"),
            file_name="puzzle_manifest.json",
            mime="application/json",
        )

    with st.expander("Run log"):
        st.text(data["stdout"] or "No log captured.")

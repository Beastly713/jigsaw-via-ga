import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from streamlit_app.solver_workflow import (
    DEFAULT_GENERATIONS,
    DEFAULT_POPULATION,
    HIGH_PIECE_COUNT_WARNING,
    MAX_PIECE_SIZE,
    MIN_PIECE_SIZE,
    bgr_to_rgb,
    crop_to_piece_grid,
    fitness_history_chart_data,
    fitness_history_to_csv_bytes,
    format_fitness,
    image_to_png_bytes,
    pil_to_bgr,
    estimate_run_score,
    should_warn_heavy_configuration,
    should_warn_piece_count,
    snapshot_caption,
    snapshot_filename,
    run_solver_workflow,
)

st.set_page_config(
    page_title="GA Jigsaw Puzzle Solver",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Genetic Algorithm Jigsaw Puzzle Solver")

st.write(
    "Local visualization dashboard for generating a shuffled jigsaw puzzle, "
    "solving it with a Genetic Algorithm, and inspecting solution quality, "
    "fitness evolution, and generation snapshots."
)

st.sidebar.header("Local Solver Controls")

st.sidebar.subheader("Image Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
)

st.sidebar.subheader("Puzzle Settings")
piece_size = st.sidebar.number_input(
    "Piece size",
    min_value=MIN_PIECE_SIZE,
    max_value=MAX_PIECE_SIZE,
    value=64,
    step=1,
    help=(
        "Size of each square puzzle piece in pixels. "
        "Matches the CLI limit of 32 to 128."
    ),
)

st.sidebar.subheader("GA Settings")
generations = st.sidebar.number_input(
    "Generations",
    min_value=1,
    value=DEFAULT_GENERATIONS,
    step=1,
    help=(
        "Number of GA evolution iterations. "
        "Higher values can improve results but increase runtime."
    ),
)

population = st.sidebar.number_input(
    "Population size",
    min_value=1,
    value=DEFAULT_POPULATION,
    step=1,
    help=(
        "Number of candidate solutions per generation. "
        "Higher values increase search diversity and runtime."
    ),
)

mutation_rate = st.sidebar.number_input(
    "Mutation rate",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.2f",
    help=(
        "Probability of applying swap mutation to each child. "
        "Matches the CLI range of 0.0 to 1.0."
    ),
)

seed = st.sidebar.number_input(
    "Seed",
    value=42,
    step=1,
    help="Integer seed for reproducible shuffling and GA randomness.",
)

st.sidebar.subheader("Snapshot Settings")
snapshot_interval = st.sidebar.number_input(
    "Snapshot interval",
    min_value=1,
    value=1,
    step=1,
    help=(
        "Capture one in-memory solution snapshot every N completed "
        "generations. This mirrors the CLI snapshot interval."
    ),
)

st.sidebar.caption(
    "Large images, small piece sizes, high generations, and high population "
    "sizes can take longer locally. The app no longer blocks heavy local runs."
)

st.sidebar.subheader("Run")

piece_size = int(piece_size)
generations = int(generations)
population = int(population)
seed = int(seed)
mutation_rate = float(mutation_rate)
snapshot_interval = int(snapshot_interval)

run_button = st.sidebar.button("Run solver", type="primary")

if uploaded_file is None:
    st.info("Upload a JPG or PNG image from the sidebar to start.")
    st.stop()

try:
    input_image = pil_to_bgr(uploaded_file)
    preview_cropped_image, _ = crop_to_piece_grid(input_image, piece_size)
except Exception as exc:
    st.error(f"Could not prepare image: {exc}")
    st.stop()

original_height, original_width = input_image.shape[:2]
cropped_height, cropped_width = preview_cropped_image.shape[:2]
rows = cropped_height // piece_size
columns = cropped_width // piece_size
total_pieces = rows * columns

st.subheader("Input Preview")

preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)

preview_col1.metric("Original size", f"{original_width}×{original_height}")
preview_col2.metric("Cropped size", f"{cropped_width}×{cropped_height}")
preview_col3.metric("Grid", f"{rows}×{columns}")
preview_col4.metric("Pieces", f"{total_pieces}")

st.image(
    bgr_to_rgb(preview_cropped_image),
    caption="Cropped original image used for puzzle generation",
    use_container_width=True,
)

if should_warn_piece_count(total_pieces):
    st.warning(
        f"This setup creates {total_pieces} pieces. "
        f"That is above the old cloud-safe warning threshold of "
        f"{HIGH_PIECE_COUNT_WARNING}, so the run may take longer locally."
    )

if should_warn_heavy_configuration(total_pieces, generations, population):
    run_score = estimate_run_score(total_pieces, generations, population)
    st.warning(
        "This is a heavy local configuration: "
        f"{total_pieces} pieces × {generations} generations × "
        f"{population} population = {run_score:,} estimated work score. "
        "It will still run, but it may take noticeably longer."
    )

if run_button:
    with st.spinner("Creating puzzle and running Genetic Algorithm..."):
        st.session_state["solver_result"] = run_solver_workflow(
            image=input_image,
            piece_size=piece_size,
            generations=generations,
            population=population,
            mutation_rate=mutation_rate,
            seed=seed,
            snapshot_interval=snapshot_interval,
        )

    st.success("Solver finished.")

if "solver_result" in st.session_state:
    data = st.session_state["solver_result"]
    metrics = data["metrics"]

    st.subheader("Run Summary")

    summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(
        5
    )

    summary_col1.metric(
        "Piece accuracy",
        f"{metrics['piece_position_accuracy'] * 100:.2f}%",
    )

    summary_col2.metric(
        "Adjacency accuracy",
        f"{metrics['adjacency_accuracy'] * 100:.2f}%",
    )

    summary_col3.metric(
        "Best fitness",
        format_fitness(data["best_fitness"]),
    )

    summary_col4.metric(
        "Runtime",
        f"{data['runtime']:.2f}s",
    )

    summary_col5.metric(
        "Generations",
        f"{data['generations_completed']}/{data['generations']}",
    )

    (
        overview_tab,
        images_tab,
        metrics_tab,
        history_tab,
        snapshots_tab,
        downloads_tab,
    ) = st.tabs(
        [
            "Overview",
            "Images",
            "Metrics",
            "Fitness History",
            "Snapshots",
            "Downloads & Log",
        ]
    )

    with overview_tab:
        st.subheader("Configuration")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write(f"Piece size: {data['piece_size']}")
            st.write(f"Population: {data['population']}")
            st.write(f"Generations requested: {data['generations']}")
            st.write(f"Generations completed: {data['generations_completed']}")

        with config_col2:
            st.write(f"Mutation rate: {data['mutation_rate']}")
            st.write(f"Seed: {data['seed']}")
            st.write(f"Snapshot interval: {data['snapshot_interval']}")
            st.write(f"Termination reason: {data['termination_reason']}")

        st.subheader("Artifacts Captured")
        artifact_col1, artifact_col2, artifact_col3 = st.columns(3)
        artifact_col1.metric("Fitness rows", len(data["fitness_history"]))
        artifact_col2.metric("Snapshots", len(data["snapshots"]))
        artifact_col3.metric("Metric method", metrics["metric_method"])

    with images_tab:
        st.subheader("Puzzle Images")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                bgr_to_rgb(data["cropped_image"]),
                caption="Cropped Original",
                use_container_width=True,
            )

        with col2:
            st.image(
                bgr_to_rgb(data["puzzle_image"]),
                caption="Shuffled Puzzle",
                use_container_width=True,
            )

        with col3:
            st.image(
                bgr_to_rgb(data["solution_image"]),
                caption="GA Solution",
                use_container_width=True,
            )

        st.image(
            bgr_to_rgb(data["comparison_image"]),
            caption="Original | Puzzle | Solution",
            use_container_width=True,
        )

    with metrics_tab:
        st.subheader("Solution Quality Metrics")

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
            format_fitness(data["best_fitness"]),
        )

        metric_col4.metric(
            "Runtime",
            f"{data['runtime']:.2f}s",
        )

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write(
                f"Correct positions: {metrics['correct_positions']}/"
                f"{metrics['total_pieces']}"
            )
            st.write(
                f"Correct adjacencies: {metrics['correct_adjacencies']}/"
                f"{metrics['total_adjacencies']}"
            )

        with detail_col2:
            st.write(f"Generations completed: {data['generations_completed']}")
            st.write(f"Termination reason: {data['termination_reason']}")
            st.write(f"Metric method: {metrics['metric_method']}")
            st.write(f"Fitness history rows: {len(data['fitness_history'])}")
            st.write(f"Snapshots captured: {len(data['snapshots'])}")

    with history_tab:
        st.subheader("Fitness History")

        history = data["fitness_history"]

        if history:
            st.line_chart(
                fitness_history_chart_data(history),
                x="generation",
                y=["best_fitness", "average_fitness", "worst_fitness"],
            )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No fitness history was recorded for this run.")

    with snapshots_tab:
        st.subheader("Generation Snapshots")

        snapshots = data["snapshots"]

        if snapshots:
            st.caption(
                f"Showing {len(snapshots)} in-memory snapshots captured every "
                f"{data['snapshot_interval']} completed generation(s)."
            )

            snapshots_per_row = 3
            for start in range(0, len(snapshots), snapshots_per_row):
                row_snapshots = snapshots[start : start + snapshots_per_row]
                columns = st.columns(snapshots_per_row)

                for column, snapshot in zip(columns, row_snapshots):
                    with column:
                        st.image(
                            bgr_to_rgb(snapshot["image"]),
                            caption=snapshot_caption(snapshot),
                            use_container_width=True,
                        )
                        st.download_button(
                            "Download snapshot",
                            data=image_to_png_bytes(snapshot["image"]),
                            file_name=snapshot_filename(snapshot),
                            mime="image/png",
                            key=f"snapshot_download_{start}_{snapshot['generation']}",
                        )
        else:
            st.info(
                "No snapshots were captured. Try a smaller snapshot interval or more generations."
            )

    with downloads_tab:
        st.subheader("Downloads")

        download_col1, download_col2, download_col3, download_col4 = st.columns(4)

        with download_col1:
            st.download_button(
                "Download solution PNG",
                data=image_to_png_bytes(data["solution_image"]),
                file_name="ga_solution.png",
                mime="image/png",
            )

        with download_col2:
            st.download_button(
                "Download comparison PNG",
                data=image_to_png_bytes(data["comparison_image"]),
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

        with download_col4:
            st.download_button(
                "Download history CSV",
                data=fitness_history_to_csv_bytes(data["fitness_history"]),
                file_name="fitness_history.csv",
                mime="text/csv",
                disabled=not data["fitness_history"],
            )

        st.subheader("Run Log")

        with st.expander("Show captured GA stdout log"):
            st.text(data["stdout"] or "No log captured.")

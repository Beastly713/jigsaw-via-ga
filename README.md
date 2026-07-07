# Genetic Algorithm Jigsaw Puzzle Solver

This project creates and solves square-piece jigsaw puzzles from images. It can
shuffle an input image into a puzzle, record the true piece mapping, and then
try to reconstruct the puzzle.

The solver uses a Genetic Algorithm instead of brute force. Candidate
arrangements are improved over generations using fitness scoring, selection,
crossover, mutation, and elitism.

The project provides both a command-line workflow and a local Streamlit
dashboard for visualizing solver behavior, generated artifacts, metrics,
fitness history, and generation snapshots.

## What the Project Does

- Creates shuffled puzzles from an input image.
- Records puzzle-to-original piece mapping in a manifest.
- Solves the puzzle with a Genetic Algorithm.
- Tracks fitness over generations.
- Evaluates the solution with manifest-based metrics.
- Exports artifacts such as solution image, comparison image, CSV, plot, and snapshots.
- Provides a local dashboard for visualization.

## AI / GA Approach

- `Individual`: one candidate arrangement of puzzle pieces.
- `Population`: a set of candidate arrangements.
- `Fitness`: score based on how compatible neighboring piece edges are.
- `Selection`: fitter candidates are more likely to reproduce.
- `Crossover`: combines parent arrangements into a child arrangement.
- `Mutation`: randomly swaps pieces.
- `Elitism`: preserves strong candidates between generations.
- `Termination`: stops at the requested maximum generations or after stagnation.

The implementation tracks `generations_completed`, `termination_reason`,
`best_fitness`, and per-generation fitness history.

## Assumptions and Limitations

- Pieces are square and non-overlapping.
- Piece orientation is fixed.
- There are no missing pieces.
- Pieces are not rotated.
- Images are cropped to a valid grid during puzzle creation.
- Larger piece counts and larger GA settings increase runtime.
- This is an educational/coursework solver, not a production-grade arbitrary
  jigsaw solver.

## Project Structure

```text
gaps/              Core solver, GA, fitness, crossover, selection, CLI
streamlit_app/     Local Streamlit visualization dashboard
scripts/           Demo, experiment, and cleanup scripts
images/            Sample images used for demos
tests/             Unit and workflow tests
outputs/           Generated artifacts; ignored by git
```

## Architecture

The project is organized around a small pipeline:

```text
input image
-> crop to a valid square-piece grid
-> split into pieces
-> shuffle pieces and write a manifest
-> run the Genetic Algorithm solver
-> assemble the best candidate solution
-> evaluate with manifest-based metrics
-> export images, CSVs, plots, snapshots, and dashboard downloads
```

Core responsibilities:

- `gaps/cli.py`: command-line entrypoint for puzzle creation, solving, and baseline generation.
- `gaps/utils.py`: image splitting and reassembly helpers.
- `gaps/genetic_algorithm.py`: GA loop, elitism, mutation, history tracking, and termination state.
- `gaps/individual.py`: candidate puzzle arrangement and fitness access.
- `gaps/image_analysis.py` and `gaps/fitness.py`: edge compatibility cache and dissimilarity scoring.
- `gaps/crossover.py` and `gaps/selection.py`: reproduction logic for the GA.
- `streamlit_app/solver_workflow.py`: in-memory workflow used by the local dashboard.
- `streamlit_app/app.py`: Streamlit UI for controls, results, charts, snapshots, and downloads.
- `scripts/`: reproducible demo, experiment, and cleanup entrypoints.

## Tech Stack Used

- Python: core language for the solver, CLI, scripts, and dashboard.
- NumPy: array operations for image pieces and puzzle assembly.
- OpenCV: image file loading, writing, resizing, and piece-size detection support.
- Pillow: image conversion and PNG byte generation for dashboard downloads.
- Click: `gaps` command-line interface.
- Matplotlib: fitness plot generation from CLI runs.
- Streamlit: local visualization dashboard.
- pytest: automated tests for solver workflow, metrics, and utility behavior.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r streamlit_app/requirements.txt pytest poetry-core
python -m pip install -e . --no-deps --no-build-isolation
```

`streamlit_app/requirements.txt` contains the runtime dependencies needed for
the local dashboard. The editable install exposes the local `gaps` package and
CLI. The `--no-deps --no-build-isolation` flags avoid reinstalling conflicting
pinned dependencies when the local environment already has compatible packages.

## Run Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest
```

`MPLCONFIGDIR=/tmp/matplotlib` avoids local matplotlib cache or permission
issues on some Linux systems.

## Local Streamlit Dashboard

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m streamlit run streamlit_app/app.py
```

Dashboard flow:

- Upload a JPG or PNG image.
- Choose piece size, generations, population, mutation rate, seed, and snapshot interval.
- Preview the cropped image, grid, and piece count.
- Run the solver.
- Inspect the Overview, Images, Metrics, Fitness History, Snapshots, and
  Downloads & Log tabs.

Dashboard downloads:

- Solution PNG.
- Comparison PNG.
- Puzzle manifest JSON.
- Fitness history CSV.
- Individual snapshot PNGs.

## CLI Workflow

The CLI command is `gaps`. The available commands are `create`, `run`, and
`baseline`.

### Create a Puzzle

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps create images/messi.jpg outputs/demo/puzzle.jpg \
  --size=64 \
  --seed=42 \
  --manifest outputs/demo/puzzle.manifest.json
```

This creates:

- `outputs/demo/puzzle.jpg`
- `outputs/demo/puzzle.manifest.json`

The manifest stores the shuffled puzzle-piece ID to original-piece ID mapping.

### Create a Random Baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps baseline outputs/demo/puzzle.jpg outputs/demo/baseline.jpg \
  --size=64 \
  --seed=42
```

The baseline creates a random shuffled arrangement without GA optimization.

### Run the GA Solver

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps run outputs/demo/puzzle.jpg outputs/demo/solution.jpg \
  --size=64 \
  --generations=20 \
  --population=100 \
  --seed=42 \
  --mutation-rate=0.05 \
  --history outputs/demo/history.csv \
  --fitness-plot outputs/demo/fitness.png \
  --original images/messi.jpg \
  --manifest outputs/demo/puzzle.manifest.json \
  --comparison outputs/demo/comparison.jpg \
  --snapshots-dir outputs/demo/snapshots \
  --snapshot-interval 1
```

- `--manifest` enables solution-quality metrics.
- If `--manifest` is omitted, the CLI looks for a default manifest next to the puzzle.
- `--original` is used for the comparison image.
- `--history` writes per-generation fitness CSV.
- `--fitness-plot` writes the fitness plot.
- `--snapshots-dir` writes generation snapshots.

CLI defaults and ranges:

- Piece size: `32` to `128`.
- Default generations: `20`.
- Default population: `200`.
- Mutation rate: `0.0` to `1.0`.

## Demo Script

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_demo.py
```

The demo runs create -> baseline -> solve. It uses `images/baboon.jpg`, seed
`42`, piece size `64`, generations `20`, population `100`, mutation rate
`0.05`, and snapshot interval `5`.

Artifacts are written to `outputs/demo/`:

- Puzzle image.
- Manifest JSON.
- Random baseline image.
- Solution image.
- Fitness history CSV.
- Fitness plot.
- Comparison image.
- Generation snapshots.

## Experiment Script

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_experiments.py
```

The experiment script creates one base puzzle and manifest, then runs a small
grid of GA configurations:

- Generations: `[10, 20]`.
- Population: `[50]`.
- Mutation rates: `[0.0, 0.05, 0.1]`.

Results are written to `outputs/experiments/results.csv`. The CSV is useful for
comparing generations, mutation rate, runtime, fitness, and manifest metrics.

## Cleanup

```bash
.venv/bin/python scripts/clean_artifacts.py
```

Dry run:

```bash
.venv/bin/python scripts/clean_artifacts.py --dry-run
```

The cleanup script removes generated outputs and common root-level generated
files.

## Metrics Explained

- `best_fitness`: best internal GA fitness score; higher is better.
- `average_fitness`: average population fitness per generation.
- `worst_fitness`: lowest population fitness per generation.
- `piece-position accuracy`: percentage of pieces in exact original locations.
- `adjacency accuracy`: percentage of correct neighboring piece relationships.
- `correct positions`: raw count for direct comparison.
- `correct adjacencies`: raw count for neighbor comparison.
- `runtime`: wall-clock solve time.
- `generations completed`: number of generations actually executed.
- `termination reason`: `max_generations` or `stagnation`.

Adjacency accuracy is usually the more meaningful final quality metric because
shifted but correctly connected puzzle segments can have poor absolute position
accuracy.

The final solution-quality metrics are computed using the puzzle manifest rather
than raw image-byte comparison.

## Recommended Quick Demo Settings

```text
image: images/messi.jpg
piece size: 64
generations: 20
population: 100
mutation rate: 0.05
seed: 42
snapshot interval: 1 or 5
```

These settings are useful for a local coursework demonstration because they
generate visible artifacts without making the run unnecessarily large.

## Notes for Evaluation

- The local dashboard is the best way to present the project visually.
- The CLI is useful for reproducible artifact generation.
- The experiment script gives a small structured comparison of GA settings.
- The manifest is important because it makes solution metrics deterministic and meaningful.
---

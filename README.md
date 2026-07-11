# GAPS — Genetic Algorithm Jigsaw Puzzle Solver

GAPS creates and solves grid-based image puzzles made from equal, square pieces. It
includes a Click CLI for reproducible runs, a Streamlit dashboard for interactive
experiments, manifest-based accuracy metrics, fitness history, and generation
snapshots.

The solver is intended for educational and experimental use. It solves a shuffled
rectangular grid of image tiles; it does not handle traditional interlocking piece
shapes, rotation, missing pieces, or unknown orientations.

## Current capabilities

- Center-crop an image to a grid divisible by a chosen piece size.
- Shuffle its pieces reproducibly and write a JSON manifest containing the true
  puzzle-to-original mapping.
- Solve a prepared puzzle with a genetic algorithm using edge compatibility,
  roulette-wheel selection, kernel-growing crossover, elitism, and optional swap
  mutation.
- Stop after the requested generation count or after 10 generations without an
  improvement.
- Record best, average, and worst fitness for every completed generation.
- Measure exact piece-position accuracy and directed horizontal/vertical adjacency
  accuracy when a matching manifest is available.
- Export a solution, random baseline, comparison image, fitness CSV and plot, and
  periodic solution snapshots.
- Run the same create-and-solve workflow in a local Streamlit dashboard, including
  charts, metrics, snapshots, logs, and downloadable artifacts.

## How it works

```text
source image
  -> center-crop to a valid tile grid
  -> split and shuffle tiles
  -> save puzzle + ground-truth manifest
  -> precompute pairwise edge dissimilarities
  -> evolve candidate arrangements
  -> save best arrangement + run artifacts
  -> score against the manifest (when available)
```

Each candidate (`Individual`) is a row-major permutation of the input tiles. Its
fitness is the inverse of the total color-gradient dissimilarity across all adjacent
edges, so higher fitness is better. The crossover grows a child from compatible
neighbors found in both parents and from each tile's best cached edge matches.

Image analysis compares every pair of pieces in both directions for horizontal and
vertical placement. Consequently, preprocessing and GA runtime grow quickly with the
number of pieces, population size, and generation count.

## Requirements and installation

- Python 3.10 or newer (the code uses modern union type syntax; package metadata
  still declares Python 3.8 compatibility)
- Poetry, or `pip` and a virtual environment
- A local graphical environment only when using CLI `--debug`

### Poetry

```bash
poetry install
poetry run gaps --help
```

### pip

The dashboard requirements cover the runtime dependencies and use headless OpenCV:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r streamlit_app/requirements.txt pytest poetry-core
python -m pip install -e . --no-deps --no-build-isolation
```

`pyproject.toml` pins NumPy 1.24 for the Poetry environment, while the dashboard
requirements allow any NumPy version below 2. Keep to one installation method per
environment to avoid dependency resolver surprises.

On machines where Matplotlib cannot write to its normal config directory, prefix
commands with `MPLCONFIGDIR=/tmp/matplotlib` as shown below.

## Quick start: CLI

The installed `gaps` command has three subcommands: `create`, `baseline`, and `run`.
Place a JPG or PNG source image at `images/source.jpg` (or substitute your own path)
before following the example. The only versioned image, `images/lena.gif`, is a
historical asset and is not reliably readable by the OpenCV build used by the CLI.

### 1. Create a puzzle and manifest

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps create \
  images/source.jpg outputs/quickstart/puzzle.png \
  --size 128 --seed 42
```

`create` defaults to 128-pixel pieces. If the source dimensions are not divisible by
the piece size, it center-crops them. It writes the puzzle and, by default, a manifest
beside it using the name `puzzle.manifest.json`. Use `--manifest PATH` to override
that location.

The manifest format is version 1 and contains the source/puzzle paths, piece size,
grid dimensions, cropped dimensions, and the `puzzle_to_original` permutation.

### 2. Create a random baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps baseline \
  outputs/quickstart/puzzle.png outputs/quickstart/baseline.png \
  --size 128 --seed 42
```

This reshuffles the puzzle without optimization. If `--size` is omitted, GAPS tries
to detect a valid piece size between 32 and 128 pixels from the puzzle image.

### 3. Solve and export artifacts

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps run \
  outputs/quickstart/puzzle.png outputs/quickstart/solution.png \
  --size 128 \
  --generations 20 \
  --population 100 \
  --mutation-rate 0.05 \
  --seed 42 \
  --original images/source.jpg \
  --history outputs/quickstart/history.csv \
  --fitness-plot outputs/quickstart/fitness.png \
  --comparison outputs/quickstart/comparison.png \
  --snapshots-dir outputs/quickstart/snapshots \
  --snapshot-interval 5
```

Because the default manifest is next to the puzzle, `run` discovers it automatically.
Pass `--manifest PATH` for a non-default location. With a manifest, the summary
includes piece-position and adjacency accuracy; `--original` is only needed to add
the original image to the comparison output. Without a manifest, solving still works
but ground-truth accuracy is unavailable.

Important `run` defaults and constraints:

| Option | Default | Constraint / behavior |
| --- | ---: | --- |
| `--size` | auto-detect | 32–128 pixels when explicitly supplied |
| `--generations` | 20 | use 2 or more; 1 currently fails in progress reporting |
| `--population` | 200 | positive integer |
| `--mutation-rate` | 0.0 | 0.0–1.0; probability per child |
| `--snapshot-interval` | 1 | positive integer; used with `--snapshots-dir` |
| `--debug` | off | opens a live Matplotlib view each generation |

Run `gaps COMMAND --help` for the complete generated CLI reference.

## Streamlit dashboard

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m streamlit run streamlit_app/app.py
```

Upload a JPG, JPEG, or PNG, then choose the piece size, generations, population,
mutation rate, seed, and snapshot interval. The app previews the centered crop and
grid before running, and warns (without blocking) for more than 150 pieces or an
estimated work score above 5,000,000 (`pieces × generations × population`).

Results include:

- original, shuffled, solved, and combined images;
- exact-position and adjacency metrics;
- runtime, best fitness, completed generations, and termination reason;
- fitness chart and table;
- periodic in-memory generation snapshots;
- solution/comparison PNG, manifest JSON, and history CSV downloads;
- captured solver stdout.

The dashboard creates a new puzzle from the uploaded image for every solver run and
resets the global edge-analysis cache before solving.

## Demo and experiment scripts

From the repository root:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_demo.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_experiments.py
```

Both scripts currently expect `images/baboon.jpg`. That file is ignored by the
repository's `*.jpg` rule and is not part of a fresh clone, so supply your own image
at that path or change the `IMAGE` constant first.

`run_demo.py` performs create → random baseline → solve with seed 42, 64-pixel
pieces, 20 generations, population 100, mutation rate 0.05, and snapshots every five
generations. It writes to `outputs/demo/`.

`run_experiments.py` creates one base puzzle and runs six configurations:

- generations: 10 and 20;
- population: 50;
- mutation rates: 0.0, 0.05, and 0.1.

It writes per-run solutions and histories plus `outputs/experiments/results.csv`,
which records configuration, runtime, termination, fitness, and manifest metrics.

## Tests

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest
```

The suite covers the GA smoke workflow, piece-size detection, manifest validation and
metrics, cropping and image conversion, Streamlit workflow results, fitness CSV/chart
data, warnings, and snapshot behavior.

Some legacy tests reference `images/baboon.jpg`, `images/lena.jpg`,
`images/island.jpg`, and `images/pillars.jpg`. Like the demo fixture, these JPG files
are ignored and are not tracked; add suitable fixtures at those paths before running
the complete suite in a fresh clone. The current development checkout passes all 26
tests.

## Generated files and cleanup

Generated outputs belong under `outputs/`, which is ignored by Git. Remove generated
demo, experiment, and common root-level smoke-test artifacts with:

```bash
.venv/bin/python scripts/clean_artifacts.py --dry-run
.venv/bin/python scripts/clean_artifacts.py
```

Always run the cleanup script from the repository root.

## Project layout

```text
gaps/
  cli.py                 CLI commands, manifests, metrics, and artifact writers
  genetic_algorithm.py   evolution loop, elitism, mutation, history, termination
  image_analysis.py      cached pairwise edge analysis and best-match tables
  fitness.py             edge dissimilarity calculation
  crossover.py           kernel-growing crossover
  selection.py           roulette-wheel parent selection
  individual.py          candidate arrangement and fitness evaluation
  piece.py               indexed tile model
  size_detector.py       contour-based piece-size detection
  utils.py               tile splitting and image assembly
streamlit_app/
  app.py                 dashboard UI
  solver_workflow.py     in-memory create/solve/metrics/export workflow
scripts/                 demo, experiment, and artifact-cleanup entry points
tests/                   CLI, solver, detector, metric, and dashboard workflow tests
images/lena.gif          tracked historical image asset (not a reliable CLI fixture)
outputs/                 generated artifacts (ignored; created on demand)
```

## Limitations

- Pieces must be equal-sized, square, axis-aligned image tiles.
- Rotation, flipping, missing/duplicate pieces, and irregular piece boundaries are
  unsupported.
- `run` requires puzzle dimensions to be exactly divisible by the selected/detected
  piece size; only puzzle creation performs cropping.
- Auto-detection only considers sizes from 32 through 128 that divide both image
  dimensions, and content-dependent contour detection can be ambiguous.
- Reproducible seeds control Python and NumPy randomness, but they do not guarantee
  identical results across every dependency/platform combination.
- Although the CLI and dashboard accept one generation, the current terminal
  progress calculation divides by zero for that value; configure at least two.
- Fitness measures local edge compatibility, so the highest-fitness arrangement is
  not guaranteed to match the original global placement.
- The global image-analysis cache is process-wide. The Streamlit workflow clears it
  between runs; direct library callers should do the same when solving unrelated
  puzzles in one process.

## License

MIT (as declared in `pyproject.toml`).

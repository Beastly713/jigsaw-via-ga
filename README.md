# Genetic Algorithm Jigsaw Puzzle Solver

This project solves square-piece jigsaw puzzles using a Genetic Algorithm (GA).
Given a shuffled puzzle image, the goal is to reconstruct the original image by
searching for a good arrangement of pieces.

Jigsaw reconstruction is a combinatorial optimization problem: even a modest
number of pieces creates a very large search space. Instead of brute forcing all
possible arrangements, this project uses an evolutionary search process to
improve candidate solutions over generations.

## Why This Is an AI Project

The solver uses a Genetic Algorithm, a population-based AI search technique.
Each candidate solution is an arrangement of puzzle pieces. The algorithm
evaluates candidates using a fitness function, selects better individuals,
combines them through crossover, applies mutation, preserves strong candidates
through elitism, and terminates when it reaches the requested maximum number of
generations or when progress stagnates.

Core AI concepts used:

- population-based search
- fitness evaluation
- selection
- crossover
- mutation
- elitism
- termination by max generations or stagnation

## Workflow

```text
original image
-> crop to a valid piece grid if needed
-> create shuffled puzzle + puzzle manifest
-> run GA solver
-> track fitness
-> save solution/artifacts
-> evaluate with manifest-based metrics
```

The puzzle manifest records the mapping from shuffled puzzle-piece IDs to
original-piece IDs. The solver uses this mapping to compute solution-quality
metrics after a run.

## Installation

Create a virtual environment and install the project dependencies as usual for
this repository. The commands below assume the project is installed in `.venv`
and that the `gaps` console script is available at `.venv/bin/gaps`.

```bash
.venv/bin/python -m pytest
```

## Main CLI Usage

### Create Puzzle

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps create images/baboon.jpg outputs/demo/puzzle.jpg --size=64 --seed=42
```

This creates a shuffled square-piece puzzle and writes a manifest JSON next to
the puzzle. The manifest stores the true shuffle mapping used later for
solution-quality metrics.

The command writes:

- `outputs/demo/puzzle.jpg`
- `outputs/demo/puzzle.manifest.json`

If the image dimensions are not divisible by the piece size, the CLI crops the
image to the nearest valid centered grid before creating pieces.

You can also provide the manifest path explicitly:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps create images/baboon.jpg outputs/demo/puzzle.jpg \
  --size=64 \
  --seed=42 \
  --manifest outputs/demo/puzzle.manifest.json
```

### Run Solver

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps run outputs/demo/puzzle.jpg outputs/demo/solution.jpg \
  --size=64 \
  --generations=20 \
  --population=100 \
  --seed=42 \
  --mutation-rate=0.05 \
  --history outputs/demo/history.csv \
  --fitness-plot outputs/demo/fitness.png \
  --original images/baboon.jpg \
  --manifest outputs/demo/puzzle.manifest.json \
  --comparison outputs/demo/comparison.jpg \
  --snapshots-dir outputs/demo/snapshots \
  --snapshot-interval 5
```

This runs the Genetic Algorithm and writes the solved image plus optional
artifacts:

- fitness history CSV
- fitness plot
- solution-quality metrics
- side-by-side comparison image
- generation snapshots

The `--manifest` option enables paper-aligned solution-quality metrics. If
`--manifest` is not provided, `gaps run` automatically looks for
`puzzle.manifest.json` next to the puzzle image. The `--original` option is now
only needed for comparison image output, not for metrics.

### Random Baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps baseline outputs/demo/puzzle.jpg outputs/demo/baseline.jpg --size=64 --seed=42
```

The random baseline creates a shuffled arrangement without GA optimization. It
is useful as a simple comparison point for the GA output.

## Example: Messi Image Workflow

```bash
.venv/bin/python scripts/clean_artifacts.py

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps create images/messi.jpg outputs/demo/puzzle.jpg --size=64 --seed=42

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps baseline outputs/demo/puzzle.jpg outputs/demo/baseline.jpg --size=64 --seed=42

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

Sample run output from one deterministic run:

```text
Cropped image from 976x549 to 960x512 to fit piece size 64
Created puzzle with 120 pieces
Manifest: outputs/demo/puzzle.manifest.json

Metric method: manifest
Piece-position accuracy: 97.50%
Correct positions: 117/120
Adjacency accuracy: 97.24%
Correct adjacencies: 211/217
```

## Demo Script

Run the full project workflow with one command:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_demo.py
```

The demo uses `images/baboon.jpg` with piece size `64` and writes artifacts to
`outputs/demo/`:

- `puzzle.jpg`
- `puzzle.manifest.json`
- `baseline.jpg`
- `solution.jpg`
- `history.csv`
- `fitness.png`
- `comparison.jpg`
- `snapshots/`

The demo script explicitly passes the manifest to the solver, so
solution-quality metrics are computed using manifest-based piece identity.

The GA may terminate early when progress stagnates, so the number of snapshot
images can be smaller than the requested maximum number of generations.

## Experiment Script

Run a small experiment grid:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_experiments.py
```

The experiment runner reuses one generated puzzle and manifest across all runs
so configurations are compared on the same shuffled puzzle. It compares a small
set of GA configurations over generations and mutation rate, and writes:

```text
outputs/experiments/puzzle.manifest.json
outputs/experiments/results.csv
```

The CSV includes `manifest_path`, `piece_position_accuracy`, and
`adjacency_accuracy`. It is useful for comparing how configuration choices
affect fitness, runtime, completed generations, termination reason, and
solution-quality metrics.

## Metrics Explained

- `best_fitness`: best GA optimization score found in a run. Higher is better
  because lower edge dissimilarity is converted into higher fitness.
- `average_fitness`: average population fitness per generation, saved in the
  history CSV and plotted when requested.
- `piece-position accuracy` / direct comparison: percentage of pieces placed in
  their exact original cell. This is useful but secondary because shifted
  solutions can score poorly.
- `adjacency accuracy` / neighbor comparison: percentage of correct neighboring
  piece relationships in the solved arrangement. This is the primary accuracy
  metric used for final solution quality.
- `correct positions`: raw count behind piece-position accuracy.
- `correct adjacencies`: raw count behind adjacency accuracy.
- `runtime`: wall-clock time reported by the CLI for the GA solve.
- `generations completed`: number of generations actually executed.
- `termination reason`: whether the GA stopped at max generations or due to
  stagnation.

The final solution-quality metrics are computed from the puzzle manifest, not
by comparing raw image bytes. This makes the metrics robust to JPEG save/load
behavior.

### Why Adjacency Accuracy Is Primary

The referenced jigsaw-solving literature reports two common measures: direct
comparison and neighbor comparison. Neighbor comparison is usually more
meaningful because a solution can contain correctly assembled segments even if
the whole segment is shifted away from its absolute original location.
Therefore this project reports both, but treats adjacency accuracy as the
headline solution-quality score.

## Project Structure

```text
gaps/       core solver, GA, fitness, crossover, selection, CLI
scripts/    reproducible demo, experiment, and artifact cleanup runners
images/     sample images used for testing and demos
outputs/    generated puzzles, manifests, solutions, plots, CSVs, snapshots; ignored by git
```

## Testing

Run the test suite with:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest
```

The current test suite includes manifest validation and manifest-based metric
tests.

Current expected local result: 10 passed.

## Current Limitations

- The project works on square-piece Type-1 puzzles: piece orientation and
  puzzle dimensions are known.
- `gaps create` can crop images to a valid centered grid, but the solver still
  expects an input puzzle whose dimensions are divisible by the chosen piece
  size.
- Solution-quality metrics require a puzzle manifest. Puzzles not created
  through `gaps create` can still be solved, but paper-aligned accuracy metrics
  will be unavailable unless a valid manifest is provided.
- The GA may not fully solve harder or larger puzzles in low-generation demo
  settings.
- There is no web UI yet.

## Future Work

- Streamlit demo
- richer experiment grid
- GIF or video generation from snapshots
- improved heuristic, crossover, and mutation variants
- simple demo page

## Reference

This project is inspired by research on automatic jigsaw puzzle solving with
genetic algorithms:

```text
@article{Sholomon2016,
  doi = {10.1007/s10710-015-9258-0},
  url = {https://doi.org/10.1007/s10710-015-9258-0},
  year = {2016},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {17},
  number = {3},
  pages = {291--313},
  author = {Dror Sholomon and Omid E. David and Nathan S. Netanyahu},
  title = {An automatic solver for very large jigsaw puzzles using genetic algorithms},
  journal = {Genetic Programming and Evolvable Machines}
}
```

## License

This project is available as open source under the terms of the
[MIT License](LICENSE).

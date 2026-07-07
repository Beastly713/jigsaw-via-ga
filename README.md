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
-> create shuffled puzzle
-> run GA solver
-> track fitness
-> save solution/artifacts
-> evaluate with metrics
```

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

This creates a shuffled square-piece puzzle from the original image.

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

### Random Baseline

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/gaps baseline outputs/demo/puzzle.jpg outputs/demo/baseline.jpg --size=64 --seed=42
```

The random baseline creates a shuffled arrangement without GA optimization. It
is useful as a simple comparison point for the GA output.

## Demo Script

Run the full project workflow with one command:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_demo.py
```

The demo uses `images/baboon.jpg` with piece size `64` and writes artifacts to
`outputs/demo/`:

- `puzzle.jpg`
- `baseline.jpg`
- `solution.jpg`
- `history.csv`
- `fitness.png`
- `comparison.jpg`
- `snapshots/`

The GA may terminate early when progress stagnates, so the number of snapshot
images can be smaller than the requested maximum number of generations.

## Experiment Script

Run a small experiment grid:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/run_experiments.py
```

The experiment runner reuses one generated puzzle and compares a small set of
GA configurations over generations and mutation rate. It writes results to:

```text
outputs/experiments/results.csv
```

This CSV is useful for comparing how configuration choices affect fitness,
runtime, completed generations, termination reason, and solution-quality
metrics.

## Metrics Explained

- `best_fitness`: best GA fitness value found in a run.
- `average_fitness`: average population fitness per generation, saved in the
  history CSV and plotted when requested.
- `piece-position accuracy`: percentage of solved pieces that exactly match the
  original image at the same position.
- `adjacency accuracy`: percentage of neighboring piece relationships that are
  correct in the solved arrangement.
- `runtime`: wall-clock time reported by the CLI for the GA solve.
- `generations completed`: number of generations actually executed.
- `termination reason`: whether the GA stopped at max generations or due to
  stagnation.

## Project Structure

```text
gaps/       core solver, GA, fitness, crossover, selection, CLI
scripts/    reproducible demo and experiment runners
images/     sample images used for testing and demos
outputs/    generated artifacts; ignored by git
```

## Testing

Run the test suite with:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest
```

## Current Limitations

- The project works on square-piece puzzles.
- The piece size must evenly divide the image dimensions.
- Exact pixel matching metrics assume puzzle pieces come from the provided
  original image.
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

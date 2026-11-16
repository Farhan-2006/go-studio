# Go move adviser
Farhan Naik Â· 2410110422

## Start here
This is a Python 3.12 desktop application recommending a legal move on a 19Ã—19 Go board.
The four trained models run locally on the CPU. No API, subscription payment, or internet
is needed for the saved-model demonstration.

Install Python 3.12 with Tcl/Tk, create a virtual environment and install `requirements.txt`. Run `python Go_Project.py demo` from the repository folder. The included weights are ready to use.

## Commands
Run from this folder using the project environment:
```
python Go_Project.py test
python Go_Project.py prepare --games 800 --positions 16
python Go_Project.py train --epochs 12
python Go_Project.py evaluate --limit 500
python Go_Project.py reference
python Go_Project.py demo
python Go_Project.py recommend --position examples/single_position.json --model cnn
```
Preparation may download the archive if absent. Other commands use local files.
Repreparing or retraining overwrites generated artefacts. Rerun evaluation afterwards.

## Demonstration
1. The app opens on an example position. Pick another example at the top or use the board editing controls.
2. Set **Who moves next?** and click **Find next move**. The green marker shows the recommendation.
3. Open **Analysis** or click **Model breakdown** to compare the four individual choices. Select a method to see its top five moves, scores, and numbered board markers. For an example, the panel also shows the recorded move.
4. **Play this move** applies the recommendation. **Undo** restores the previous position.
5. Use **Test results** for the measured comparison and **File** to save or open a position.

The combined score averages each move's percentile rank across the four models with equal
weights. A higher score means the models ranked the move higher overall. Use
`python Go_Project.py evaluate-combined` to evaluate it on the saved test sample.

Editing a diagram is setup, not gameplay: it does not capture stones or establish history.
The input validator rejects groups with no liberties. A new empty game tracks history for
applied moves; a standalone diagram has no earlier history. Repetition checks are then
limited to the information available. No claim of optimal play or calibrated confidence.

## Experiment design
Public professional SGFs: https://homepages.cwi.nl/~aeb/go/games/
Source states the games are public domain. Archive hash and exclusions: `data/manifest.json`.
Entire games are assigned 80/10/10 to train/validation/test. Canonicalised complete move
sequences remove rotated/reflected duplicates before splitting. Training seed: 42.
Nonstandard board sizes, setup/handicap, short, invalid and nonalternating records are excluded.
Only games fully replayable with our rules are accepted. Sixteen positions per game are sampled.

The classical models learn binary candidate ranking with twelve random unplayed alternatives
per recorded move. These are not proven bad moves. LR and MLP use a training-fitted scaler.
The forest uses unscaled features. The CNN learns 362-way move prediction from five spatial
channels. Its checkpoint is selected by validation cross-entropy. All methods share the game
splits and test positions, but representation/objective differ. This is a comparison of
practical approaches, not a controlled architecture-only experiment.

`results/metrics.csv` reports agreement with recorded moves (top-1/top-5), game-cluster
bootstrap confidence intervals, inference latency, model bytes, and fitting time.
Training times exclude shared feature preparation, recorded separately in model metadata.
Latency includes full candidate feature preparation even for CNN and excludes model loading.
Test positions are evaluated as standalone boards plus last move; historical superko context
is not supplied. Full-game playing strength is not measured.

## External reference
KataGo v1.16.3 Eigen AVX2 CPU is used only for independent analysis of eight held-out example
positions. To reproduce the engine analysis on Windows, extract the official Eigen AVX2 v1.16.3 release into `reference_engine/`. Download the b10c128 network linked in `results/engine_reference.json` as `reference_engine/network.txt.gz` and save its URL in `reference_engine/network_source.txt`.
Chinese rules, 7.5 komi and a 64-visit budget are explicit reference assumptions.
`results/engine_reference.json` contains raw engine output and model agreement.
The engine is approximate, limited-budget, and not proof of optimality. Eight examples are
illustrative, not a statistically conclusive engine benchmark. It is never used to train our models.

KataGo: https://github.com/lightvector/KataGo (MIT licence).
Its SGF-independent initial-stone input does not restore historical repetition context.

## Files
- `Go_Project.py`: complete implementation and command-line entry point.
- `models/`: fitted models and training metadata; load only trusted model files.
- `data/raw/games.tgz`: public source archive.
- `data/processed/`: reproducible game-level splits.
- `data/manifest.json`: source, checksum, filtering and split membership.
- `examples/positions.json`: held-out demonstration positions.
- `results/`: measured metrics, predictions, charts, learning curve and environment.
- `reference_engine/`: official external engine, network, settings and licence.

## Limitations and next steps
Limited training data and shallow models do not yield expert Go strength. Professional
move labels are ambiguous; multiple good moves exist. Passing is sparsely supervised.
Diagrams may be implausible or lack history. Long-range tactics and life/death reasoning are
weak. Score estimation, resignation and endgame adjudication are not implemented.
A larger dataset, symmetry augmentation inside training only, a broader CNN receptive field,
careful hyperparameter tuning and guided search are possible extensions.

## Current verification status
All seven rule tests and split checks pass.
All four model files load and produce recommendations. Automated Tkinter interaction checks cover launch, both tabs, model selection and ranked board markers,
applying a move, undo, clearing, board editing, and resizing.

The MLP reached its configured iteration limit without convergence. This is reported rather
than hidden; the fitted model remains usable. See RESULTS_SUMMARY.md and results/training_log.txt.

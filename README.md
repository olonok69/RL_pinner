# RL Connector Pinning Agent

Reinforcement Learning pipeline to learn and predict **signal-to-pin assignments** for electrical connectors.

This repository implements two RL agents:

- **Agent 1 (specialized)**: trains per connector pin-count class and uses a discrete action space (DQN only).
- **Agent 2 (generic)**: trains by connector size and uses a continuous action space over a 14-feature pin representation (TD3, PPO, A2C, SAC, DDPG).

This README is based on both the source code and `Reinforcement Learning Connector Pinning Agent.docx`.

---

## Project Goal

Given connector metadata (signal, wire, geometric and multicore features), train RL policies that score or generate good pinning layouts according to domain rules such as:

- separate power vs ground
- separate high-power vs information signals
- place high-voltage signals safely
- account for cavity internal/external constraints
- avoid same-color adjacency (unless thickness differs)
- keep multicore relationships consistent

---

## Repository Structure

- `main.py` — CLI entrypoint for training/prediction/demo workflows
- `pre_processing.py` — feature engineering, dictionary building, model orchestration, multiprocessing prediction
- `connector.py` — Agent 1 gym env + DQN train/predict
- `connector2.py` — Agent 2 gym env + continuous-action algorithms train/predict
- `utils.py` — pickle save/load helpers
- `requirements.txt` — Python dependencies
- `data/` — sample datasets and artifacts
- `conf/` — example generated dictionaries (`connectors_dicc.pkl`, `signal_cat.pkl`)
- `models/` — model files (created at runtime)
- `Notebooks/` — exploratory notebooks and experiments

---

## Environment & Dependencies

- OS: tested in a Windows-style workflow (paths are hardcoded as `c:/temp/` in code)
- Python: project doc recommends **Python 3.8**
- Core libs:
  - `stable-baselines3[extra]`
  - `pandas`
  - `numpy`
  - `gym`

Install:

```bash
pip install -r requirements.txt
```

> `gym` is imported directly in code; if your environment does not already include it through transitive deps, install it explicitly.

---

## Important Runtime Path Behavior

The pipeline reads/writes from a **hardcoded base directory**:

- `c:/temp/data`
- `c:/temp/conf`
- `c:/temp/models`

`main.py` auto-creates these folders if missing, but it does **not** copy repository `data/*` into `c:/temp/data`.

### Before running

Copy required input files from repo to `c:/temp/data`:

- `data/inline_pins.csv`
- `data/symbol_pins.csv`

For prediction/demo, also provide:

- `c:/temp/data/new_connector.csv`

---

## Data Inputs

### 1) Inline connector data (`inline_pins.csv`)

Expected columns (from dataset/code):

- `Connector Name`
- `Connector PartNumber`
- `Pin Name`
- `Pin PreferredSignal`
- `Signal Name`
- `Wire WireColor`
- `Wire WireCSA`
- `MulticoreInnerToOutter1`
- (other multicore columns may exist in source CSV)

### 2) Symbol geometry data (`symbol_pins.csv`)

Expected columns:

- `Symbol Name`
- `Pin Name`
- `Pin CenterY`
- `Pin CenterX`
- `Pin Width`
- `Pin Height`

These are merged in preprocessing to compute pin geometry features:

- distance-to-center
- nearest neighbors
- internal/external cavity heuristics
- centroid-related metrics

---

## Agent Overview

### Agent 1 — Specialized (DQN)

- Environment: `conn` in `connector.py`
- Action space: `Discrete(state_space)`
- Observation: vector of pin assignments sized by connector pins
- Trains one model for a selected connector class (effectively by pin count)

Model output location:

- `c:/temp/models/model_<num_pins>.pkl`

### Agent 2 — Generic (Continuous action)

- Environment: `agent2` in `connector2.py`
- Action/observation vector size: **14 features**

Feature schema (as implemented):

1. number of pins
2. distance to center (bucketized)
3. number of neighbors
4. internal/external flag
5. mean-neighbor distance (bucketized)
6. centroid distance (bucketized)
7. signal category
8. wire color category
9. wire thickness (bucketized)
10. multicore flag
11. number of internal pins
12. empty-cavity flag
13. multicore indicator
14. multicore category

Supported algorithms:

- `TD3`, `PPO`, `A2C`, `SAC`, `DDPG`

Model output location:

- `c:/temp/models/model_<ALGO>_agent2_<size>.pkl`

---

## Preprocessing Pipeline

`preprocess_data()` in `pre_processing.py` performs:

- missing-value filling (`na`, numeric defaults)
- connector/pin aggregation (counts, uniqueness)
- signal categorization by naming patterns (e.g., `BATT_`, `CAN_`, `CANFD_`, `GND_`, `HV_`)
- category encoding for signals and wire colors
- multicore feature extraction
- merge with symbol geometry + neighbor computation

`create_connector_dicc()` builds a nested connector dictionary consumed by environments.

Serialized artifacts:

- `c:/temp/conf/connectors_dicc.pkl`
- `c:/temp/conf/signal_cat.pkl`
- `c:/temp/data/connector_processed.pickle`
- `c:/temp/data/symbols_processed.pickle`

---

## CLI Usage

Entrypoint:

```bash
python main.py --mode <training|prediction|demo> --connector <name> --agent <1|2> --algo <algo>
```

Arguments:

- `--mode` (required): `training`, `prediction`, or `demo`
- `--connector` (required by parser): connector name, especially relevant for Agent 1 training
- `--agent` (required): `1` (specialized) or `2` (generic)
- `--algo` (required):
  - Agent 1: `DQN`
  - Agent 2: one of `TD3`, `PPO`, `A2C`, `SAC`, `DDPG`

### Typical workflow

### 1) Train

Agent 1 (specialized):

```bash
python main.py --mode training --connector "TO_111_LH_F/143_3P" --agent 1 --algo DQN
```

Agent 2 (generic / multi-size models):

```bash
python main.py --mode training --connector "TO_111_LH_F/143_3P" --agent 2 --algo PPO
```

> For Agent 2 training, `--connector` is parsed but training actually iterates connector-size groups from processed data.

### 2) Predict

Prepare `c:/temp/data/new_connector.csv` with the same schema used by preprocessing.

Agent 1:

```bash
python main.py --mode prediction --connector "placeholder" --agent 1 --algo DQN
```

Agent 2:

```bash
python main.py --mode prediction --connector "placeholder" --agent 2 --algo PPO
```

Prediction artifacts (Agent 2):

- `c:/temp/data/prediction_pins.csv`
- `c:/temp/data/prediction_pins_with_p.csv`
- `c:/temp/data/best_prediction.csv`

### 3) Demo

```bash
python main.py --mode demo --connector "placeholder" --agent 2 --algo PPO
```

Demo artifacts:

- `c:/temp/data/demo_13_pins.csv`
- `c:/temp/data/demo_13_pins_with_predictions.csv`

---

## Reward Logic (High-level)

Reward components encode manufacturing best-practice heuristics, including:

- signal-category compatibility/separation
- adjacency penalties/rewards using neighbor topology
- wire color and thickness interactions
- internal/external cavity suitability
- multicore placement coherence

`calculate_max_reward()` in `pre_processing.py` provides an estimated theoretical cap for comparison during Agent 2 prediction reporting.

---

## Multiprocessing Behavior

- Agent 2 training (`train_models_multi`) spawns one process per detected connector size.
- Agent 2 prediction (`prediction_multiprocessing`) splits candidate permutations into CPU-based chunks and scores them in parallel.

This improves throughput but can increase memory/CPU pressure on large candidate sets.

---

## Known Caveats / Current Constraints

- Hardcoded runtime path: `c:/temp/`.
- Strong dependence on exact CSV column names.
- `--connector` is required even in some modes where it is not semantically used.
- Prediction requires prior training/preprocessing artifacts.
- Some logic relies on `eval()` over serialized dictionary strings.
- Limited built-in logging/metrics and no experiment tracking by default.

---

## Reproducibility Notes

To make runs reproducible and easier to debug:

- pin Python/package versions
- seed all random generators consistently
- log model/config metadata per run
- archive input data snapshot + generated `connectors_dicc.pkl`

---

## Suggested Next Improvements (aligned with project TODO)

- hyperparameter tuning workflow
- improved training evaluation and validation metrics
- richer logging/reporting
- reward-function calibration
- broader model comparison beyond current algorithms
- configuration-driven paths instead of hardcoded `c:/temp/`

---

## Troubleshooting

- **File not found in `c:/temp/data`**: copy required CSVs from repo `data/` into `c:/temp/data`.
- **Prediction exits early with missing attributes**: ensure `new_connector.csv` has `Connector PartNumber` and required columns.
- **Model load errors**: verify training completed and model filename matches selected algorithm/connector size.
- **Dependency issues**: verify Python 3.8+ environment and reinstall `requirements.txt`.

---

## Quick Start Checklist

- Install dependencies
- Create `c:/temp/{data,conf,models}` (or run once to auto-create)
- Copy training CSVs to `c:/temp/data`
- Run `--mode training`
- Add `new_connector.csv` to `c:/temp/data`
- Run `--mode prediction`
- Inspect generated prediction CSVs in `c:/temp/data`

---

## MAT Quick Start (5 Minutes)

Use this sequence in PowerShell from the repo root for a first end-to-end run.

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Create runtime folders used by the code

```powershell
New-Item -ItemType Directory -Force -Path C:\temp\data, C:\temp\conf, C:\temp\models | Out-Null
```

### 3) Copy required training inputs

```powershell
Copy-Item .\data\inline_pins.csv C:\temp\data\inline_pins.csv -Force
Copy-Item .\data\symbol_pins.csv C:\temp\data\symbol_pins.csv -Force
```

### 4) Train a first model

Agent 1 (DQN, specialized by connector class):

```powershell
python main.py --mode training --connector "TO_111_LH_F/143_3P" --agent 1 --algo DQN
```

or Agent 2 (generic, example with PPO):

```powershell
python main.py --mode training --connector "TO_111_LH_F/143_3P" --agent 2 --algo PPO
```

### 5) Run prediction

Place a `new_connector.csv` in `C:\temp\data` (same schema as training input), then run:

```powershell
python main.py --mode prediction --connector "placeholder" --agent 2 --algo PPO
```

Check outputs:

- `C:\temp\data\prediction_pins.csv`
- `C:\temp\data\prediction_pins_with_p.csv`
- `C:\temp\data\best_prediction.csv`

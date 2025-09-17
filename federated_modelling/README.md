# Federated Modelling

This module implements federated learning for microbiome classification, enabling distributed training across multiple nodes while preserving data privacy. The workflow is orchestrated using configuration in `pyproject.toml` and automated via the `start_federation.sh` script.

---

## Overview

- **Federated Learning:** Multiple clients (nodes) collaboratively train a global model without sharing raw data.
- **Orchestration:** The federation is managed by a central server and multiple clients, with logging and model aggregation.
- **Configuration:** All key parameters (e.g., number of nodes, training settings) are defined in `pyproject.toml`.
- **Inference:** Predictions are made using the final global model produced by the federation.

---

## Data Preparation for Federation
```bash
cd federated_modelling
python data_preparation.py
```

To ensure robust and fair federated learning, we carefully partitioned the data into clients using **Stratified Group K-Fold** based on subject ID. This guarantees:

- **No subject leakage:** Each subject's samples are assigned to only one client, preventing data leakage across clients.
- **Balanced classes:** Each client contains a mix of all sample types (body sites), which is necessary for multi-class logloss to work correctly.

> **Note:**  
> The approach suggested on the competition info page—assigning only one sample type per client—was not feasible, as logloss requires each client to have multiple classes for proper training and evaluation.

The resulting files in `Data/` (e.g., `train_with_5_folds_gkf.csv`) reflect these balanced, non-leaking splits.

---

## Directory Structure

```
federated_modelling/
├── client.py                # Client-side training logic
├── server.py                # Server-side orchestration and aggregation
├── main.py                  # Entry point for running federation
├── data_preparation.py      # Prepares data for federated training
├── inference.py             # Inference using the global model
├── task.py                  # Task definitions for training rounds
├── start_federation.sh      # Shell script to launch the federation
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # uv lock file
├── Data/                    # Processed data for training/testing
├── logs/                    # Logs for federation and nodes
├── Models/                  # Saved global models
├── Subs/                    # Submission files
├── README.md                # This documentation
```

---

## Quick Start

### 1. Install Dependencies

This module uses [uv](https://github.com/astral-sh/uv) for reproducible Python environments.

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync
```

All dependencies and configuration are managed in `pyproject.toml`.

### 2. Start Federated Learning

Change directory to `federated_modelling` and launch the federation using the provided shell script:

```bash
cd federated_modelling
./start_federation.sh
```

This script will:
- Start the server and all client nodes as defined in your configuration.
- Prepare data splits for each node.
- Begin federated training rounds, logging progress in `logs/`.
- Save the final global model in `Models/`.

---

## Configuration

- **pyproject.toml:**  
  - Defines the number of clients, training parameters, model settings, and dependencies.
  - Adjust parameters here to change federation behavior (e.g., number of rounds, aggregation method, model type).

- **start_federation.sh:**  
  - Automates the launch of server and clients.
  - Reads configuration from `pyproject.toml` and sets up the federation environment.
  - Handles logging and process management for all nodes.

---

## Data Handling

- Training and test data are stored in `Data/` (e.g., `train_with_5_folds_gkf.csv`, `test_with_formatted_id.csv`).
- Each client receives its own data partition for local training, with no subject overlap.
- No raw data is shared between nodes; only model updates are communicated.

---

## Logs and Models

- **Logs:**  
  - All federation activity is logged in the `logs/` directory for debugging and auditing.
  - Includes logs for the server, each supernode, and the overall federation.

- **Models:**  
  - Final global models are saved in `Models/` (e.g., `final_global_model_500_5_bagging_gkf.json`).

---

## Inference

After federated training, you can perform inference using the saved global model:

1. Ensure your test data is formatted and available in `Data/`.
2. Run the inference script:
   ```bash
   python inference.py
   ```
   This will load the final global model from `Models/` and generate predictions on the test set, saving results in `Subs/`.
   This is the sub that you will use to get your federated score.

---

## Notes

- Ensure you have run data preparation and have the required CSVs in `Data/`.
- You can customize the federation setup by editing `pyproject.toml` and `start_federation.sh`.
- For advanced usage, modify `client.py`, `server.py`, or `task.py` to implement custom training or aggregation logic.

---
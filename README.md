# UAV Mission Planning for Post-Disaster Victim Localisation via Federated Multi-Agent Reinforcement Learning

This repository contains the official implementation for the paper:

> **UAV Mission Planning for Post-Disaster Victim Localisation via Federated Multi-Agent Reinforcement Learning**  
> Alparslan Güzey, Mehmet Akif Çifçi, Fazlı Yıldırım, Arda Yaşar Erdoğan  
> *Drones*, vol. 10, no. 5, Art. 385, MDPI, 2026.  
> DOI: [10.3390/drones10050385](https://doi.org/10.3390/drones10050385)

---

## Overview

After urban disasters, rapidly localising trapped survivors from their Bluetooth Low Energy (BLE) beacons is critical but difficult: signals are intermittent, rubble obstructs radio propagation, and UAVs are energy-constrained.

This work formulates post-disaster victim localisation as a **cooperative Dec-POMDP** and proposes **Model-Aided FedQMIX** — a framework that combines:

- A **LoS/NLoS surrogate channel model** (SLAL) that learns the local propagation environment from BLE anchor measurements and enables efficient digital-twin training.
- **PSO-based victim position estimation** that fuses sporadic RSSI snapshots into coarse but physically consistent location hypotheses.
- **FedQMIX (FedAvg over QMIX parameters)** for privacy-preserving, communication-aware coordination — UAVs share only network weights, not raw trajectories or RSSI logs.
- An **SAR-aligned shaped reward** that explicitly incentivises victim discovery while retaining throughput as a secondary term.
- A **leakage-free global state** that uses PSO-estimated victim positions rather than privileged ground-truth coordinates during centralised training.
- **Return-to-base and map-feasibility safety filters** enforced at the action level.

The framework is evaluated on two synthetic post-earthquake urban maps across five independent seeds per method, and benchmarked against IQL, plain QMIX, Model-Aided QMIX, IPPO, and a coverage heuristic.

---

## Repository Structure

```
.
├── main.py                      # Entry point for training
├── runner.py                    # Training loop and evaluation
├── training_procedures.py       # Dispatcher: plain QMIX / MOD / FED / FedMOD / IQL
├── evaluate.py                  # Standalone evaluation script
├── run_all.py                   # Run all algorithms (IQL, MOD, FED, IPPO) in sequence
│
├── config/
│   ├── RBM_define.py            # Return-to-Base Mission map (600×800 m)
│   └── RDM_define.py            # Reach-to-Destination Mission map (1000×1200 m)
│
├── Libs/
│   ├── ChannelEstimator.py      # SLAL surrogate channel model + PSO localisation
│   ├── ChannelModel/            # LoS/NLoS segmented channel model
│   ├── CitySimulator/           # Synthetic urban map generation
│   └── Environments/
│       ├── DataCollection.py    # Main RL environment (Dec-POMDP)
│       ├── EnvironmentBase.py
│       ├── IoTDevice.py         # BLE device / victim model
│       ├── IPPOEnv.py           # IPPO-compatible wrapper
│       └── RLLibWrapper.py      # RLLib wrapper (for IPPO)
│
├── agent/agent.py               # Agent interface
├── common/
│   ├── arguments.py             # All CLI arguments
│   ├── rollout.py               # Episode rollout worker
│   └── replay_buffer.py        # Experience replay buffer
├── network/                     # QMIX mixing network + GRU agent network
├── policy/                      # QMIX policy
│
├── run_model_qmix.sh            # Shell script: Model-Aided QMIX sweep
├── run_fed_model_qmix.sh        # Shell script: Model-Aided FedQMIX sweep
├── run_plain_qmix.sh            # Shell script: Plain QMIX sweep
├── run_ablation_grid.sh         # Ablation over surrogate/federation toggles
│
├── run_sensitivity.py           # Reward-weight sensitivity analysis
├── run_channel_shift.py         # RF channel-shift robustness
├── run_device_heterogeneity.py  # BLE/smartphone hardware heterogeneity
├── run_lambda_sensitivity.py    # Lambda (reward shaping) sensitivity
├── run_partial_fedavg.py        # Partial-client FedAvg robustness
├── analyse_noniid.py            # Non-IID client data analysis
│
├── plot_learning_curves.py      # Plot training curves
├── plot_all_baselines.py        # Compare all methods
├── generate_revision_figures.py # Reproduce paper figures
│
└── result/                      # Training outputs (gitignored, created at runtime)
    ├── qmix/<map><tag>/
    │   ├── eval_metrics.csv     # Per-checkpoint evaluation metrics
    │   ├── global_data.npy
    │   └── global_rewards.npy
    ├── iql/
    └── ippo/
```

---

## Scenarios

| | **RBM** (Return-to-Base Mission) | **RDM** (Reach-to-Destination Mission) |
|---|---|---|
| Map size | 600 × 800 m | 1000 × 1200 m |
| UAVs | 3 (altitudes 55, 60, 65 m) | 3 (altitudes 55, 60, 65 m) |
| Battery budget | 60 time steps each | 80 time steps each |
| Victims / IoT devices | 10 total (3 known anchors, 7 unknown) | 10 total (3 known anchors, 7 unknown) |
| Channel step size | 5 m grid | 5 m grid |

UAVs share a 6-action discrete horizontal motion space (N/S/E/W/HOVER/NO-OP) and operate under TDMA-style channel access: each UAV contacts at most one device per time slot, and each device is served by at most one UAV per slot.

---

## Requirements

```
python >= 3.7.15
numpy >= 1.21.5
torch >= 1.12.1
matplotlib >= 3.5.3
pyswarms >= 1.3.0
```

Install dependencies:

```bash
pip install numpy torch matplotlib pyswarms
```

For IPPO support (optional), RLLib is also required:

```bash
pip install "ray[rllib]"
```

---

## Quick Start

### Train a single run

```bash
# Model-Aided FedQMIX on RDM, 3 UAVs, 30 000 episodes
python main.py \
    --map RDM \
    --alg qmix \
    --federated True \
    --model True \
    --n_agents 3 \
    --sar_reward True \
    --total_episodes 30000 \
    --tag my_run

# Plain QMIX on RBM
python main.py --map RBM --alg qmix --total_episodes 30000 --tag plain

# IQL on RDM
python main.py --map RDM --alg iql --total_episodes 30000 --tag iql_run
```

### Multi-seed runs

```bash
# Run seeds 1–5 sequentially (default)
python main.py --map RBM --alg qmix --federated True --model True \
    --sar_reward True --seeds 1 2 3 4 5 --tag fedmod

# Run seeds in parallel processes
python main.py --map RBM --alg qmix --model True \
    --sar_reward True --seeds 1 2 3 4 5 --parallel True --tag mod
```

### Run all baselines at once

```bash
python run_all.py --map RBM --seeds 1 2 3 4 5
python run_all.py --map RDM --seeds 1 2 3 4 5
```

This runs IQL → MOD → FED → IPPO sequentially then generates the seed evaluation table.

### Shell scripts

```bash
bash run_plain_qmix.sh             # Plain QMIX, seeds 1–3, both maps
bash run_model_qmix.sh             # Model-Aided QMIX
bash run_fed_model_qmix.sh         # Model-Aided FedQMIX
```

---

## Algorithm Variants

| Flag combination | Method label | Description |
|---|---|---|
| `--alg qmix` | **QMIX** | Centralised QMIX, no surrogate |
| `--alg qmix --model True` | **MOD** | Model-Aided QMIX (SLAL surrogate, no federation) |
| `--alg qmix --federated True` | **FED** | FedQMIX (FedAvg, no surrogate) |
| `--alg qmix --federated True --model True` | **FED-MOD** | **Model-Aided FedQMIX** (proposed) |
| `--alg iql` | **IQL** | Independent Q-Learning baseline |

IPPO is run separately via `ippo_custom.py` or through `run_all.py`.

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--map` | `RDM` | Map scenario: `RBM` or `RDM` |
| `--alg` | `qmix` | Algorithm: `qmix` or `iql` |
| `--federated` | `False` | Enable FedAvg aggregation |
| `--model` | `False` | Enable SLAL surrogate channel |
| `--workers` | `3` | Number of federated workers (= UAVs) |
| `--n_agents` | `3` | Number of agents |
| `--total_episodes` | `30000` | Training episodes |
| `--aggregation_period` | `50` | FedAvg aggregation frequency (episodes) |
| `--model_learning_period` | `1000` | SLAL update frequency (episodes) |
| `--sar_reward` | `False` | Use SAR-aligned shaped reward |
| `--lambda_new` | `1.0` | One-time bonus per newly localised victim |
| `--lambda_thr` | `1.0` | Throughput term scaling coefficient |
| `--seed` | `123` | Random seed |
| `--seeds` | `None` | Multi-seed list, e.g. `--seeds 1 2 3 4 5` |
| `--drop_prob` | `0.0` | Federated client drop probability (0 = all clients participate) |
| `--eval_snr_offset_db` | `0.0` | SNR offset at evaluation only (negative = degraded channel) |
| `--device` | `cpu` | Compute device: `cpu` or `cuda` |

---

## Evaluate a Trained Model

```bash
python evaluate.py --map RDM --model True --alg qmix --tag my_run --n_agents 3
```

---

## Diagnostic Experiments

These scripts reproduce the robustness analyses in Section 5 of the paper:

```bash
# Reward-weight (lambda) sensitivity
python run_sensitivity.py

# RF channel-shift robustness (eval_snr_offset_db)
python run_channel_shift.py

# BLE / smartphone hardware heterogeneity
python run_device_heterogeneity.py

# Lambda (reward shaping) sensitivity sweep
python run_lambda_sensitivity.py

# Partial-client FedAvg (random client drops)
python run_partial_fedavg.py

# Non-IID client data variance analysis
python analyse_noniid.py
```

---

## Visualisation

```bash
# Learning curves for a single run
python plot_learning_curves.py

# Compare all methods on both maps
python plot_all_baselines.py

# Reproduce paper figures
python generate_revision_figures.py

# Coverage trajectory plot
python plot_coverage_figure.py

# Time-to-threshold plots
python plot_time_to_threshold_RBM.py
python plot_time_to_threshold_RDM.py
```

Maps are pre-generated as `config/RBM_map.npy` and `config/RDM_map.npy`. To regenerate:

```bash
python generate_city_map.py
```

---

## Results Structure

Each training run saves to `result/<method>/<map><tag>/`:

```
result/
├── qmix/
│   ├── <map><tag>/            # Plain QMIX
│   ├── mod/<map><tag>/        # Model-Aided QMIX
│   ├── fed/<map><tag>/        # FedQMIX
│   └── fed_mod/<map><tag>/    # Model-Aided FedQMIX
├── iql/<map><tag>/
└── ippo/<map><tag>/
```

Each run directory contains:
- `eval_metrics.csv` — per-checkpoint metrics (victims localised, success rate, time to first detection, energy per victim, throughput, PSO localisation error)
- `global_data.npy`, `global_rewards.npy` — episode-level arrays for plotting
- `log_params.txt` — full argument snapshot for reproducibility

---

## Acknowledgements

This work builds on:

- The **Model-aided DRL** framework for single-UAV trajectory design by [Chen et al. (2022)](https://ieeexplore.ieee.org/abstract/document/9685774), which we extended to cooperative multi-UAV victim localisation.
- The **QMIX and IQL** implementations from [starry-sky6688/marl-algorithms](https://github.com/starry-sky6688/marl-algorithms).
- The simulation environment design drew inspiration from [SMAC](https://github.com/oxwhirl/smac).

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{guzey2026uav,
  title     = {UAV Mission Planning for Post-Disaster Victim Localisation
               via Federated Multi-Agent Reinforcement Learning},
  author    = {G{\"u}zey, Alparslan and {\c{C}}if{\c{c}}i, Mehmet Akif
               and Y{\i}ld{\i}r{\i}m, Fazl{\i} and Erdo{\u{g}}an, Arda Ya{\c{s}}ar},
  journal   = {Drones},
  volume    = {10},
  number    = {5},
  pages     = {385},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/drones10050385}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).

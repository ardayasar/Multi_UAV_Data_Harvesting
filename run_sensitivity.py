import os
import json
from types import MethodType
import itertools

import numpy as np
import pandas as pd
import torch

from common.arguments import get_common_args, get_mixer_args
from runner import Runner
from Libs.Environments.DataCollection import DataCollection
from training_procedures import _fill_env_info_into_args, _set_seeds

# ============================================================
# LIGHTWEIGHT SENSITIVITY SETTINGS
# ============================================================

# We keep both maps but use a compact set of cases per map.
MAPS = ["RBM", "RDM"]

# Baseline + 1D perturbations around it.
# Baseline is (p_on = 0.8, snr_thr_db = 0.0, safety_radius = 3.0).
SENS_CASES = [
    {"name": "baseline",      "p_on": 0.8, "snr_thr_db":  0.0, "safety_radius": 3.0},
    {"name": "pon_low",       "p_on": 0.6, "snr_thr_db":  0.0, "safety_radius": 3.0},
    {"name": "snr_low",       "p_on": 0.8, "snr_thr_db": -5.0, "safety_radius": 3.0},
    {"name": "radius_high",   "p_on": 0.8, "snr_thr_db":  0.0, "safety_radius": 6.0},
]

# One seed by default to keep runtime in "tens of minutes".
# If you later want more robustness, change to SEEDS = [1, 2] or [1, 2, 3].
SEEDS = [1]

TOTAL_EPISODES = 100          # strong reduction vs. 400–1000
AGGREGATION_PERIOD = 20       # 5 federated rounds per run
EVALUATE_CYCLE = 20
EVAL_EPOCH = 3                # a few eval episodes to stabilise mean_victims

# We disable model-aided channel re-learning for speed in this sensitivity run.
MODEL_LEARNING_PERIOD = 10**9  # effectively "never"
EVAL_ROUND_EVERY = 1           # log every round; eval is cheap now

DEVICE = "cpu"
N_AGENTS = 3
N_WORKERS = 1                  # 1 worker is fastest on a single laptop

# ============================================================
# 1) Environment wrapper: inject p_on, snr_thr_db, safety_radius
# ============================================================

def make_env_with_sensitivity(args, params, p_on, snr_thr_db, safety_radius_cells):
    """
    Wrap DataCollection to:
      - inject p_on into the comm_step (drop packets with prob 1 - p_on),
      - set SNR-like thresholds (if the env uses these fields),
      - enforce a minimum inter-UAV spacing via safety_radius.
    """
    env = DataCollection(args, params=params, learning_channel_model=None)

    # --- SNR threshold injection (only effective if env uses these attrs) ----
    env.device_sight_range = float(snr_thr_db)
    env.uav_sight_range = float(snr_thr_db)

    # --- p_on: drop measurements with prob (1 - p_on) ------------------------
    base_comm_step = env.comm_step

    def comm_step_with_pon(self, collected_meas, d_id, model):
        """
        Bernoulli on/off for each BLE "packet". When off, we pass a zeroed
        measurement to base_comm_step so downstream code keeps its shape.
        """
        if p_on < 1.0 and (np.random.rand() > p_on):
            # "Zero" measurement preserving structure as best as possible.
            if model:
                try:
                    zero_meas = np.zeros_like(collected_meas, dtype=float)
                except Exception:
                    zero_meas = 0.0
                return base_comm_step(zero_meas, d_id, model)

            ch = getattr(collected_meas, "ch_capacity", collected_meas)
            try:
                zero_ch = np.zeros_like(ch, dtype=float)
            except Exception:
                zero_ch = 0.0

            class Dummy:
                pass

            dm = Dummy()
            dm.ch_capacity = zero_ch
            return base_comm_step(dm, d_id, model)

        return base_comm_step(collected_meas, d_id, model)

    env.comm_step = MethodType(comm_step_with_pon, env)

    # --- safety_radius: enforce min spacing between UAVs ---------------------
    safety_radius_m = float(safety_radius_cells) * env.step_size
    base_get_avail_agent_actions = env.get_avail_agent_actions

    def get_avail_agent_actions_safe(self, agent_id):
        """
        Wrap env.get_avail_agent_actions to prune actions that would move
        UAV 'agent_id' closer than safety_radius_m to any other UAV.
        """
        avail_actions = base_get_avail_agent_actions(agent_id)

        if safety_radius_m <= 0.0:
            return avail_actions

        agent = self.get_agent_by_id(agent_id)

        # Actions 1..4 are the 4 moves (0 is "stay" / no-op).
        for act in (1, 2, 3, 4):
            if avail_actions[act] != 1:
                continue

            pos_index = agent.current_pose_index + agent.action_space[act]
            cand_pose = self.index_to_pose(pos_index)

            for other_id, other in enumerate(self.agents):
                if other_id == agent_id:
                    continue
                if np.linalg.norm(other.current_pose - cand_pose) < safety_radius_m:
                    avail_actions[act] = 0
                    break

        # If we pruned everything, fall back to original to avoid deadlock.
        if sum(avail_actions) == 0:
            return base_get_avail_agent_actions(agent_id)
        return avail_actions

    env.get_avail_agent_actions = MethodType(get_avail_agent_actions_safe, env)

    # Uncomment to debug that the knobs are actually applied:
    # print("[SENS-CHECK]",
    #       f"p_on={p_on}, snr_thr_db={snr_thr_db},",
    #       f"safety_radius_cells={safety_radius_cells}, safety_radius_m={safety_radius_m}")

    return env

# ============================================================
# 2) Extra evaluation for sensitivity metrics
# ============================================================

def evaluate_energy_and_victims(runner, args, n_eval=None):
    """
    Run a few eval episodes and compute:
      - mean_victims: average # of distinct victims localised
      - energy_per_victim: total energy / mean_victims (NaN if no victims)
    """
    if n_eval is None:
        n_eval = getattr(args, "evaluate_epoch", 1)

    total_victims = 0.0
    total_energy = 0.0

    for ep in range(n_eval):
        _, _, _, collected_data, _, _, _ = runner.rolloutWorker.generate_episode(
            ep, evaluate=True, model=args.model
        )
        total_victims += float(collected_data)
        total_energy += float(getattr(runner.env, "energy_used", 0.0))

    mean_victims = total_victims / max(n_eval, 1)

    if mean_victims <= 1e-9:
        energy_per_victim = float("nan")  # will render as '--' in LaTeX
    else:
        energy_per_victim = total_energy / mean_victims

    return mean_victims, energy_per_victim

# ============================================================
# 3) FedQMIX training for a single (map, case, seed)
# ============================================================

def train_fedqmix_one_setting(args, params, p_on, snr_thr_db, safety_radius_cells):
    n_workers = args.workers
    save_path = os.path.join(args.result_dir, "qmix", args.map + args.tag)
    os.makedirs(save_path, exist_ok=True)

    envs = [
        make_env_with_sensitivity(args, params, p_on, snr_thr_db, safety_radius_cells)
        for _ in range(n_workers)
    ]
    _fill_env_info_into_args(envs[0], args)

    # runners
    runners = []
    # simple epsilon schedule across workers (if N_WORKERS>1)
    eps_list = [0.3, 0.1, 0.05]

    for i in range(n_workers):
        r = Runner(envs[i], args)
        r.rolloutWorker.epsilon = r.rolloutWorker.min_epsilon = eps_list[i % len(eps_list)]
        runners.append(r)

    # sync networks across workers
    for r in runners[1:]:
        r.agents.policy.eval_qmix_net.load_state_dict(
            runners[0].agents.policy.eval_qmix_net.state_dict()
        )
        r.agents.policy.target_qmix_net.load_state_dict(
            runners[0].agents.policy.target_qmix_net.state_dict()
        )
        r.agents.policy.eval_rnn.load_state_dict(
            runners[0].agents.policy.eval_rnn.state_dict()
        )
        r.agents.policy.target_rnn.load_state_dict(
            runners[0].agents.policy.target_rnn.state_dict()
        )

    # NOTE: args.model is False for this script; we skip SLAL/PSO for speed.

    _set_seeds(args.seed)

    total_eps = args.total_episodes
    agg_period = args.aggregation_period

    data_eval_lists = [[] for _ in range(n_workers)]
    reward_eval_lists = [[] for _ in range(n_workers)]
    data_train_lists = [[] for _ in range(n_workers)]
    reward_train_lists = [[] for _ in range(n_workers)]
    global_data, global_rewards = [], []

    # initial evaluation
    d0, r0, *_ = runners[0].evaluate(model=args.model)
    global_data.append(d0)
    global_rewards.append(r0)

    train_steps = [0] * n_workers
    eval_steps = [-1] * n_workers
    episode_cnt = [0] * n_workers
    best_ep_data = [0] * n_workers

    num_rounds = max(1, total_eps // agg_period)

    for round_k in range(num_rounds):
        # Each runner performs its local FedQMIX block.
        pq, prnn = [], []
        for wi, r in enumerate(runners):
            ts, es, ep, best = r.federated_run(
                wi,
                train_steps[wi],
                eval_steps[wi],
                episode_cnt[wi],
                data_train_lists[wi],
                reward_train_lists[wi],
                data_eval_lists[wi],
                reward_eval_lists[wi],
                best_ep_data[wi],
                model=args.model,
            )
            train_steps[wi], eval_steps[wi], episode_cnt[wi], best_ep_data[wi] = ts, es, ep, best

            pq.append(
                torch.nn.utils.parameters_to_vector(
                    r.agents.policy.eval_qmix_net.parameters()
                )
            )
            prnn.append(
                torch.nn.utils.parameters_to_vector(
                    r.agents.policy.eval_rnn.parameters()
                )
            )

        # FedAvg across workers
        q_agg = sum(pq) / n_workers
        rnn_agg = sum(prnn) / n_workers
        for r in runners:
            torch.nn.utils.vector_to_parameters(
                q_agg, r.agents.policy.eval_qmix_net.parameters()
            )
            torch.nn.utils.vector_to_parameters(
                rnn_agg, r.agents.policy.eval_rnn.parameters()
            )

        # evaluation at every round (cheap when args.model=False)
        if (round_k % EVAL_ROUND_EVERY == 0) or (round_k == num_rounds - 1):
            gd, gr, *_ = runners[0].evaluate(model=args.model)
            global_data.append(gd)
            global_rewards.append(gr)
            print(
                f"[Sens/FedQMIX] map={args.map} p_on={p_on:.2f} "
                f"snr={snr_thr_db:.1f} R={safety_radius_cells:.1f} "
                f"seed={args.seed} round {round_k+1:3d}/{num_rounds:3d}  gd={gd:.2f}"
            )

    # Save global learning curves (optional)
    np.save(os.path.join(save_path, "global_data.npy"), np.asarray(global_data))
    np.save(os.path.join(save_path, "global_rewards.npy"), np.asarray(global_rewards))

    # Final sensitivity metrics
    mean_victims, energy_per_victim = evaluate_energy_and_victims(
        runners[0], args, n_eval=args.evaluate_epoch
    )

    info = {
        "map": args.map,
        "seed": int(args.seed),
        "p_on": float(p_on),
        "snr_thr_db": float(snr_thr_db),
        "safety_radius": float(safety_radius_cells),
        "mean_victims": float(mean_victims),
        "energy_per_victim": float(energy_per_victim),
        "tag": args.tag,
        "total_episodes": int(args.total_episodes),
        "aggregation_period": int(args.aggregation_period),
        "eval_round_every": int(EVAL_ROUND_EVERY),
        "model_learning_period": int(args.model_learning_period),
        "workers": int(args.workers),
    }

    with open(os.path.join(save_path, "sens_log_params.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info

# ============================================================
# 4) Top-level grid over maps × cases × seeds
# ============================================================

def main():
    results = []

    for map_name in MAPS:
        if map_name == "RBM":
            from config.RBM_define import params
        else:
            from config.RDM_define import params

        for case in SENS_CASES:
            p_on = case["p_on"]
            snr_thr_db = case["snr_thr_db"]
            safety_radius = case["safety_radius"]

            for seed in SEEDS:
                args = get_common_args()
                args.alg = "qmix"
                args = get_mixer_args(args)

                args.map = map_name
                args.federated = True
                args.model = False           # disable surrogate re-learning for speed
                args.n_agents = N_AGENTS
                args.total_episodes = TOTAL_EPISODES
                args.evaluate_cycle = EVALUATE_CYCLE
                args.aggregation_period = AGGREGATION_PERIOD
                args.model_learning_period = MODEL_LEARNING_PERIOD
                args.evaluate_epoch = EVAL_EPOCH
                args.device = DEVICE
                args.seed = seed
                args.workers = N_WORKERS

                # helpful to separate sensitivity outputs from main training
                if not hasattr(args, "result_dir") or args.result_dir == "":
                    args.result_dir = "result"

                # stable tag naming (handles negative SNR nicely)
                snr_tag = f"m{abs(int(snr_thr_db))}" if snr_thr_db < 0 else f"{int(snr_thr_db)}"
                tag = (
                    f"_sens_{case['name']}"
                    f"_pon{int(p_on*10):02d}"
                    f"_snr{snr_tag}_rad{int(safety_radius)}_s{seed}"
                )
                args.tag = tag

                print(
                    f"\n▶ MAP={map_name}, case={case['name']}, "
                    f"p_on={p_on}, snr={snr_thr_db} dB, "
                    f"R={safety_radius}, seed={seed}  → tag={tag}"
                )

                info = train_fedqmix_one_setting(
                    args, params, p_on, snr_thr_db, safety_radius
                )
                results.append(info)

    # ------------------- aggregate results → CSV & LaTeX ----------------------
    os.makedirs("result", exist_ok=True)
    df = pd.DataFrame(results)
    out_csv = os.path.join("result", "sensitivity_results.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved raw sensitivity results to:", out_csv)

    # Build Table C summary (aggregate across seeds).
    group_cols = ["map", "p_on", "snr_thr_db", "safety_radius"]

    summary = (
        df.groupby(group_cols)
        .agg(
            n_seeds=("seed", "count"),
            mean_victims_mean=("mean_victims", "mean"),
            mean_victims_std=("mean_victims", "std"),
            energy_per_victim_mean=("energy_per_victim", "mean"),
            energy_per_victim_std=("energy_per_victim", "std"),
        )
        .reset_index()
    )

    # nicer formatting for LaTeX
    def fmt_mean_std(mean_val, std_val):
        if pd.isna(mean_val):
            return r"--"
        std_val = 0.0 if pd.isna(std_val) else std_val
        return f"{mean_val:.2f} $\\pm$ {std_val:.2f}"

    summary["mean_victims"] = summary.apply(
        lambda r: fmt_mean_std(r["mean_victims_mean"], r["mean_victims_std"]),
        axis=1,
    )
    summary["energy_per_victim"] = summary.apply(
        lambda r: fmt_mean_std(
            r["energy_per_victim_mean"], r["energy_per_victim_std"]
        ),
        axis=1,
    )

    tableC = summary[group_cols + ["n_seeds", "mean_victims", "energy_per_victim"]]

    out_csv2 = os.path.join("result", "sensitivity_results_summary.csv")
    tableC.to_csv(out_csv2, index=False)

    out_tex = os.path.join("result", "tableC_sensitivity.tex")
    latex_str = tableC.to_latex(index=False, escape=False)
    with open(out_tex, "w") as f:
        f.write(latex_str)

    print("Saved Table C summary to:", out_csv2)
    print("Saved LaTeX Table C to:", out_tex)


if __name__ == "__main__":
    main()

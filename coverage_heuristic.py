#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coverage_heuristic.py
Coverage / lawnmower heuristic baseline for SARMultiAgentEnvBasic.

Outputs (per seed):
  result/coverage/<MAP>_sX/global_data.npy   # victims_found per episode
  result/coverage/<MAP>_sX/metrics.csv       # per-episode metrics

Reviewer-aligned changes:
- Randomize "victim on/off" each episode by deactivating devices (remaining_data = 0).
- Pre-compute a lawnmower sweep path for each UAV, then execute it without learning.
- Record victims_found, time-to-first-detection, energy, and completion-to-active-victims metrics.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import csv
import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np

from Libs.Environments.IPPOEnv import SARMultiAgentEnvBasic
from common.arguments import get_common_args
from config.RBM_define import set_env as rbm_set_env
from config.RDM_define import set_env as rdm_set_env


# Action indices from AgentModel.action_space:
# 0 hover, 1 north(+y), 2 west(-x), 3 south(-y), 4 east(+x), 5 no-op
A_HOVER = 0
A_NORTH = 1
A_WEST  = 2
A_SOUTH = 3
A_EAST  = 4
A_NOOP  = 5


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_env(map_name: str, alpha: float = 0.1) -> SARMultiAgentEnvBasic:
    """
    IMPORTANT: get_common_args() parses sys.argv. We must not let it see this script's CLI flags,
    otherwise it will crash due to unknown args.
    """
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = get_common_args()
    finally:
        sys.argv = old_argv

    args.map = map_name
    params = rbm_set_env(args) if map_name == "RBM" else rdm_set_env(args)
    env = SARMultiAgentEnvBasic(args=args, params=params, alpha=alpha)
    return env


def split_y_stripes(y_min: int, y_max: int, n_agents: int):
    """Split inclusive [y_min, y_max] into n_agents contiguous stripes."""
    ys = list(range(y_min, y_max + 1))
    if not ys:
        return [(0, -1)] * n_agents
    chunks = np.array_split(np.array(ys), n_agents)
    stripes = []
    for c in chunks:
        if len(c) == 0:
            stripes.append((0, -1))
        else:
            stripes.append((int(c[0]), int(c[-1])))
    return stripes


def make_lawnmower_path(x_min: int, x_max: int, y_start: int, y_end: int, row_stride: int = 1):
    """
    Return list of (x,y) covering rows y_start..y_end in lawnmower pattern (inclusive),
    optionally skipping rows with row_stride (>1).
    """
    if y_end < y_start:
        return []
    path = []
    xs_fwd = list(range(x_min, x_max + 1))
    xs_rev = list(reversed(xs_fwd))

    y_vals = list(range(y_start, y_end + 1, max(1, int(row_stride))))
    for i, y in enumerate(y_vals):
        xs = xs_fwd if (i % 2 == 0) else xs_rev
        for x in xs:
            path.append((x, y))
    return path


def choose_action_to_target(cur_xy, tgt_xy, avail_agent_actions):
    """
    Greedy 1-step move towards target, respecting available actions.
    cur_xy, tgt_xy: (x,y) ints in index space.
    avail_agent_actions: list length 6 of {0,1}.
    """
    cx, cy = cur_xy
    tx, ty = tgt_xy
    dx = tx - cx
    dy = ty - cy

    candidates = []
    if abs(dx) >= abs(dy):
        if dx > 0: candidates.append(A_EAST)
        if dx < 0: candidates.append(A_WEST)
        if dy > 0: candidates.append(A_NORTH)
        if dy < 0: candidates.append(A_SOUTH)
    else:
        if dy > 0: candidates.append(A_NORTH)
        if dy < 0: candidates.append(A_SOUTH)
        if dx > 0: candidates.append(A_EAST)
        if dx < 0: candidates.append(A_WEST)

    for a in candidates:
        if avail_agent_actions[a] == 1:
            return a

    if avail_agent_actions[A_HOVER] == 1:
        return A_HOVER

    for a, ok in enumerate(avail_agent_actions):
        if ok == 1:
            return a

    return A_NOOP


def _done_flag(dones):
    """Robustly interpret dones returned by env.step()."""
    if isinstance(dones, (list, tuple, np.ndarray)):
        return bool(dones[0])
    return bool(dones)


def randomize_victim_activity(base_env, rng: np.random.RandomState,
                              mode: str = "bernoulli",
                              p_on: float = 0.8,
                              k_active: int = None) -> Tuple[np.ndarray, int]:
    """
    Victim on/off randomization per episode:
    - OFF victim => device.remaining_data = 0 (device.depleted becomes True)
    - ON victim  => device.remaining_data = device.data

    Then refresh base_env.device_snr and base_env.device_access accordingly.
    Returns (active_mask, active_count).
    """
    n = int(base_env.n_devices)

    if mode == "fixed_k":
        if k_active is None:
            raise ValueError("k_active must be provided when mode='fixed_k'")
        k = int(max(0, min(n, int(k_active))))
        active = np.zeros(n, dtype=bool)
        if k > 0:
            idx = rng.choice(n, size=k, replace=False)
            active[idx] = True
    else:
        p = float(max(0.0, min(1.0, p_on)))
        active = (rng.rand(n) < p)
        if not active.any():
            active[rng.randint(0, n)] = True

    for i, dev in enumerate(base_env.device_list.devices):
        if active[i]:
            dev.remaining_data = float(dev.data)
        else:
            dev.remaining_data = 0.0

    base_env.device_snr = base_env.get_agents_device_snr(model=False)
    base_env.device_access = base_env.get_avail_devices()
    return active, int(active.sum())


def run_episode(env: SARMultiAgentEnvBasic,
                paths,
                rng: np.random.RandomState,
                activity_mode: str,
                p_on: float,
                k_active: int,
                stop_on_all_found: bool = True):
    """
    Run one episode following given per-agent paths.

    Returns:
      victims_found, t_first, energy_used_total, steps, active_count, move_count, e_first
    """
    _ = env.reset()
    base_env = getattr(env, "base_env", None)
    if base_env is None:
        raise AttributeError("SARMultiAgentEnvBasic has no attribute 'base_env'.")

    # Defensive: ensure empty at episode start (prevents carry-over across episodes)
    if hasattr(env, "detected_victims"):
        try:
            env.detected_victims.clear()
        except Exception:
            env.detected_victims = []

    active_mask, active_count = randomize_victim_activity(
        base_env, rng,
        mode=activity_mode,
        p_on=p_on,
        k_active=k_active
    )

    # reset accounting (tracked inside DataCollection)
    if hasattr(base_env, "energy_used"):
        base_env.energy_used = 0.0
    if hasattr(base_env, "move_count"):
        base_env.move_count = 0

    # --- KEY FIX: define "first detection" as first detection of an ACTIVE victim
    #               that was NOT already detectable at t=0 (spawn communication range).
    active_ids = set(np.where(active_mask)[0].tolist())

    # base_env.device_snr is already refreshed in randomize_victim_activity()
    # A victim is "detectable at t=0" if ANY UAV has SNR >= sight_range to it.
    detectable0 = set()
    for d in active_ids:
        if np.any(base_env.device_snr[:, d] >= base_env.device_sight_range):
            detectable0.add(d)

    # We measure time/energy to first "non-trivial" detection (requires actual search/motion)
    target_ids = active_ids - detectable0
    # Fallback: if all active victims are already detectable at spawn, just measure first detection normally
    if len(target_ids) == 0:
        target_ids = active_ids

    n_agents = env.n_agents
    t_first = None
    e_first = None
    steps = 0
    path_ptrs = [0] * n_agents

    done_all = False
    max_steps = int(getattr(base_env, "episode_limit", 10_000))

    prev_detected = set()

    while not done_all and steps < max_steps:
        avail_all = base_env.get_avail_actions()

        actions = []
        for a in range(n_agents):
            aa = avail_all[a]

            if not paths[a]:
                act = A_NOOP if aa[A_NOOP] == 1 else (A_HOVER if aa[A_HOVER] == 1 else int(np.argmax(aa)))
                actions.append(act)
                continue

            cur_idx = base_env.agents[a].current_pose_index
            cx, cy = int(cur_idx[0, 0]), int(cur_idx[0, 1])

            while path_ptrs[a] < len(paths[a]) and (cx, cy) == paths[a][path_ptrs[a]]:
                path_ptrs[a] += 1

            if path_ptrs[a] >= len(paths[a]):
                act = A_NOOP if aa[A_NOOP] == 1 else (A_HOVER if aa[A_HOVER] == 1 else int(np.argmax(aa)))
                actions.append(act)
                continue

            tx, ty = paths[a][path_ptrs[a]]
            actions.append(choose_action_to_target((cx, cy), (tx, ty), aa))

        _, _, dones, _ = env.step(actions)
        done_all = _done_flag(dones)
        steps += 1

        # update detection-based metrics
        if hasattr(env, "detected_victims"):
            cur_detected = set(env.detected_victims)
            new_detected = cur_detected - prev_detected

            # First time we detect ANY victim in target_ids
            if t_first is None and len(new_detected.intersection(target_ids)) > 0:
                t_first = steps
                e_first = float(getattr(base_env, "energy_used", 0.0))

            prev_detected = cur_detected

            # EARLY STOP: once all active victims are found, end mission
            if stop_on_all_found and active_count > 0 and len(cur_detected) >= active_count:
                break

    victims_found = float(len(env.detected_victims)) if hasattr(env, "detected_victims") else float(env.get_episode_score())
    if t_first is None:
        t_first = steps
    if e_first is None:
        e_first = float(getattr(base_env, "energy_used", 0.0))

    energy_used = float(getattr(base_env, "energy_used", 0.0))
    move_count = int(getattr(base_env, "move_count", 0))

    return victims_found, int(t_first), energy_used, int(steps), int(active_count), int(move_count), float(e_first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="RBM", choices=["RBM", "RDM"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable early stop when all active victims are detected.")

    parser.add_argument("--episodes", type=int, default=200, help="episodes to run (no learning)")
    parser.add_argument("--total_episodes", type=int, default=None, help="alias for --episodes")
    parser.add_argument("--n_episodes", type=int, default=None, help="alias for --episodes")

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--out_root", type=str, default="result/coverage")

    parser.add_argument("--margin_cells", type=int, default=0)
    parser.add_argument("--safety_radius_cells", type=int, default=0)

    parser.add_argument("--activity_mode", type=str, default="bernoulli",
                        choices=["bernoulli", "fixed_k"])
    parser.add_argument("--p_on", type=float, default=0.8)
    parser.add_argument("--k_active", type=int, default=None)

    args = parser.parse_args()

    if args.total_episodes is not None:
        args.episodes = args.total_episodes
    if args.n_episodes is not None:
        args.episodes = args.n_episodes

    stop_on_all_found = (not args.no_early_stop)

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    env = build_env(args.map, alpha=args.alpha)
    base_env = env.base_env
    n_agents = env.n_agents

    x_min = int(args.margin_cells)
    y_min = int(args.margin_cells)
    x_max = int(base_env.max_index_x) - int(args.margin_cells)
    y_max = int(base_env.max_index_y) - int(args.margin_cells)
    x_max = max(x_min, x_max)
    y_max = max(y_min, y_max)

    row_stride = 2 * int(max(0, args.safety_radius_cells)) + 1

    stripes = split_y_stripes(y_min, y_max, n_agents)
    paths = []
    for a in range(n_agents):
        ys, ye = stripes[a]
        paths.append(make_lawnmower_path(x_min, x_max, ys, ye, row_stride=row_stride))

    out_dir = Path(args.out_root) / f"{args.map}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    victims_curve = []
    rows = []

    for ep in range(1, int(args.episodes) + 1):
        v, t_first, e_used, steps, active_count, move_count, e_first = run_episode(
            env, paths, rng,
            activity_mode=args.activity_mode,
            p_on=args.p_on,
            k_active=args.k_active,
            stop_on_all_found=stop_on_all_found
        )

        completion = 1 if (active_count > 0 and v >= active_count) else 0
        completion_ratio = (v / active_count) if active_count > 0 else 0.0

        victims_curve.append(float(v))
        rows.append({
            "episode": ep,
            "victims_found": float(v),
            "time_to_first_detection": int(t_first),
            "energy_used": float(e_used),
            "energy_to_first_detection": float(e_first),
            "steps": int(steps),
            "move_steps": int(move_count),
            "active_victims": int(active_count),
            "completion": int(completion),
            "completion_ratio": float(completion_ratio),
        })

        if ep % 20 == 0:
            print(f"[{args.map} s{args.seed}] ep={ep:4d} victims={v:.1f} "
                  f"t_first={t_first} energy={e_used:.1f} e_first={e_first:.1f} "
                  f"moves={move_count} active={active_count}/{base_env.n_devices} "
                  f"comp={completion} ({completion_ratio:.2f})")

    np.save(out_dir / "global_data.npy", np.array(victims_curve, dtype=np.float32))

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "victims_found",
                "time_to_first_detection",
                "energy_used",
                "energy_to_first_detection",
                "steps",
                "move_steps",
                "active_victims",
                "completion",
                "completion_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Coverage heuristic saved:")
    print(f"  - {out_dir/'global_data.npy'}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
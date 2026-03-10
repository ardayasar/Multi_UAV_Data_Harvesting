# ippo_custom.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Libs.Environments.IPPOEnv import SARMultiAgentEnvBasic
from common.arguments import get_common_args
from config.RBM_define import set_env as rbm_set_env
from config.RDM_define import set_env as rdm_set_env

# --------------------- Simple switches --------------------- #
MAP_NAME = "RBM"          # "RBM" or "RDM"
ALPHA = 0.1               # reward penalty weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPISODES = 1000       # total training episodes
BATCH_SIZE = 4096         # timesteps before each PPO update
GAMMA = 0.99
LAMBDA = 0.95             # GAE lambda
LR = 3e-4
CLIP_EPS = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 512
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5

EVAL_EVERY = 10           # evaluate every 10 episodes
N_EVAL_EPISODES = 5       # eval rollouts per checkpoint


# --------------------- Utilities --------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------- Environment init --------------------- #
def build_env(map_name: str = "RBM", alpha: float = 0.1) -> SARMultiAgentEnvBasic:
    args = get_common_args()
    args.map = map_name

    if map_name == "RBM":
        params = rbm_set_env(args)
    else:
        params = rdm_set_env(args)

    env = SARMultiAgentEnvBasic(args=args, params=params, alpha=alpha)
    return env


# --------------------- Actor-Critic network --------------------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], n_actions),
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, obs: torch.Tensor):
        logits = self.policy_net(obs)
        value = self.value_net(obs).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: 1D tensor [obs_dim]
        deterministic: if True, use argmax; else sample (used for training).
        """
        logits, value = self.forward(obs.unsqueeze(0))  # [1, obs_dim]
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)


# --------------------- Rollout buffer --------------------- #
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    next_advantage = 0.0
    next_return = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        returns[t] = rewards[t] + gamma * next_return * mask

        td_error = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = td_error + gamma * lam * next_advantage * mask

        next_return = returns[t]
        next_value = values[t]
        next_advantage = advantages[t]

    return advantages, returns


def ppo_update(model: ActorCritic,
               optimizer: optim.Optimizer,
               buffer: RolloutBuffer):
    obs = torch.stack(buffer.obs).to(DEVICE)
    actions = torch.stack(buffer.actions).to(DEVICE)
    old_log_probs = torch.stack(buffer.log_probs).to(DEVICE)
    rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=DEVICE)
    dones = torch.tensor(buffer.dones, dtype=torch.float32, device=DEVICE)
    values = torch.stack(buffer.values).to(DEVICE)

    with torch.no_grad():
        advantages_np, returns_np = compute_gae(
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            values.cpu().numpy(),
            gamma=GAMMA,
            lam=LAMBDA,
        )
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = obs.size(0)
    for _ in range(UPDATE_EPOCHS):
        idx = torch.randperm(N)
        for start in range(0, N, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_idx = idx[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            logits, values_pred = model(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(mb_actions)

            ratio = torch.exp(log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred, mb_returns)
            entropy_loss = dist.entropy().mean()

            loss = policy_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


# --------------------- Evaluation --------------------- #
def evaluate_policy(map_name: str, model: ActorCritic) -> float:
    """
    Run N_EVAL_EPISODES greedy rollouts and return mean number of
    distinct victims localised (|eval_env.detected_victims|).
    """
    eval_env = build_env(map_name, ALPHA)
    model.eval()
    scores = []

    with torch.no_grad():
        for _ in range(N_EVAL_EPISODES):
            obs = eval_env.reset()
            done_all = False
            while not done_all:
                actions = []
                for a in range(eval_env.n_agents):
                    obs_tensor = torch.from_numpy(obs[a]).float().to(DEVICE)
                    action, _, _ = model.act(obs_tensor, deterministic=True)
                    actions.append(int(action.item()))
                obs, rewards, dones, _ = eval_env.step(actions)
                done_all = bool(dones[0])
            # metric = total distinct victims localised in this rollout
            scores.append(float(len(eval_env.detected_victims)))

    model.train()
    return float(np.mean(scores))


# --------------------- Main training loop --------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (defines IPPO-sX)")
    parser.add_argument("--map", type=str, default=MAP_NAME,
                        choices=["RBM", "RDM"],
                        help="map name RBM or RDM")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES,
                        help="number of training episodes")
    args = parser.parse_args()

    seed = args.seed
    map_name = args.map
    num_episodes = args.episodes

    set_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"Using device: {DEVICE}")
    print(f"Running IPPO on map={map_name}, seed={seed}")

    print("Building environment...")
    env = build_env(map_name, ALPHA)
    n_agents = env.n_agents
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    episode_limit = env.episode_limit

    print(f"Env ready: {map_name}, agents={n_agents}, "
          f"obs_dim={obs_dim}, actions={n_actions}, ep_limit={episode_limit}")

    model = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer()
    reward_history = []
    eval_scores = []   # list of (episode, mean_victims)

    total_steps = 0

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done_all = False
        ep_reward = 0.0

        while not done_all:
            actions = []
            for a in range(n_agents):
                obs_tensor = torch.from_numpy(obs[a]).float().to(DEVICE)
                action, log_prob, value = model.act(obs_tensor, deterministic=False)

                actions.append(int(action.item()))
                buffer.obs.append(obs_tensor.detach())
                buffer.actions.append(action.detach())
                buffer.log_probs.append(log_prob.detach())
                buffer.values.append(value.detach())

            next_obs, rewards, dones, _ = env.step(actions)

            r = float(rewards[0])           # shared reward
            done_all = bool(dones[0])

            for _ in range(n_agents):
                buffer.rewards.append(r)
                buffer.dones.append(1.0 if done_all else 0.0)

            ep_reward += r
            total_steps += 1
            obs = next_obs

            if len(buffer) >= BATCH_SIZE:
                ppo_update(model, optimizer, buffer)
                buffer.clear()

        reward_history.append(ep_reward)

        if ep % 10 == 0:
            avg_last_50 = np.mean(reward_history[-50:])
            print(
                f"[Episode {ep:4d}] "
                f"ep_reward={ep_reward:.3f} | "
                f"avg_last_50={avg_last_50:.3f} | "
                f"total_steps={total_steps}"
            )

        # --- periodic evaluation ---
        if ep % EVAL_EVERY == 0:
            mean_victims = evaluate_policy(map_name, model)
            eval_scores.append((ep, mean_victims))
            print(f"    [Eval @ ep {ep}] mean distinct victims = {mean_victims:.2f}")

    # save curves
    np.save(f"ippo_rewards_{map_name}_s{seed}.npy", np.array(reward_history))
    np.save(f"ippo_eval_{map_name}_s{seed}.npy", np.array(eval_scores))

    # write eval_metrics.csv compatible with generate_seed_table.py
    csv_dir = os.path.join("result", "ippo", f"{map_name}_s{seed}")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "eval_metrics.csv")
    fieldnames = [
        "method", "map_name", "seed", "train_episode",
        "victims_localised_mean", "victims_localised_std",
        "time_to_first_detection_mean", "time_to_first_detection_std",
        "energy_per_victim_mean", "energy_per_victim_std",
        "throughput_mean", "reward_mean",
        "map", "alg", "tag", "model", "federated",
        "eval_index", "bytes_per_round",
        "victims_found_mean", "victims_found_std",
        "energy_used_mean", "energy_used_std",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (ep, mean_v) in enumerate(eval_scores):
            writer.writerow({
                "method": "IPPO", "map_name": map_name, "seed": seed,
                "train_episode": ep, "victims_localised_mean": mean_v,
                "victims_localised_std": 0.0,
                "time_to_first_detection_mean": "", "time_to_first_detection_std": "",
                "energy_per_victim_mean": "", "energy_per_victim_std": "",
                "throughput_mean": "", "reward_mean": "",
                "map": map_name, "alg": "ippo", "tag": f"_s{seed}",
                "model": 0, "federated": 0,
                "eval_index": idx, "bytes_per_round": 0,
                "victims_found_mean": mean_v, "victims_found_std": 0.0,
                "energy_used_mean": "", "energy_used_std": "",
            })
    print(f"Saved eval_metrics.csv → {csv_path}")
    print("Training finished.")

    # ---- print last 5 evals in table order ----
    if len(eval_scores) >= 5:
        last5 = eval_scores[-5:]  # chronological: oldest .. newest
        episodes_last5 = [e for (e, _) in last5]
        vals = [v for (_, v) in last5]
        eval1, eval2, eval3, eval4, eval5 = vals  # Eval-1 earliest, Eval-5 latest

        print("\n=== IPPO summary for LaTeX ===")
        print(f"Map: {map_name}, Seed: {seed}")
        print("Episodes of Eval-1..Eval-5:", episodes_last5)
        print("Row order (Eval-5 Eval-4 Eval-3 Eval-2 Eval-1):")
        print(f"{eval5:.1f} {eval4:.1f} {eval3:.1f} {eval2:.1f} {eval1:.1f}")
    else:
        print("Not enough evaluation checkpoints collected to compute Eval-1..Eval-5.")


if __name__ == "__main__":
    main()
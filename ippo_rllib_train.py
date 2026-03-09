# ippo_rllib_train.py
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from Libs.Environments.RLLibWrapper import SARMultiAgentEnv
from common.arguments import get_common_args
from config.RBM_define import set_env as rbm_set_env
from config.RDM_define import set_env as rdm_set_env

# ---- Simple switches (edit these when needed) ----
MAP_NAME = "RBM"    # "RBM" or "RDM"
ALPHA = 0.1         # reward penalty weight for energy usage (in wrapper)
MAX_ITERS = 400     # PPO training iterations


def build_args_and_params(map_name: str = "RBM"):
    """
    Reuse EXACTLY the same logic you currently use
    in main.py / training_procedures.py to construct:
      - args (with args.n_agents, args.map, etc.)
      - params (city, radio_ch_model, device_position, ...)
    """
    args = get_common_args()
    args.map = map_name

    if map_name == "RBM":
        params = rbm_set_env(args)
    else:
        params = rdm_set_env(args)

    return args, params


def make_env_config(map_name: str = "RBM", alpha: float = 0.1):
    """Bundle everything the wrapper needs into a single config dict."""
    args, params = build_args_and_params(map_name)
    return {"args": args, "params": params, "alpha": alpha}


def env_creator(env_config):
    """Factory for RLlib: wraps your DataCollection env in SARMultiAgentEnv."""
    return SARMultiAgentEnv(env_config)


def main():
    print(">>> [IPPO] Starting Ray.init()")
    ray.init(local_mode=True,
             include_dashboard=False,
             ignore_reinit_error=True)
    print(">>> [IPPO] Ray.init() done")

    # Build the environment configuration once
    env_config = make_env_config(map_name=MAP_NAME, alpha=ALPHA)
    print(f">>> [IPPO] Env config built for map={MAP_NAME}")

    # Register the env so RLlib workers can create it
    register_env("sar_env", env_creator)
    print(">>> [IPPO] Environment 'sar_env' registered in RLlib")

    # Create a dummy env locally to inspect spaces & basic sanity
    env = SARMultiAgentEnv(env_config)
    print(">>> [IPPO] Dummy env created")
    obs_space, act_space = env.observation_space, env.action_space
    n_agents = env.n_agents
    print(f">>> [IPPO] n_agents={n_agents}, obs_dim={obs_space.shape}, n_actions={act_space.n}")

    # One independent PPO policy per agent (IPPO)
    policies = {
        f"agent_{i}": (None, obs_space, act_space, {})
        for i in range(n_agents)
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        # independent PPO = each agent gets its own policy
        return agent_id

    print(">>> [IPPO] Building PPOConfig")
    config = (
        PPOConfig()
        .environment(env="sar_env", env_config=env_config)
        .framework("torch")
        .rollouts(
            num_rollout_workers=0,                 # keep single-process while debugging
            rollout_fragment_length=env.episode_limit,
        )
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000,
            model={"fcnet_hiddens": [128, 128]},
            entropy_coeff=0.01,
            clip_param=0.2,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    print(">>> [IPPO] Building PPO algorithm instance")
    algo = config.build()
    print(">>> [IPPO] PPO algorithm built successfully")

    for i in range(MAX_ITERS):
        print(f">>> [IPPO] Training iter {i} ...")
        result = algo.train()
        print(
            f"Iter {i} | "
            f"reward_mean={result['episode_reward_mean']:.3f} | "
            f"len={result['episode_len_mean']:.1f}"
        )

    print(">>> [IPPO] Training finished")


if __name__ == "__main__":
    main()
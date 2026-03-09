# Libs/Environments/RLLibWrapper.py
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from Libs.Environments.DataCollection import DataCollection


class SARMultiAgentEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv wrapper around your existing DataCollection env.

    It reuses:
      - reset / step / get_obs / get_obs_size / get_total_actions
      - self.agents[a_id].collected_device
    and replaces the reward with:
      new_victims - alpha * (#non-hover actions)
    """

    def __init__(self, config):
        super().__init__()
        self.args = config["args"]
        self.params = config["params"]
        self.alpha = config.get("alpha", 0.1)

        # underlying environment
        self.base_env = DataCollection(
            args=self.args,
            params=self.params,
            learning_channel_model=None,
        )

        self.n_agents = self.base_env.n_agents
        self.n_actions = self.base_env.get_total_actions()
        self.episode_limit = self.base_env.episode_limit

        obs_dim = self.base_env.get_obs_size()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = Discrete(self.n_actions)

        self.detected_victims = set()
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.base_env.reset(model=False)
        self.detected_victims.clear()
        self.step_count = 0

        obs_list = self.base_env.get_obs()
        obs_dict = {f"agent_{i}": obs_list[i].astype(np.float32)
                    for i in range(self.n_agents)}
        info_dict = {f"agent_{i}": {} for i in range(self.n_agents)}
        return obs_dict, info_dict

    def step(self, action_dict):
        # 1) Convert dict to joint action list
        actions = [action_dict[f"agent_{i}"] for i in range(self.n_agents)]

        # 2) Call original env step (ignore its reward)
        _, terminated, _ = self.base_env.step(actions, model=False)

        # 3) SAR reward: count new victims
        new_victims = 0
        for a_id, agent in enumerate(self.base_env.agents):
            dev_array = agent.collected_device
            try:
                dev = int(dev_array.item())
            except Exception:
                dev = int(np.array(dev_array).flatten()[0])

            if dev != -1 and dev not in self.detected_victims:
                new_victims += 1
                self.detected_victims.add(dev)

        energy_penalty = sum(1 for a in actions if a != 0)
        reward_scalar = new_victims - self.alpha * energy_penalty

        rewards = {f"agent_{i}": reward_scalar for i in range(self.n_agents)}

        # 4) Next observations
        obs_list = self.base_env.get_obs()
        obs_dict = {f"agent_{i}": obs_list[i].astype(np.float32)
                    for i in range(self.n_agents)}

        # 5) Termination / truncation flags
        self.step_count += 1
        reached_limit = self.step_count >= self.episode_limit
        done_all = terminated or reached_limit

        terminations = {f"agent_{i}": done_all for i in range(self.n_agents)}
        terminations["__all__"] = done_all

        # if you want to be more “correct”, treat time-limit as truncation:
        truncations = {f"agent_{i}": reached_limit and not terminated
                       for i in range(self.n_agents)}
        truncations["__all__"] = reached_limit and not terminated

        infos = {f"agent_{i}": {} for i in range(self.n_agents)}

        return obs_dict, rewards, terminations, truncations, infos
# Libs/Environments/IPPOEnv.py

import numpy as np
from gymnasium.spaces import Box, Discrete

from Libs.Environments.DataCollection import DataCollection


class SARMultiAgentEnvBasic:
    """
    Simple multi-agent env wrapper around your existing DataCollection env.

    - No Ray / RLlib dependency.
    - API:

        obs = env.reset()
        obs, rewards, dones, info = env.step(actions)

      where:
        obs:     np.ndarray, shape (n_agents, obs_dim)
        rewards: np.ndarray, shape (n_agents,)
        dones:   np.ndarray[bool], shape (n_agents,)  (all True when episode ends)
    """

    def __init__(self, args, params, alpha: float = 0.1):
        self.args = args
        self.params = params
        self.alpha = alpha  # kept for consistency; NOT used in reward now

        # params can be either:
        #   - a dict (rbm_set_env / rdm_set_env output)
        #   - an already-built DataCollection instance
        if isinstance(params, DataCollection):
            self.base_env = params
        else:
            self.base_env = DataCollection(
                args=self.args,
                params=self.params,
                learning_channel_model=None,
            )

        self.n_agents = self.base_env.n_agents

        # DataCollection uses n_actions=6 internally; keep wrapper aligned with base env
        # If your DataCollection implements get_total_actions(), use it. Else fall back to 6.
        if hasattr(self.base_env, "get_total_actions"):
            self.n_actions = int(self.base_env.get_total_actions())
        else:
            self.n_actions = int(getattr(self.base_env, "n_actions", 6))

        self.episode_limit = int(self.base_env.episode_limit)

        obs_dim = int(self.base_env.get_obs_size())
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = Discrete(self.n_actions)

        # Evaluation bookkeeping (NOT used as training reward)
        self.detected_victims = set()
        self.step_count = 0

    # ---------- Standard API ----------

    def reset(self):
        """Reset episode and return observations for all agents (N x obs_dim)."""
        self.base_env.reset(model=False)
        self.detected_victims.clear()
        self.step_count = 0

        obs_list = self.base_env.get_obs()  # list length = n_agents
        obs = np.stack([o.astype(np.float32) for o in obs_list], axis=0)
        return obs

    def step(self, actions):
        """
        One step in the environment.

        Parameters
        ----------
        actions: list or 1D np.ndarray of length n_agents

        Returns
        -------
        obs:     (n_agents, obs_dim)
        rewards: (n_agents,)  (all agents share same scalar reward)
        dones:   (n_agents,) bool
        info:    dict
        """
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        # 1) Step the ORIGINAL env and USE its reward for fairness with other baselines
        # DataCollection.step returns: reward, terminated, info
        base_reward, terminated, info = self.base_env.step(actions, model=False)
        reward_scalar = float(base_reward)

        # 2) Bookkeeping only: count newly detected victims for evaluation metric
        # (does not affect training reward)
        for agent in self.base_env.agents:
            dev_array = agent.collected_device
            try:
                dev = int(dev_array.item())
            except Exception:
                dev = int(np.array(dev_array).flatten()[0])

            if dev != -1 and dev not in self.detected_victims:
                self.detected_victims.add(dev)

        rewards = np.full(self.n_agents, reward_scalar, dtype=np.float32)

        # 3) Next observations
        obs_list = self.base_env.get_obs()
        obs = np.stack([o.astype(np.float32) for o in obs_list], axis=0)

        # 4) Termination
        self.step_count += 1
        done = bool(terminated) or (self.step_count >= self.episode_limit)
        dones = np.full(self.n_agents, done, dtype=bool)

        # Keep info as dict to satisfy calling code (and avoid None)
        if info is None or not isinstance(info, dict):
            info = {}

        return obs, rewards, dones, info

    # ---------- Evaluation helper ----------

    def get_episode_score(self) -> float:
        """
        Evaluation metric used in the paper:
        total number of distinct victims localised in the current episode.
        """
        return float(len(self.detected_victims))
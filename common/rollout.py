import numpy as np
import torch
from torch.distributions import one_hot_categorical


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.device = torch.device(args.device)

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        # last episode info dict (for Runner / stats)
        self.last_info = None

        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False, model=False, random_action=None):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        uav_trjs = [[] for _ in range(self.env.n_agents)]
        d_idx = [[] for _ in range(self.env.n_agents)]
        action_record = [[] for _ in range(self.env.n_agents)]
        self.env.reset(model=model)
        terminated = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.n_agents, self.n_actions))
        self.agents.policy.init_hidden(1)

        # Track which victim indices we've already credited
        detected_victims = set()
        time_to_first_detection = None

        # SAR reward shaping: visited grid cells (5 m resolution)
        visited_cells = set()

        # Epsilon scheduling
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode' and not evaluate:
            epsilon = max(self.min_epsilon, epsilon - self.anneal_epsilon)

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # 1) Choose actions
            for agent_id in range(self.n_agents):
                avail = self.env.get_avail_agent_actions(agent_id)
                if random_action:
                    a = np.random.choice(np.nonzero(avail)[0])
                else:
                    a = self.agents.choose_action(
                        obs[agent_id], last_action[agent_id], agent_id, avail, epsilon
                    )
                onehot = np.zeros(self.n_actions)
                onehot[a] = 1
                actions.append(int(a))
                actions_onehot.append(onehot)
                avail_actions.append(avail)
                last_action[agent_id] = onehot
                action_record[agent_id].append(int(a))

            # 2) Step environment
            _, terminated, _ = self.env.step(actions, model=model)

            # 3) Custom reward: # newly detected victims minus energy cost
            new_victims = 0
            for a_id, agent in enumerate(self.env.agents):
                # agent.collected_device is a 1-element np.ndarray
                dev_array = agent.collected_device
                # extract an int robustly
                try:
                    dev = int(dev_array.item())
                except Exception:
                    dev = int(np.array(dev_array).flatten()[0])

                # credit only the first time we see this victim
                if dev != -1 and dev not in detected_victims:
                    new_victims += 1
                    detected_victims.add(dev)
                    if time_to_first_detection is None:
                        # step index when first victim was detected
                        time_to_first_detection = step

                # record positions & indices
                uav_trjs[a_id].append(agent.current_pose.flatten())
                d_idx[a_id].append(dev)

            # energy penalty: 1 per non-hover (action != 0) move
            energy_penalty = sum(1 for a in actions if a != 0)

            alpha = 0.1

            if getattr(self.args, 'sar_reward', False):
                # --- SAR-aligned reward shaping ---

                # 1) Time urgency: earlier victim detection earns more reward
                time_factor = 1.0 - step / self.episode_limit
                victim_reward = new_victims * (1.0 + self.args.sar_time_weight * time_factor)

                # 2) Coverage bonus: reward visiting new 5 m grid cells
                current_cells = set()
                for agent in self.env.agents:
                    pos = agent.current_pose.flatten()[:2]
                    cell = (int(round(pos[0] / 5)), int(round(pos[1] / 5)))
                    current_cells.add(cell)
                new_cells = current_cells - visited_cells
                visited_cells.update(current_cells)
                coverage_bonus = self.args.sar_coverage_weight * len(new_cells)

                # 3) Coordination penalty: penalise UAVs occupying the same cell
                positions = []
                for agent in self.env.agents:
                    pos = agent.current_pose.flatten()[:2]
                    positions.append((int(round(pos[0] / 5)), int(round(pos[1] / 5))))
                overlap = sum(
                    1 for i in range(len(positions))
                    for j in range(i + 1, len(positions))
                    if positions[i] == positions[j]
                )
                coordination_penalty = self.args.sar_coordination_weight * overlap

                reward = victim_reward - alpha * energy_penalty + coverage_bonus - coordination_penalty
            else:
                reward = new_victims - alpha * energy_penalty

            # 4) Log step data
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            # epsilon anneal per step
            if self.args.epsilon_anneal_scale == 'step' and not evaluate:
                epsilon = max(self.min_epsilon, epsilon - self.anneal_epsilon)

        # Append final positions & mark termination
        for a_id, agent in enumerate(self.env.agents):
            uav_trjs[a_id].append(agent.current_pose.flatten())
            d_idx[a_id].append(-1)

        # Last observation
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)

        # Build next-state arrays
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # Avail actions for last obs
        last_avail = [self.env.get_avail_agent_actions(i) for i in range(self.n_agents)]
        avail_u.append(last_avail)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # Pad if early termination
        for _ in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        # Package into dict
        episode = dict(
            o=o.copy(), s=s.copy(), u=u.copy(), r=r.copy(),
            avail_u=avail_u.copy(), o_next=o_next.copy(), s_next=s_next.copy(),
            avail_u_next=avail_u_next.copy(), u_onehot=u_onehot.copy(),
            padded=padded.copy(), terminated=terminate.copy()
        )
        for k in episode:
            episode[k] = np.array([episode[k]])

        # --- Episode-level metrics for statistics ---
        total_detected = len(detected_victims)

        if time_to_first_detection is None:
            # no victim detected: set to horizon as per spec
            time_to_first_detection = self.env.episode_limit

        energy_used = float(getattr(self.env, "energy_used", 0.0))
        energy_per_victim = energy_used / max(1, total_detected)

        info = {
            "victims_found": total_detected,
            "time_to_first_detection": time_to_first_detection,
            "energy_used": energy_used,
            "energy_per_victim": energy_per_victim,
            "episode_steps": step,
        }

        # expose metrics for Runner / external scripts
        self.env.victims_found = total_detected
        self.env.time_to_first_detection = time_to_first_detection
        self.env.energy_per_victim = energy_per_victim
        self.last_info = info

        # Return same outputs as before (compatibility)
        return episode, episode_reward, step, total_detected, uav_trjs, d_idx, action_record
import numpy as np
import os
import torch
from runner import Runner
from Libs.Utils.Plots import *
from common.arguments import get_common_args, get_mixer_args
from Libs.Environments.DataCollection import DataCollection


def plot_device(ue_pose, fig_id, color=None, marker='o', marker_size=50):
    plt.figure(fig_id)
    # Updated label:
    plt.scatter(ue_pose[:, 0], ue_pose[:, 1],
                marker=marker, c=color, s=marker_size,
                label='Estimated Victim Position', zorder=12)


def plot_uav(uav_trj, fig_id, color=None, marker='', marker_size=5):
    plt.figure(fig_id)
    plt.scatter(uav_trj[:, 0], uav_trj[:, 1],
                c=color, marker=marker, s=marker_size)


ColorMap = ["brown", "orange", "green", "olive", "purple",
            "blue", "pink", "gray", "red", "cyan", "black"]


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)

    if args.map == 'RBM':
        from config.RBM_define import params
    elif args.map == 'RDM':
        from config.RDM_define import params
    else:
        raise Exception("No such map!")

    env = DataCollection(args, params=params)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    device = torch.device(args.device)

    runner = Runner(env, args)

    result_dir = args.result_dir + '/' + args.alg + '/' + args.map + args.tag
    model_dir = args.model_dir + '/' + args.alg + '/' + args.map + args.tag

    # Load the trained model parameters
    if args.alg == 'qmix':
        rnn_path = model_dir + '/final_rnn_net_params.pkl'
        qmix_path = model_dir + '/final_qmix_net_params.pkl'
        if os.path.exists(rnn_path) and os.path.exists(qmix_path):
            runner.agents.policy.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=device))
            runner.agents.policy.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=device))
            print(f'Successfully loaded model: {rnn_path} & {qmix_path}')
        else:
            raise Exception("Model files not found!")
        # Sync target networks
        runner.agents.policy.target_rnn.load_state_dict(runner.agents.policy.eval_rnn.state_dict())
        runner.agents.policy.target_qmix_net.load_state_dict(runner.agents.policy.eval_qmix_net.state_dict())
        runner.agents.policy.eval_parameters = (
            list(runner.agents.policy.eval_qmix_net.parameters()) +
            list(runner.agents.policy.eval_rnn.parameters())
        )

    elif args.alg == 'iql':
        rnn_path = model_dir + '/final_rnn_net_params.pkl'
        if os.path.exists(rnn_path):
            runner.agents.policy.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=device))
            print(f'Successfully loaded model: {rnn_path}')
        else:
            raise Exception("Model file not found!")
        # Sync target network
        runner.agents.policy.target_rnn.load_state_dict(runner.agents.policy.eval_rnn.state_dict())
        runner.agents.policy.eval_parameters = list(runner.agents.policy.eval_rnn.parameters())

    # Evaluate: now returns (avg_victims_detected, avg_reward, trajectories, device_idx, actions)
    total_victims_detected, episode_rewards, uav_trjs, device_idx, action_record = runner.evaluate()

    # Print total victims detected
    print(f'Total victims detected: {total_victims_detected}')

    # Prepare color lists for UAV trajectories
    device_color_list = [[] for _ in range(args.n_agents)]
    for i in range(args.n_agents):
        uav_trjs[i] = np.array(uav_trjs[i])
        device_idx[i] = np.array(device_idx[i])
        for idx in device_idx[i]:
            # If idx == -1 (termination), default color
            color = ColorMap[int(idx)] if idx >= 0 else 'black'
            device_color_list[i].append(color)
        device_color_list[i] = np.array(device_color_list[i])

    # Plot city background
    ax = plot_city_top_view(params['city'], 10, figsize=(8, 6), fontsize=12)

    # True victim positions and anchor beacon positions
    device_positions = np.resize(params['device_position'], (len(params['device_position']), 3))
    unknown_positions = device_positions[params['unknown_device_idx']]
    anchor_positions = device_positions[params['known_device_idx']]
    unknown_colors = np.array(params['color'])[params['unknown_device_idx']]
    anchor_colors = np.array(params['color'])[params['known_device_idx']]

    # Updated legend labels:
    plt.scatter(unknown_positions[:, 0], unknown_positions[:, 1],
                marker='*', s=100, c=unknown_colors,
                label='True Victim Position')
    plt.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
                marker='^', s=100, c=anchor_colors,
                label='Anchor Beacon Position')

    if args.model:
        est_path = result_dir + '/est_device_pos.npy'
        est_positions = np.load(est_path)
        est_positions = np.array([p.flatten() for p in est_positions[-1]])
        plot_device(est_positions[params['unknown_device_idx']], 10, marker='+', marker_size=50, color='r')

    # UAV start/terminal zones
    start_poses = np.resize(params['start_pose'], (len(params['start_pose']), 3))
    end_poses = np.resize(params['end_pose'], (len(params['end_pose']), 3))
    plt.scatter(start_poses[:, 0], start_poses[:, 1], marker='H', s=150,
                c='lightgray', label='UAV Start Zone')
    plt.scatter(end_poses[:, 0], end_poses[:, 1], marker='H', s=150,
                c='lightblue', label='UAV End Zone')

    # Plot UAV trajectories
    markers = ['o', 's', 'D']
    for i in range(args.n_agents):
        plt.scatter(uav_trjs[i][:, 0], uav_trjs[i][:, 1],
                    marker=markers[i], s=7, c=device_color_list[i], zorder=10)
        # Add a legend entry for each UAV
        plt.scatter(uav_trjs[i][0, 0], uav_trjs[i][0, 1],
                    marker=markers[i], s=20, c='b',
                    label=f"UAV {i+1} Trajectory", zorder=-1)

    # Adjust legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              fancybox=True, shadow=True, ncol=3, fontsize=11,
              columnspacing=0.2, handlelength=1, handletextpad=0.2)

    # Save figure
    plt.savefig(result_dir + '/trj_victims.pdf', format='pdf',
                bbox_inches='tight', pad_inches=0.1)
    plt.savefig(result_dir + '/trj_victims.png', format='png',
                bbox_inches='tight', pad_inches=0.1, dpi=1200)
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


if __name__ == '__main__':
    plt.figure(figsize=(8, 6.5))
    window_size = 5
    model_learning_period = 1000
    evaluation_cycle = 50
    scale = model_learning_period / evaluation_cycle

    # Directories for the various algorithms
    # (update these to your actual result folders)
    dir_iql = 'dir_1_1'
    dir_qmix = 'dir_2_1'
    dir_model_qmix = 'dir_3_1'
    dir_model_fedqmix = 'dir_4_1'

    # Load episode data (number of victims detected per evaluation)
    # and normalize if needed (here assumed raw counts)
    data_iql = np.load(dir_iql + '/episode_data.npy')
    data_qmix = np.load(dir_qmix + '/episode_data.npy')
    data_model_qmix = np.load(dir_model_qmix + '/episode_data.npy')[:600]
    data_model_fedqmix = np.load(dir_model_fedqmix + '/global_data.npy')[:600]

    # Compute moving averages and x‐axes
    def process(data, scale_factor=1.0):
        mv = moving_average(data, window_size)
        xs = list(range(len(mv)))
        return xs, mv

    xs1, mv1 = process(data_iql)
    xs2, mv2 = process(data_qmix)
    xs3, mv3 = process(data_model_qmix, scale)
    xs4, mv4 = process(data_model_fedqmix, scale)

    # Plot each curve
    plt.plot(xs4, mv4, linewidth=3, label='FedQMIX (model‐aided)')
    plt.plot(xs3, mv3, linewidth=3, linestyle='--', label='QMIX (model‐aided)')
    plt.plot(xs2, mv2, linewidth=2, linestyle='-.', label='QMIX (model‐free)')
    plt.plot(xs1, mv1, linewidth=2, linestyle=':', label='IQL (model‐free)')

    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlim([0.11, max(xs4) + 1])
    plt.ylim(bottom=0)

    # Updated labels for victim localization
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Episode (log scale)', fontsize=16)
    plt.ylabel('Average Victims Detected', fontsize=16)
    plt.title('Victims Detected vs. Training Episodes', fontsize=18)
    plt.legend(loc='lower right', fontsize=12)

    # Save figures
    result_dir = './Figures'
    plt.savefig(result_dir + '/victim_detection_results.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(result_dir + '/victim_detection_results.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
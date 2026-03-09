import random
import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args
from training_procedures import run_training


def main():
    args = get_common_args()

    # QMIX / IQL use the mixer hyperparams
    if args.alg == 'qmix' or args.alg == 'iql':
        args = get_mixer_args(args)

    # Map-specific configuration (RBM / RDM)
    if args.map == 'RBM':
        from config.RBM_define import params
    elif args.map == 'RDM':
        from config.RDM_define import params
    else:
        raise Exception("No such map!")

    # Reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Unified training entry
    run_training(args, params)


if __name__ == '__main__':
    main()

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

    # Multi-seed support: --seeds 1 2 3 4 5 runs one full experiment per seed
    seeds = args.seeds if args.seeds is not None else [args.seed]
    base_tag = args.tag

    for seed in seeds:
        args.seed = seed
        # Each seed gets its own result subdirectory (e.g. _s1, _s2, ...)
        args.tag = f"{base_tag}_s{seed}" if len(seeds) > 1 else base_tag

        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"[main] Starting run  seed={seed}  tag='{args.tag}'")
        run_training(args, params)


if __name__ == '__main__':
    main()

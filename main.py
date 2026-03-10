import copy
import random
import numpy as np
import torch
import multiprocessing as mp

from common.arguments import get_common_args, get_mixer_args
from training_procedures import run_training


def _run_seed(args, params):
    """Worker function: set seed and run training in a separate process."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[main] Starting run  seed={args.seed}  tag='{args.tag}'")
    run_training(args, params)


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

    seeds = args.seeds if args.seeds is not None else [args.seed]
    base_tag = args.tag

    if len(seeds) == 1 or not getattr(args, 'parallel', False):
        # Sequential (default)
        for seed in seeds:
            a = copy.deepcopy(args)
            a.seed = seed
            a.tag = f"{base_tag}_s{seed}" if len(seeds) > 1 else base_tag
            _run_seed(a, params)
    else:
        # Parallel: one process per seed
        processes = []
        for seed in seeds:
            a = copy.deepcopy(args)
            a.seed = seed
            a.tag = f"{base_tag}_s{seed}"
            p = mp.Process(target=_run_seed, args=(a, params))
            p.start()
            processes.append(p)
            print(f"[main] Launched process for seed={seed}")
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()

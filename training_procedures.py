# ───────────────────────── training_procedures.py ──────────────────────────
import os
import json
import random

import numpy as np
import torch

from runner import Runner
from Libs.Environments.DataCollection import DataCollection
from Libs.ChannelEstimator import SLAL


# ═════════════════════════ Dispatcher ═════════════════════════
def run_training(args, params):
    """
    Single entry-point called from main.py.

    It now supports two ablation toggles (with backward compatibility):

        --use_surrogate_channel {0,1}
        --use_federation        {0,1}

    Mapping:

        use_federation=0, use_surrogate_channel=0  → plain QMIX
        use_federation=0, use_surrogate_channel=1  → model-aided QMIX
        use_federation=1, use_surrogate_channel=0  → FedQMIX (no surrogate)
        use_federation=1, use_surrogate_channel=1  → Model-aided FedQMIX

    IQL ignores these toggles (no model, no federation).
    """
    # Normalise algorithm name
    args.alg = args.alg.lower()
    alg = args.alg

    # ── 1) Read ablation toggles with backward compatibility ──────────────
    # Surrogate channel toggle
    use_sur = getattr(args, "use_surrogate_channel", None)
    if use_sur is None:
        # Fall back to old flag "model"
        use_sur = bool(getattr(args, "model", False))
    else:
        use_sur = bool(use_sur)
    args.model = use_sur  # keep old field in sync

    # Federation toggle
    use_fed = getattr(args, "use_federation", None)
    if use_fed is None:
        # Fall back to old flag "federated"
        use_fed = bool(getattr(args, "federated", False))
    else:
        use_fed = bool(use_fed)

    # If user explicitly chose "fedqmix", force federation on
    if alg == "fedqmix":
        use_fed = True
        args.federated = True
        alg = "qmix"
        args.alg = "qmix"
    else:
        args.federated = use_fed

    # ── 2) Route to correct training routine ───────────────────────────────
    # IQL: always non-federated, no surrogate
    if alg == "iql":
        args.model = False
        args.federated = False
        _train_iql(args, params)
        return

    # All QMIX variants (centralised + federated) go through here
    if alg == "qmix":
        if use_fed:
            # Federated variants (FedQMIX), with or without surrogate
            args.model = use_sur  # True → attach SLAL; False → no surrogate
            _train_fedqmix(args, params)
        else:
            # Centralised variants
            if use_sur:
                _train_qmix_model(args, params)   # model-aided QMIX
            else:
                _train_qmix_plain(args, params)   # plain QMIX
        return

    # Anything else is unsupported
    raise ValueError(
        f"Unsupported combo (alg={args.alg}, use_surrogate_channel={use_sur}, use_federation={use_fed})"
    )


# ═════════════════════ Non-federated learners ═══════════════════

def _train_qmix_plain(args, params):
    """Centralised QMIX, NO surrogate channel."""
    args.model = False
    args.federated = False
    _run_single(args, params)


def _train_qmix_model(args, params):
    """Centralised QMIX, WITH surrogate channel (Model-aided QMIX)."""
    args.model = True
    args.federated = False
    _run_single(args, params)


def _train_iql(args, params):
    """IQL baseline: no surrogate, no federation."""
    args.model = False
    args.federated = False
    _run_single(args, params)


# ═════════════════════ FedQMIX (article’s FL training) ═════════════════════

def _train_fedqmix(args, params):
    """
    Federated QMIX training loop.

    Behaviour:
        args.model == True  → model-aided FedQMIX (uses SLAL surrogate)
        args.model == False → FedQMIX without surrogate
    """
    n_workers = args.workers
    _fed_sub = "fed_mod" if args.model else "fed"
    save_path = os.path.join(args.result_dir, "qmix", _fed_sub, args.map + args.tag)
    os.makedirs(save_path, exist_ok=True)

    # Seed BEFORE creating environments so all RNG state is deterministic
    _set_seeds(args.seed)

    # 1) Build envs (no SLAL yet)
    envs = [DataCollection(args, learning_channel_model=None, params=params)
            for _ in range(n_workers)]
    _fill_env_info_into_args(envs[0], args)

    # 2) Runners + custom epsilons
    # NOTE: assumes <=3 workers; adapt if you increase args.workers.
    eps = [0.3, 0.1, 0.05]
    runners = []
    for i in range(n_workers):
        r = Runner(envs[i], args)
        # each worker has slightly different exploration
        r.rolloutWorker.epsilon = eps[i]
        r.rolloutWorker.min_epsilon = eps[i]
        runners.append(r)

    # 3) Sync initial networks across workers
    for r in runners[1:]:
        r.agents.policy.eval_qmix_net.load_state_dict(
            runners[0].agents.policy.eval_qmix_net.state_dict())
        r.agents.policy.target_qmix_net.load_state_dict(
            runners[0].agents.policy.target_qmix_net.state_dict())
        r.agents.policy.eval_rnn.load_state_dict(
            runners[0].agents.policy.eval_rnn.state_dict())
        r.agents.policy.target_rnn.load_state_dict(
            runners[0].agents.policy.target_rnn.state_dict())

    # 4) Attach SLAL (only if surrogate channel is enabled)
    if args.model:
        hidden_layers = [(50, 'tanh'), (20, 'linear')]
        for r in runners:
            r.env.learning_channel_model = SLAL(
                buffer_size=10_000,
                channel_param=params['ch_param'],
                hidden_layers=hidden_layers,
                city=params['city'],
                device=args.device,
            )

    # ─── bookkeeping ────────────────────────────────────────────
    total_eps = args.total_episodes
    agg_period = args.aggregation_period
    model_period = args.model_learning_period

    data_eval_lists = [[] for _ in range(n_workers)]
    reward_eval_lists = [[] for _ in range(n_workers)]
    data_train_lists = [[] for _ in range(n_workers)]
    reward_train_lists = [[] for _ in range(n_workers)]

    global_data, global_rewards, est_pos_log = [], [], []

    # Initial global evaluation (worker 0)
    d0, r0, *_ = runners[0].evaluate(model=args.model)
    global_data.append(d0)
    global_rewards.append(r0)

    train_steps = [0] * n_workers
    eval_steps = [-1] * n_workers
    episode_cnt = [0] * n_workers
    best_ep_data = [0] * n_workers

    # ─── main federation loop ───────────────────────────────────
    num_rounds = total_eps // agg_period
    for round_k in range(num_rounds):

        # --- channel-model update & broadcast (only if surrogate is ON)
        if args.model and (episode_cnt[0] % model_period == 0):
            runners[0].env.set_default_position()
            pos = runners[0].model_learning(round_k)
            est_pos_log.append(pos)

            # broadcast channel model weights
            vec = torch.nn.utils.parameters_to_vector(
                runners[0].env.learning_channel_model.model.parameters()
            )
            for r in runners[1:]:
                torch.nn.utils.vector_to_parameters(
                    vec, r.env.learning_channel_model.model.parameters()
                )
                r.env.set_device_position(pos)

        # --- local learning on each worker
        pq_vecs = []
        prnn_vecs = []

        for wi, r in enumerate(runners):
            ts, es, ep, best = r.federated_run(
                wi,
                train_steps[wi],
                eval_steps[wi],
                episode_cnt[wi],
                data_train_lists[wi],
                reward_train_lists[wi],
                data_eval_lists[wi],
                reward_eval_lists[wi],
                best_ep_data[wi],
                model=args.model,
            )
            train_steps[wi], eval_steps[wi], episode_cnt[wi], best_ep_data[wi] = ts, es, ep, best

            pq_vecs.append(
                torch.nn.utils.parameters_to_vector(
                    r.agents.policy.eval_qmix_net.parameters()
                )
            )
            prnn_vecs.append(
                torch.nn.utils.parameters_to_vector(
                    r.agents.policy.eval_rnn.parameters()
                )
            )

        # --- FedAvg aggregation (simple unweighted mean)
        q_agg = sum(pq_vecs) / n_workers
        rnn_agg = sum(prnn_vecs) / n_workers

        for r in runners:
            torch.nn.utils.vector_to_parameters(
                q_agg, r.agents.policy.eval_qmix_net.parameters()
            )
            torch.nn.utils.vector_to_parameters(
                rnn_agg, r.agents.policy.eval_rnn.parameters()
            )

        # --- global evaluation from worker 0
        gd, gr, *_ = runners[0].evaluate(model=args.model)
        global_data.append(gd)
        global_rewards.append(gr)
        print(f"[FedQMIX] round {round_k + 1:3d}/{num_rounds:3d}  global-data={gd:.2f}")

    # ─── save exactly like the article code ─────────────────────
    np.save(os.path.join(save_path, 'global_data.npy'), np.asarray(global_data))
    np.save(os.path.join(save_path, 'global_rewards.npy'), np.asarray(global_rewards))
    np.save(os.path.join(save_path, 'est_device_pos.npy'), np.asarray(est_pos_log))

    with open(os.path.join(save_path, 'log_params.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"📦  FedQMIX finished → {save_path}")


# ═════════════════════ shared helpers (unchanged) ═════════════════════

def _run_single(args, params):
    """
    Executes one *non-federated* training run:

        • plain-QMIX   (model=False)
        • model-QMIX   (model=True)
        • IQL          (alg=iql, model=False)

    Produces global_data.npy / global_rewards.npy exactly like before.
    """
    # Seed BEFORE creating the environment so all RNG state is deterministic
    _set_seeds(args.seed)

    # Optional SLAL channel model
    chan_model = None
    if args.model:
        chan_model = SLAL(
            buffer_size=10_000,
            channel_param=params["ch_param"],
            hidden_layers=[(50, 'tanh'), (20, 'linear')],
            city=params["city"],
            device=args.device,
        )

    # Build env + runner
    env = DataCollection(args, learning_channel_model=chan_model, params=params)
    _fill_env_info_into_args(env, args)

    runner = Runner(env, args)
    runner.run(model=args.model)

    _save_single_results(runner, args)


def _save_single_results(runner, args):
    """
    Saves arrays and log_params.txt under:

        result/<alg>/<map><tag>/
    """
    model = bool(getattr(args, 'model', False))
    federated = bool(getattr(args, 'federated', False))
    alg = args.alg.lower()
    if alg == 'iql':
        _sub = os.path.join('iql')
    elif federated and model:
        _sub = os.path.join('qmix', 'fed_mod')
    elif federated:
        _sub = os.path.join('qmix', 'fed')
    elif model:
        _sub = os.path.join('qmix', 'mod')
    else:
        _sub = os.path.join('qmix')
    out_dir = os.path.join(args.result_dir, _sub, args.map + args.tag)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'global_data.npy'),
            np.asarray(runner.episode_data))
    np.save(os.path.join(out_dir, 'global_rewards.npy'),
            np.asarray(runner.episode_rewards))

    with open(os.path.join(out_dir, 'log_params.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)


def _fill_env_info_into_args(env, args):
    """
    Copies env-specific dimensions into `args`.
    """
    info = env.get_env_info()
    args.n_actions = info["n_actions"]
    args.n_agents = info["n_agents"]
    args.state_shape = info["state_shape"]
    args.obs_shape = info["obs_shape"]
    args.episode_limit = info["episode_limit"]


def _set_seeds(seed: int):
    """Reproducibility helper — seeds Python, NumPy, PyTorch CPU and GPU RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from torch import optim
import eve_rl
import eve
import numpy as np


def create_agent(
    device_trainer,
    device_worker,
    lr,
    lr_end_factor,
    lr_linear_end_steps,
    hidden_layers,
    embedder_nodes,
    embedder_layers,
    gamma,
    batch_size,
    reward_scaling,
    replay_buffer_size,
    train_env: eve.Env,
    eval_env: eve.Env,
    consecutive_action_steps,
    n_worker,
    stochastic_eval: bool = False,
    single: bool = False,
):
    obs_dict = train_env.observation_space.sample()
    obs_list = [obs.flatten() for obs in obs_dict.values()]
    obs_np = np.concatenate(obs_list)

    n_observations = obs_np.shape[0]
    n_actions = train_env.action_space.sample().flatten().shape[0]

    q1_embedder = eve_rl.network.component.LSTM(
        n_layer=embedder_layers, n_nodes=embedder_nodes
    )

    q1_base = eve_rl.network.component.MLP(hidden_layers)
    q2_base = eve_rl.network.component.MLP(hidden_layers)
    policy_base = eve_rl.network.component.MLP(hidden_layers)

    q1 = eve_rl.network.QNetwork(q1_base, n_observations, n_actions, q1_embedder)
    q1_optim = eve_rl.optim.Adam(
        q1,
        lr=lr,
    )
    q1_scheduler = optim.lr_scheduler.LinearLR(
        q1_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    q2 = eve_rl.network.QNetwork(q2_base, n_observations, n_actions, q1_embedder)
    q2_optim = eve_rl.optim.Adam(
        q2_base,
        lr=lr,
    )
    q2_scheduler = optim.lr_scheduler.LinearLR(
        q2_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    policy = eve_rl.network.GaussianPolicy(
        policy_base, n_observations, n_actions, q1_embedder
    )
    policy_optim = eve_rl.optim.Adam(
        policy_base,
        lr=lr,
    )
    policy_scheduler = optim.lr_scheduler.LinearLR(
        policy_optim,
        start_factor=1.0,
        end_factor=lr_end_factor,
        total_iters=lr_linear_end_steps,
    )

    sac_model = eve_rl.model.SACModel(
        lr_alpha=lr,
        q1=q1,
        q2=q2,
        policy=policy,
        q1_optimizer=q1_optim,
        q2_optimizer=q2_optim,
        policy_optimizer=policy_optim,
        q1_scheduler=q1_scheduler,
        q2_scheduler=q2_scheduler,
        policy_scheduler=policy_scheduler,
    )

    algo = eve_rl.algo.SAC(
        sac_model,
        n_actions=n_actions,
        gamma=gamma,
        reward_scaling=reward_scaling,
        stochastic_eval=stochastic_eval,
    )

    replay_buffer = eve_rl.replaybuffer.VanillaEpisodeShared(
        replay_buffer_size, batch_size, device_trainer
    )
    if not single:
        agent = eve_rl.agent.Synchron(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            trainer_device=device_trainer,
            worker_device=device_worker,
            n_worker=n_worker,
            normalize_actions=True,
        )
    else:
        agent = eve_rl.agent.Single(
            algo,
            train_env,
            eval_env,
            replay_buffer,
            consecutive_action_steps=consecutive_action_steps,
            device=device_trainer,
            normalize_actions=True,
        )

    return agent

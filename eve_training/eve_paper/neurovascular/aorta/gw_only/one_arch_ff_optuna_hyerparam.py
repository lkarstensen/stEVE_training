import argparse
from copy import deepcopy
import math
import os
import logging
import eve
import torch.multiprocessing as mp
import torch
import optuna
from eve_training.util import get_result_checkpoint_config_and_log_path
from eve_training.optunapruner import CombinationPruner, StagnatingPruner
from eve_rl import Runner
from eve_bench.neurovascular.aorta.guidewire_only.archgentypeI import ArchGenTypeI
from eve_training.eve_paper.neurovascular.aorta.gw_only.agent1 import create_agent
from eve_training.eve_paper.neurovascular.aorta.gw_only.env1 import GwOnly


RESULTS_FOLDER = (
    os.getcwd()
    + "/results/eve_paper/neurovascular/aorta/gw_only/arch_1669028594_ff_hyperparam_opti"
)

EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
EVAL_SEEDS = EVAL_SEEDS.split(",")
EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]

HEATUP_STEPS = 5e5
TRAINING_STEPS = 1e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 5e5

# HEATUP_STEPS = 5e3
# TRAINING_STEPS = 1e7
# CONSECUTIVE_EXPLORE_EPISODES = 10
# EXPLORE_STEPS_BTW_EVAL = 2.5e3
# EVAL_SEEDS = list(range(20))
# RESULTS_FOLDER = os.getcwd() + "/results/test"


GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


def objective(trial: optuna.trial.Trial):
    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=trial_name
    )

    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    lr = trial.suggest_float("lr", 8e-5, 2e-3, log=True)
    n_hidden_layer = trial.suggest_int("n_hidden_layer", 3, 5)
    hidden_layer_nodes = trial.suggest_int("hidden_layer_nodes", 300, 900, step=100)
    hidden_layers = [hidden_layer_nodes] * n_hidden_layer
    embedder_nodes = 0
    embedder_layers = 0

    custom_parameters = {
        "learning_rate": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=1669028594,
        rotation_yzx_deg=[0, 0, 0],
        scaling_xyzd=[
            0.9995400602406534,
            0.9995400602406534,
            0.8178605960609021,
            0.85,
        ],
    )

    device = eve.intervention.device.JShaped(
        name="guidewire",
        velocity_limit=(35, 3.14),
        length=450,
        tip_radius=12.1,
        tip_angle=0.4 * math.pi,
        tip_outer_diameter=0.7,
        tip_inner_diameter=0.0,
        straight_outer_diameter=0.89,
        straight_inner_diameter=0.0,
        poisson_ratio=0.49,
        young_modulus_tip=17e3,
        young_modulus_straight=80e3,
        mass_density_tip=0.000021,
        mass_density_straight=0.000021,
        visu_edges_per_mm=0.5,
        collis_edges_per_mm_tip=2,
        collis_edges_per_mm_straight=0.1,
        beams_per_mm_tip=1.4,
        beams_per_mm_straight=0.5,
        color=(0.0, 0.0, 0.0),
    )
    simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1)

    fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
        simulation=simulation,
        vessel_tree=vessel_tree,
        image_frequency=7.5,
        image_rot_zx=[25, 0],
        image_center=[0, 0, 0],
        field_of_view=None,
    )

    target = eve.intervention.target.CenterlineRandom(
        vessel_tree=vessel_tree,
        fluoroscopy=fluoroscopy,
        threshold=5,
        branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
    )
    intervention = eve.intervention.MonoPlaneStatic(
        vessel_tree,
        [device],
        simulation,
        fluoroscopy,
        target,
        stop_device_at_tree_end=True,
        normalize_action=False,
    )

    intervention2 = deepcopy(intervention)
    env_train = GwOnly(intervention=intervention, mode="train", visualisation=False)

    env_eval = GwOnly(intervention=intervention2, mode="eval", visualisation=False)
    agent = create_agent(
        trainer_device,
        worker_device,
        lr,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        hidden_layers,
        embedder_nodes,
        embedder_layers,
        GAMMA,
        BATCH_SIZE,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        n_worker,
        stochastic_eval,
        False,
        ff_only=True,
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())

    runner = Runner(
        agent=agent,
        heatup_action_low=[-10.0, -1.0],
        heatup_action_high=[25, 3.14],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
    )
    runner_config = os.path.join(config_folder, "runner.yml")
    runner.save_config(runner_config)

    runner.heatup(HEATUP_STEPS)
    next_eval_limit = EXPLORE_STEPS_BTW_EVAL
    while runner.step_counter.exploration < TRAINING_STEPS:
        runner.explore_and_update(
            CONSECUTIVE_EXPLORE_EPISODES,
            UPDATE_PER_EXPLORE_STEP,
            explore_steps=EXPLORE_STEPS_BTW_EVAL,
        )
        quality, _ = runner.eval(seeds=EVAL_SEEDS)
        trial.report(quality, runner.step_counter.exploration)
        next_eval_limit += EXPLORE_STEPS_BTW_EVAL

        if trial.should_prune():
            agent.close()
            raise optuna.TrialPruned()
        if agent.update_error:
            break

    agent.close()
    del agent
    del intervention
    del env_eval
    del env_train
    del runner
    torch.cuda.empty_cache()
    return quality


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="perform IJCARS23 Optuna Optimization")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=5, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda", "mps"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="run", help="Name of the training run"
    )

    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    worker_device = torch.device("cpu")

    pruner_median = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=TRAINING_STEPS / 5
    )
    pruner_threshold = optuna.pruners.ThresholdPruner(
        lower=0.2, n_warmup_steps=TRAINING_STEPS / 3
    )
    pruner_finished = optuna.pruners.ThresholdPruner(upper=0.99)
    stagnation_prunter = StagnatingPruner(
        fluctuation_boundary=0.01,
        n_warmup_steps=TRAINING_STEPS / 4,
        n_averaged_values=10,
        n_strikes=5,
    )
    pruner = CombinationPruner(
        pruners=[pruner_median, pruner_threshold, stagnation_prunter, pruner_finished]
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, 20)
    logging.basicConfig(
        filename=RESULTS_FOLDER + "main.log",
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("main")
    log_info = f"{study.best_params = }"
    logger.info(log_info)
    param_importance = optuna.importance.get_param_importances(study)
    log_info = f"{param_importance = }"
    logger.info(log_info)

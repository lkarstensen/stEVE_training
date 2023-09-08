import argparse
import os
import logging
import torch.multiprocessing as mp
import torch
import optuna
from eve_training.eve_paper.neurovascular.aorta.two_device.env1 import TwoDevice
from eve_training.util import get_result_checkpoint_config_and_log_path
from eve_training.optunapruner import CombinationPruner, StagnatingPruner
from eve_training.eve_paper.neurovascular.aorta.two_device.agent1 import create_agent
from eve_rl import Runner
import eve_bench.neurovascular.aorta.simple_cath.arch_generator

RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/cerebral/aorta/heatup_opti"

HEATUP_STEPS = 5e5
TRAINING_STEPS = 1e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 5e5
EVAL_SEEDS = list(range(100))
LEARNING_RATE = 0.00021989352630306626
HIDDEN_LAYERS = [900, 900, 900, 900]
N_EMBEDDER_LAYER = 1
N_EMBEDDER_NODES = 500

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

    intervention = (
        eve_bench.neurovascular.aorta.simple_cath.arch_generator.ArchGenerator(
            episodes_between_arch_change=1
        )
    )
    env_train = TwoDevice(intervention=intervention, mode="train", visualisation=False)
    intervention = (
        eve_bench.neurovascular.aorta.simple_cath.arch_generator.ArchGenerator(
            episodes_between_arch_change=1
        )
    )
    env_eval = TwoDevice(intervention=intervention, mode="eval", visualisation=False)
    agent = create_agent(
        trainer_device,
        worker_device,
        LEARNING_RATE,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        HIDDEN_LAYERS,
        N_EMBEDDER_NODES,
        N_EMBEDDER_LAYER,
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
    )

    env_train_config = os.path.join(config_folder, "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = os.path.join(config_folder, "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())

    heatup_0_high = trial.suggest_float("heatup_0_high", 10.0, 25.0)
    heatup_0_low = trial.suggest_float("heatup_0_range", -20.0, -1.0)
    heatup_1_diff = trial.suggest_float("heatup_1_diff", -8, 0.0)
    custom_parameters = {
        "heatup_0_high": heatup_0_high,
        "heatup_0_low": heatup_0_low,
        "heatup_1_diff": heatup_1_diff,
    }

    heatup_action_low = [[heatup_0_low, -1.0], [heatup_0_low - heatup_1_diff, 3.14]]
    heatup_action_high = [[heatup_0_high, -1.0], [heatup_0_high - heatup_1_diff, 3.14]]

    runner = Runner(
        agent=agent,
        heatup_action_low=heatup_action_low,
        heatup_action_high=heatup_action_high,
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
    stagnation_prunter = StagnatingPruner(
        fluctuation_boundary=0.01,
        n_warmup_steps=TRAINING_STEPS / 4,
        n_averaged_values=10,
        n_strikes=5,
    )
    pruner = CombinationPruner(
        pruners=[pruner_median, pruner_threshold, stagnation_prunter]
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.RandomSampler(),
    )
    study.optimize(objective, 10)
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

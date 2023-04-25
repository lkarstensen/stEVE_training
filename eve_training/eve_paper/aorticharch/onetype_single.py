import os
import logging
import argparse
import torch.multiprocessing as mp
import torch
from eve_training.eve_paper.aorticharch.env import AorticArchSingleType
from eve_training.util import get_result_checkpoint_config_and_log_path
from eve_training.eve_paper.aorticharch.agent1 import create_agent
from eve_rl import Runner
from eve.vesseltree import ArchType

EVAL_SEEDS_I = "1,2,4,5,6,7,9,10,11,13,14,15,16,17,18,20,21,23,24,25,26,27,29,30,31,33,34,35,37,39,40,42,43,44,45,46,47,48,49,52,53,54,56,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,78,79,80,82,84,85,86,88,89,90,92,93,94,95,96,97,98,99,100,102,104,105,106,107,108,109,110,112,113,114,115,116,117,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,138,139,140,141,143,144,145,146,147,148,149,150,151,152,153,154,155,157,158,159,160,161,162,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,188,189,190,191,193,194,195,197,198,199,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,223,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,241,242,243,244,245,246,247,248,250,252,254,255,256,257,259"
EVAL_SEEDS_I = EVAL_SEEDS_I.split(",")
EVAL_SEEDS_I = [int(seed) for seed in EVAL_SEEDS_I]

eval_seeds = {ArchType.I: EVAL_SEEDS_I}

RESULTS_FOLDER = os.getcwd() + "/results/eve_paper/aorticarch/one_type"

HEATUP_STEPS = 5e5
TRAINING_STEPS = 1e7
CONSECUTIVE_EXPLORE_EPISODES = 100
EXPLORE_STEPS_BTW_EVAL = 2.5e5

GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
BATCH_SIZE = 32
UPDATE_PER_EXPLORE_STEP = 1 / 20


LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6

DEBUG_LEVEL = logging.INFO


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="perform IJCARS23 training")
    parser.add_argument(
        "-nw", "--n_worker", type=int, default=5, help="Number of workers"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device of trainer, wehre the NN update is performed. ",
        choices=["cpu", "cuda:0", "cuda:1", "cuda"],
    )
    parser.add_argument(
        "-se",
        "--stochastic_eval",
        action="store_true",
        help="Runs optuna run with stochastic eval function of SAC.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="test", help="Name of the training run"
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.00021989352630306626,
        help="Learning Rate of Optimizers",
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[900, 900, 900, 900],
        help="Hidden Layers",
    )
    parser.add_argument(
        "-en",
        "--embedder_nodes",
        type=int,
        default=500,
        help="Number of nodes per layer in embedder",
    )
    parser.add_argument(
        "-el",
        "--embedder_layers",
        type=int,
        default=1,
        help="Number of layers in embedder",
    )
    args = parser.parse_args()

    trainer_device = torch.device(args.device)
    n_worker = args.n_worker
    trial_name = args.name
    stochastic_eval = args.stochastic_eval
    lr = args.learning_rate
    hidden_layers = args.hidden
    embedder_nodes = args.embedder_nodes
    embedder_layers = args.embedder_layers
    worker_device = torch.device("cpu")

    custom_parameters = {
        "lr": lr,
        "hidden_layers": hidden_layers,
        "embedder_nodes": embedder_nodes,
        "embedder_layers": embedder_layers,
        "HEATUP_STEPS": HEATUP_STEPS,
        "EXPLORE_STEPS_BTW_EVAL": EXPLORE_STEPS_BTW_EVAL,
        "CONSECUTIVE_EXPLORE_EPISODES": CONSECUTIVE_EXPLORE_EPISODES,
        "BATCH_SIZE": BATCH_SIZE,
        "UPDATE_PER_EXPLORE_STEP": UPDATE_PER_EXPLORE_STEP,
    }

    for archtype, seeds in eval_seeds.items():
        trial_name += f"_{archtype.value}"
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

        env_train = AorticArchSingleType(mode="train", archtype=archtype)
        env_eval = AorticArchSingleType(mode="eval", archtype=archtype)
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
        )
        agent_config = os.path.join(config_folder, "agent.yml")
        agent.save_config(agent_config)
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
        )

        reward, success = runner.training_run(
            HEATUP_STEPS,
            TRAINING_STEPS,
            EXPLORE_STEPS_BTW_EVAL,
            CONSECUTIVE_EXPLORE_EPISODES,
            UPDATE_PER_EXPLORE_STEP,
            eval_episodes=None,
            eval_seeds=seeds,
        )
        agent.close()

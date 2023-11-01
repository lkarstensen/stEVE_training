import os
import logging
import argparse
import torch.multiprocessing as mp
import torch
from eve_training.eve_paper.neurovascular.full.env1 import GwOnly
from eve_training.util import get_result_checkpoint_config_and_log_path
from eve_training.eve_paper.neurovascular.full.agent1 import create_agent
from eve_rl import Runner
from eve_bench.neurovascular.full import Neurovascular2Ins

CHECKPOINT = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2023-09-18_110156_full_mt_lstm/checkpoints/best_checkpoint.everl"

RESULT_FILE = (
    "/Users/lennartkarstensen/stacie/eve_training/results/test/replay_test.csv"
)

# EVAL_SEEDS = "1,2,3,5,6,7,8,9,10,12,13,14,16,17,18,21,22,23,27,31,34,35,37,39,42,43,44,47,48,50,52,55,56,58,61,62,63,68,69,70,71,73,79,80,81,84,89,91,92,93,95,97,102,103,108,109,110,115,116,117,118,120,122,123,124,126,127,128,129,130,131,132,134,136,138,139,140,141,142,143,144,147,148,149,150,151,152,154,155,156,158,159,161,162,167,168,171,175"
# EVAL_SEEDS = EVAL_SEEDS.split(",")
# EVAL_SEEDS = [int(seed) for seed in EVAL_SEEDS]
EVAL_SEEDS = list(range(100))
HEATUP_STEPS = 5e5
TRAINING_STEPS = 2e7
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

DEBUG_LEVEL = logging.DEBUG

N_WORKER = 6

if __name__ == "__main__":
    cp = torch.load(CHECKPOINT)

    trainer_device = torch.device("cpu")
    worker_device = torch.device("cpu")
    lr = cp["algo"]["model"]["lr_alpha"]
    hidden_layers = cp["algo"]["model"]["q1"]["body"]["hidden_layers"]
    embedder_nodes = cp["algo"]["model"]["q1"]["head"]["n_nodes"]
    embedder_layers = cp["algo"]["model"]["q1"]["head"]["n_layer"]
    stochastic_eval = cp["algo"]["stochastic_eval"]

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

    intervention = Neurovascular2Ins()
    env_train = GwOnly(intervention=intervention, mode="train", visualisation=False)
    intervention_eval = Neurovascular2Ins()
    env_eval = GwOnly(intervention=intervention_eval, mode="eval", visualisation=False)
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
        N_WORKER,
        stochastic_eval,
        False,
    )

    agent.load_checkpoint(CHECKPOINT)
    print("done")
    checkpoint_folder, _ = os.path.split(CHECKPOINT)
    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
        heatup_action_high=[[35, 3.14], [30, 3.14]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=RESULT_FILE,
        info_results=infos,
        quality_info="success",
    )

    runner.eval(seeds=EVAL_SEEDS)
    agent.close()

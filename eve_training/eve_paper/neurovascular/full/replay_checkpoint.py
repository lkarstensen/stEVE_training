import csv
import datetime
import os
import eve_rl
import eve
from eve.visualisation import SofaPygame, VisualisationDummy
import numpy as np
from eve_training.eve_paper.neurovascular.full.env1 import GwOnly
from eve_bench.neurovascular.full import Neurovascular2Ins
from eve.intervention.vesseltree.vesseltree import find_nearest_branch_to_point


def save_result(result, seed, branch: str, path_ratio: float, steps: int):
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    time = datetime.datetime.today().strftime("%H%M%S")
    with open(RESULT_FILE, "a+", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [result, seed, branch, path_ratio, steps, f"{today}_{time}", CHECKPOINT]
        )


CHECKPOINT = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2023-09-18_110156_full_mt_lstm/checkpoints/best_checkpoint.everl"
RESULT_FILE = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/full/mesh_ben/2023-09-18_110156_full_mt_lstm_eval.csv"
seed = 0

if not os.path.isfile(RESULT_FILE):
    with open(RESULT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(
            [
                "result",
                "seed",
                "branch",
                "path_ratio",
                "steps",
                "timestamp",
                "checkpoint",
            ]
        )

algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(CHECKPOINT)
intervention_eval = Neurovascular2Ins()
env = GwOnly(intervention=intervention_eval, mode="eval", visualisation=False)
env.intervention.make_non_mp()
env.intervention.normalize_action = True
# visu = SofaPygame(env.intervention, env.interim_target)
visu = VisualisationDummy()
env.visualisation = visu


for _ in range(100):
    algo.reset()
    obs, _ = env.reset(seed=seed)
    successes = []
    tip_start = env.intervention.fluoroscopy.tracking2d[0]
    target = env.intervention.target.coordinates2d
    steps = 0
    while True:
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        action = algo.get_eval_action(obs_flat)
        obs, r, terminal, trunc, info = env.step(action)
        steps += 1
        env.render()
        if terminal or trunc:
            tip_end = env.intervention.fluoroscopy.tracking2d[0]
            start_dist = np.linalg.norm(target - tip_start)
            end_dist = np.linalg.norm(target - tip_end)
            path_ratio = 1 - (end_dist / start_dist)

            successes.append(env.intervention.target.reached)
            branch = find_nearest_branch_to_point(
                env.intervention.target.coordinates3d,
                env.intervention.vessel_tree,
            )
            save_result(
                float(env.intervention.target.reached),
                seed,
                branch.name,
                path_ratio,
                steps,
            )
            break
    seed += 1

print(sum(successes) / len(successes))

algo.close()
env.close()

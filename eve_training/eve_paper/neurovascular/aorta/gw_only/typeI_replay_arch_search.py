import csv
import os
import numpy as np

import eve_rl
import eve
from eve.visualisation import SofaPygame

CHECKPOINT = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/aorta/gw_only/typeI_hyperparam_opti/2023-08-01_192350_typeI_archgen_optuna/checkpoints/best_checkpoint.everl"

RESULT_FILE = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/aorta/gw_only/typeI_aortic_arch_search.csv"

VISU = True


def get_random_aortic_arch(rng_seed: int = None):
    rng = np.random.default_rng(rng_seed)

    vessel_seed = rng.integers(0, 2**31)

    width_scaling = rng.random() * 0.6 + 0.7
    heigth_scaling = rng.random() * 0.6 + 0.7

    return eve.intervention.vesseltree.AorticArch(
        arch_type=eve.intervention.vesseltree.ArchType.I,
        seed=vessel_seed,
        rotation_yzx_deg=[0, 0, 0],
        scaling_xyzd=[width_scaling, width_scaling, heigth_scaling, 0.85],
    )


if __name__ == "__main__":
    if not os.path.isfile(RESULT_FILE):
        with open(RESULT_FILE, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(
                [
                    "success",
                    "n_episodes",
                    "vessel_seed",
                    "width_scaling",
                    "height_scaling",
                    "checkpoint",
                ]
            )

    algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(CHECKPOINT)

    for _ in range(1):
        successes = []

        vessel_tree = eve.intervention.vesseltree.AorticArch(
            seed=2018282503,
            rotation_yzx_deg=[0, 0, 0],
            scaling_xyzd=[
                0.8457229734455495,
                0.8457229734455495,
                0.9821829390767296,
                0.85,
            ],
        )

        to_exchange = {eve.intervention.vesseltree.VesselTree: vessel_tree}

        env: eve.Env = eve_rl.util.get_env_from_checkpoint(
            CHECKPOINT, "eval", to_exchange
        )
        env.intervention.normalize_action = True
        env.truncation.max_steps = 400
        if VISU:
            env.intervention.make_non_mp()
            visu = SofaPygame(env.intervention, env.interim_target)
            env.visualisation = visu

        seed = 0

        for _ in range(100):
            algo.reset()
            obs, _ = env.reset(seed=seed)
            seed += 1
            obs_flat, _ = eve_rl.util.flatten_obs(obs)
            while True:
                action = algo.get_eval_action(obs_flat)
                obs, r, terminal, trunc, info = env.step(action)
                obs_flat, _ = eve_rl.util.flatten_obs(obs)
                env.render()
                if terminal or trunc:
                    successes.append(info["success"])
                    break
        with open(RESULT_FILE, "a+", newline="", encoding="utf-8") as csvfile:
            success = sum(successes) / len(successes)

            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(
                [
                    success,
                    len(successes),
                    vessel_tree.seed,
                    vessel_tree.scaling_xyzd[0],
                    vessel_tree.scaling_xyzd[2],
                    CHECKPOINT,
                ]
            )
        env.close()

    algo.close()

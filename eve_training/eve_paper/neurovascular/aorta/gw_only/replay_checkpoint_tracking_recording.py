import os
from typing import List, Tuple
import eve_rl
import eve
from eve.visualisation import SofaPygame, VisualisationDummy
from tqdm import tqdm
from eve.util.coordtransform import tracking3d_to_vessel_cs
import pickle

CHECKPOINT = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94/2023-08-14_112345_arch_vmr_94_lstm/checkpoints/best_checkpoint.everl"
TRAJECTORY_FOLDER = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/neurovascular/aorta/gw_only/arch_vmr_94/trajectories"
SEEDS = [0, 6, 9]


def save_trajectory(
    name: str,
    trajectory: List[Tuple[float, float, float]],
    target_coords: Tuple[float, float, float],
    insertion_point: Tuple[float, float, float],
):
    to_save = {
        "trajectory": trajectory,
        "target": target_coords,
        "insertion_point": insertion_point,
    }
    save_file = os.path.join(TRAJECTORY_FOLDER, f"{name}.pickle")
    with open(save_file, "wb") as pickle_file:
        pickle.dump(to_save, pickle_file)


algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(CHECKPOINT)
env: eve.Env = eve_rl.util.get_env_from_checkpoint(CHECKPOINT)
env.intervention.make_non_mp()
env.intervention.normalize_action = True
env.intervention.devices[0].velocity_limit = (
    30,
    env.intervention.devices[0].velocity_limit[1],
)
# visu = SofaPygame(env.intervention, env.interim_target)
visu = VisualisationDummy()
env.visualisation = visu


for seed in tqdm(SEEDS):
    algo.reset()
    obs, _ = env.reset(seed=seed)
    image_rot_zx = env.intervention.fluoroscopy.image_rot_zx
    image_center = env.intervention.fluoroscopy.image_center
    target = tracking3d_to_vessel_cs(
        env.intervention.target.coordinates3d, image_rot_zx, image_center
    )
    tip_tracking3d = env.intervention.fluoroscopy.tracking3d[0]
    tip_trajectory = [
        tracking3d_to_vessel_cs(
            tip_tracking3d,
            image_rot_zx,
            image_center,
        )
    ]

    insertion_point = tracking3d_to_vessel_cs(
        env.intervention.vessel_tree.insertion.position,
        image_rot_zx,
        image_center,
    )
    while True:
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        action = algo.get_eval_action(obs_flat)
        obs, r, terminal, trunc, info = env.step(action)
        tip_tracking3d = env.intervention.fluoroscopy.tracking3d[0]
        tip_trajectory.append(
            tracking3d_to_vessel_cs(
                tip_tracking3d,
                image_rot_zx,
                image_center,
            )
        )

        env.render()
        if terminal or trunc:
            path = CHECKPOINT
            for _ in range(3):
                path, experiment_name = os.path.split(path)
            save_trajectory(
                f"traj_{experiment_name}_simulation_seed_{seed}",
                tip_trajectory,
                target,
                insertion_point,
            )
            break


algo.close()
env.close()

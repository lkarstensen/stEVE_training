# pylint: disable=no-member,protected-access
import pprint
from time import perf_counter
import csv
import os.path
import numpy as np
import pygame
from eve_bench.cerebral.aorta.simple_cath import ArchGenerator
from eve_training.eve_paper.cerebral.aorta.env2 import TwoDeviceInterimTarget
import multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    seed = 100

    result_file = "/Users/lennartkarstensen/stacie/eve_training/eve_training/eve_paper/cerebral/aorta/test.csv"

    if not os.path.isfile(result_file):
        with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(
                [
                    "reset seed",
                    "success",
                    "n_steps",
                    "target branch",
                    "target",
                ]
            )
        n_episode = 0
    else:
        rowcount = 0
        for row in open(result_file, "r", encoding="utf-8"):
            rowcount += 1
        n_episode = rowcount - 1

    intervention = ArchGenerator()

    env = TwoDeviceInterimTarget(
        intervention=intervention, mode="train", visualisation=True
    )

    n_steps = 0
    r_cum = 0.0
    success_episode = 0.0
    successfull_seeds = []
    steps = []
    while True:
        env.reset()
        while True:
            start = perf_counter()
            trans = 0.0
            rot = 0.0
            camera_trans = np.array((0.0, 0.0, 0.0))
            camera_rot = np.array((0.0, 0.0, 0.0))
            zoom = 0
            pygame.event.get()
            keys_pressed = pygame.key.get_pressed()

            if keys_pressed[pygame.K_ESCAPE]:
                break
            if keys_pressed[pygame.K_UP]:
                trans += 25
            if keys_pressed[pygame.K_DOWN]:
                trans -= 25
            if keys_pressed[pygame.K_LEFT]:
                rot += 1 * 3.14
            if keys_pressed[pygame.K_RIGHT]:
                rot -= 1 * 3.14
            if keys_pressed[pygame.K_r]:
                lao_rao = 0
                cra_cau = 0
                if keys_pressed[pygame.K_d]:
                    lao_rao += 10
                if keys_pressed[pygame.K_a]:
                    lao_rao -= 10
                if keys_pressed[pygame.K_w]:
                    cra_cau -= 10
                if keys_pressed[pygame.K_s]:
                    cra_cau += 10
                env.visualisation.rotate(lao_rao, cra_cau)
            else:
                if keys_pressed[pygame.K_w]:
                    camera_trans += np.array((0.0, 0.0, 200.0))
                if keys_pressed[pygame.K_s]:
                    camera_trans -= np.array((0.0, 0.0, 200.0))
                if keys_pressed[pygame.K_a]:
                    camera_trans -= np.array((200.0, 0.0, 0.0))
                if keys_pressed[pygame.K_d]:
                    camera_trans = np.array((200.0, 0.0, 0.0))
                env.visualisation.translate(camera_trans)
            if keys_pressed[pygame.K_e]:
                env.visualisation.zoom(1000)
            if keys_pressed[pygame.K_q]:
                env.visualisation.zoom(-1000)

            action = (
                ((0, 0), (trans, rot))
                if keys_pressed[pygame.K_v]
                else ((trans, rot), (0, 0))
            )

            s, r, term, trunc, i = env.step(action=action)
            env.render()
            n_steps += 1

            print(s)
            print(round(r, 4))

            if keys_pressed[pygame.K_RETURN] or term:
                success_episode = env.intervention.target.reached
                with open(result_file, "a+", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile, delimiter=";")
                    target = env.intervention.target.coordinates3d
                    target = env.intervention.fluoroscopy.tracking3d_to_vessel_cs(
                        target
                    )
                    target_branch = (
                        env.intervention.vessel_tree.find_nearest_branch_to_point(
                            target
                        )
                    )
                    writer.writerow(
                        [
                            seed,
                            success_episode,
                            n_steps,
                            target_branch.name,
                            target,
                        ]
                    )
                if success_episode:
                    successfull_seeds.append(seed)
                    steps.append(n_steps)
                n_steps = 0
                n_episode += 1
                break
            # print(f"FPS: {1/(perf_counter()-start)}")
        if keys_pressed[pygame.K_ESCAPE]:
            break
        seed += 1
    if successfull_seeds:
        average_steps = sum(steps) / len(steps)
        max_steps = max(steps)
        with open(result_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(successfull_seeds)
            writer.writerow(
                ["n_successfull_episodes", "average steps", "maximum_steps"]
            )
            writer.writerow([len(steps), average_steps, max_steps])
    env.close()

# pylint: disable=no-member,protected-access
from time import perf_counter
import csv
import os.path
import numpy as np
import pygame
from eve.vesseltree import ArchType
from eve_training.eve_paper.aorticharch.env import AorticArchSingleType

start_seed = 100

ARCHTYPE = ArchType.I

result_file = f"/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/aorticarch/one_type/human_play_type_{ARCHTYPE.value}.csv"

if not os.path.isfile(result_file):
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(
            [
                "reset seed",
                "success",
                "n_steps",
                "arch seed",
                "archtype",
                "scale width",
                "scale_heigth",
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

env = AorticArchSingleType(mode="eval", visualisation=True, mp=False)
n_steps = 0
r_cum = 0.0
success_episode = 0.0
successfull_seeds = []
successfull_seeds_branch = {
    "bct": [],
    "lcca": [],
    "rcca": [],
    "lsa": [],
    "rsa": [],
    "co": [],
}
successfull_seeds_archtype = {
    "I": [],
    "II": [],
    "IV": [],
    "Va": [],
    "Vb": [],
    "VI": [],
    "VII": [],
}
steps = []
while True:
    env.reset(seed=start_seed)
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
            trans += 50
        if keys_pressed[pygame.K_DOWN]:
            trans -= 50
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
        action = (trans, rot)
        s, r, term, trunc, i = env.step(action=action)
        env.render()
        n_steps += 1

        print(trunc)

        if keys_pressed[pygame.K_RETURN] or term:
            success_episode = env.target.reached
            arch_type = env.vessel_tree._aortic_arch.arch_type
            with open(result_file, "a+", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                target = env.target.coordinates_vessel_cs
                target_branch = env.vessel_tree.find_nearest_branch_to_point(target)
                writer.writerow(
                    [
                        start_seed,
                        success_episode,
                        n_steps,
                        env.vessel_tree._aortic_arch.seed,
                        arch_type,
                        env.vessel_tree._aortic_arch.scale_xyzd[0],
                        env.vessel_tree._aortic_arch.scale_xyzd[2],
                        target_branch,
                        target,
                    ]
                )
            if success_episode:
                successfull_seeds.append(start_seed)
                successfull_seeds_branch[target_branch.name].append(start_seed)
                successfull_seeds_archtype[arch_type].append(start_seed)
                steps.append(n_steps)
            n_steps = 0
            n_episode += 1
            break
        # print(f"FPS: {1/(perf_counter()-start)}")
    if keys_pressed[pygame.K_ESCAPE]:
        break
    start_seed += 1
if successfull_seeds:
    average_steps = sum(steps) / len(steps)
    max_steps = max(steps)
    with open(result_file, "a+", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(successfull_seeds)
        writer.writerow(["n_successfull_episodes", "average steps", "maximum_steps"])
        writer.writerow([len(steps), average_steps, max_steps])
        writer = csv.DictWriter(csvfile, successfull_seeds_branch.keys(), delimiter=";")
        writer.writeheader()
        writer.writerow(successfull_seeds_branch)
        writer = csv.DictWriter(
            csvfile, successfull_seeds_archtype.keys(), delimiter=";"
        )
        writer.writeheader()
        writer.writerow(successfull_seeds_archtype)
env.close()

# pylint: disable=no-member
import csv
from datetime import datetime
import os
from time import perf_counter
import pygame
from eve_bench.neurovascular.aorta.guidewire_only.arch_generator import ArchGenerator
from eve.util.interventionstatestorage import InterventionStateRecorder
from eve.util.userinput.visumanipulator import VisuManipulator
from eve.util.userinput.instrumentaction import JoyOneDevice, KeyboardOneDevice
from eve.visualisation import SofaPygame

FOLDER = "/Users/lennartkarstensen/stacie/eve_training/recorded_interventions/eve_paper/neurovascular/aorta/gw_only"
RECORD = False

pygame.init()

intervention = ArchGenerator()
visu = SofaPygame(intervention)
visu_manip = VisuManipulator(visu)
instrument_action = KeyboardOneDevice(action_limits=(35, 3.14))

recorder = InterventionStateRecorder(intervention)


seed = 175

r_cum = 0.0

intervention.reset(seed=seed)
visu.reset()
recorder.reset()
last_tracking = None
while True:
    start = perf_counter()
    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()
    if keys_pressed[pygame.K_ESCAPE]:
        break

    visu_manip.step()
    action = instrument_action.get_action()
    intervention.step(action=action)
    visu.render()
    recorder.step()

    if keys_pressed[pygame.K_RETURN] or intervention.target.reached:
        today = datetime.today().strftime("%Y-%m-%d")
        time = datetime.today().strftime("%H%M%S")
        path = os.path.join(FOLDER, f"seed-{seed}_{today}_{time}.eve")
        if RECORD:
            recorder.save_intervention_states(path)
        if intervention.target.reached and seed is not None:
            path_seed = os.path.join(FOLDER, "success_seeds.csv")
            with open(path_seed, "a+", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=";")
                writer.writerow([seed])
        if seed is not None:
            seed += 1
        print(seed)
        intervention.reset(seed=seed)
        visu.reset()
        recorder.reset()
        n_steps = 0

intervention.close()
visu.close()

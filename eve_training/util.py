from dataclasses import dataclass
import os
from typing import List
import csv


def get_result_checkpoint_config_and_log_path(all_results_folder, name):
    file_id = 0
    while True:
        main_resultfile = all_results_folder + f"/{name}_{file_id}.csv"
        if os.path.isfile(main_resultfile):
            file_id += 1
        else:
            break

    results_folder = os.path.join(all_results_folder, f"{name}_{file_id}")

    checkpoint_folder = os.path.join(results_folder, "checkpoints")

    mkdir_recursive(checkpoint_folder)

    log_file = os.path.join(results_folder, "main.log")
    return main_resultfile, checkpoint_folder, results_folder, log_file


def mkdir_recursive(path: str):
    subfolders = []

    while not os.path.isdir(path):
        path, subfolder = os.path.split(path)
        subfolders.append(subfolder)

    for subfolder in reversed(subfolders):
        path = os.path.join(path, subfolder)
        if not os.path.isdir(path):
            os.mkdir(path)


@dataclass
class ResultFile:
    name: str
    path: str


@dataclass
class ResultData:
    name: str
    steps: List[int]
    episodes: List[int]
    rewards: List[float]
    successes: List[float]


def plot_result(
    result_files: List[ResultFile],
    plot_name: str,
    save_plot_path: str,
    legend_outside: bool = False,
):
    results: List[ResultData] = []
    for result_file in result_files:
        result = get_result(result_file.path, result_file.name)
        results.append(result)
    import matplotlib.pyplot as plt

    if legend_outside:
        _, (ax, ax2) = plt.subplots(
            1, 2, gridspec_kw={"width_ratios": [3, 1]}, figsize=(10, 7)
        )
        ax2.set_visible(False)
    else:
        _, ax = plt.subplots(figsize=(10, 7))

    steps = 0
    for result in results:
        ax.plot(result.steps, result.successes, label=result.name)
        steps = max(steps, result.steps[-1])
    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc="best")

    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_xlabel("exploration steps")
    ax.set_xlim(0, steps)
    ax.set_title(plot_name)
    ax.grid(True, which="both", linestyle="--")
    plt.savefig(save_plot_path)


def get_result(path, name):
    steps = [0]
    episodes = [0]
    rewards = [0]
    successes = [0]
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        for _ in range(3):
            row = next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            steps.append(int(row[1]))
            rewards.append(float(row[2]))
            successes.append(float(row[3]))
    result = ResultData(name, steps, episodes, rewards, successes)
    return result

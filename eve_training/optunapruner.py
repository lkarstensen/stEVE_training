from typing import List
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
import optuna
import numpy as np


class CombinationPruner(BasePruner):
    def __init__(self, pruners: List[BasePruner]):
        self.pruners = pruners

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        prune = False
        for pruner in self.pruners:
            prune = prune or pruner.prune(study, trial)
        return prune


class StagnatingPruner(BasePruner):
    def __init__(
        self,
        fluctuation_boundary: float,
        n_averaged_values: int,
        n_strikes: int,
        n_warmup_steps: int,
    ):
        self.fluctuation_boundary = fluctuation_boundary
        self.n_averaged_values = n_averaged_values
        self.n_warmup_steps = n_warmup_steps
        self.n_strikes = n_strikes
        self._below_average_counter = 0
        self._improvement_row_counter = 0

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:

        step = trial.last_step
        if step is None:
            return False

        if step < self.n_warmup_steps:
            return False

        values = np.asarray(list(trial.intermediate_values.values()), dtype=float)
        n_values = values.shape[0]
        if n_values < self.n_averaged_values:
            return False
        rated_average = np.average(values[-self.n_averaged_values :])
        compare_to_average = np.average(values[-self.n_averaged_values - 1 : -1])
        average_diff = rated_average - compare_to_average
        if study.direction == StudyDirection.MINIMIZE:
            average_diff = -average_diff

        if average_diff > 0:
            self._improvement_row_counter += 1
        else:
            self._improvement_row_counter = 0

        if average_diff < self.fluctuation_boundary:
            self._below_average_counter += 1

            if (
                self._below_average_counter >= self.n_strikes
                and self._below_average_counter > self._improvement_row_counter
            ):
                return True
            else:
                return False

        else:
            self._below_average_counter = 0
            return False

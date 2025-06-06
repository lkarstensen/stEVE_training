import eve
import eve.visualisation


class BenchEnv(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.SimulatedIntervention,
        mode: str = "train",
        visualisation: bool = False,
        n_max_steps=1000,
        reward_type: int = 0
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation
        start = eve.start.InsertionPoint(intervention)
        pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
        # Observation

        tracking = eve.observation.Tracking2D(intervention, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        target_state = eve.observation.Target2D(intervention)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        last_action = eve.observation.LastAction(intervention)
        last_action = eve.observation.wrapper.Normalize(last_action)

        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=1.0,
            final_only_after_all_interim=False,
        )
        step_reward = eve.reward.Step(factor=-0.005)
        path_delta = eve.reward.PathLengthDelta(pathfinder, 0.001)

        rewards: list[eve.reward.Reward] = []
        if reward_type == 0:
            rewards = [target_reward, path_delta, step_reward]
        elif reward_type == 1:
            rewards = [target_reward, step_reward]
        elif reward_type == 2:
            rewards = [target_reward, path_delta]
        elif reward_type == 3:
            rewards = [path_delta, step_reward]
        elif reward_type == 4:
            rewards = [target_reward]
        elif reward_type == 5:
            rewards = [path_delta]
        elif reward_type == 6:
            rewards = [step_reward]
        else:
            raise ValueError("Invalid reward type specified.")

        reward = eve.reward.Combination(rewards)

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(intervention)

        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(intervention)
        sim_error = eve.truncation.SimError(intervention)

        if mode == "train":
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])
        else:
            truncation = max_steps

        # Info
        target_reached = eve.info.TargetReached(intervention, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(intervention)
        trajectory_length = eve.info.TrajectoryLength(intervention)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        if visualisation:
            intervention.make_non_mp()
            visu = eve.visualisation.SofaPygame(intervention)
        else:
            intervention.make_mp()
            visu = None
        super().__init__(
            intervention,
            observation,
            reward,
            terminal,
            truncation=truncation,
            start=start,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
            interim_target=None,
        )

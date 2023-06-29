import eve
import eve.visualisation


class TwoDeviceInterimTarget(eve.Env):
    def __init__(
        self,
        intervention: eve.intervention.Intervention,
        mode: str = "train",
        visualisation: bool = False,
    ) -> None:
        self.mode = mode
        self.visualisation = visualisation
        start = eve.start.VesselEnd(intervention)
        # start = eve.start.InsertionPoint(intervention)
        pathfinder = eve.pathfinder.BruteForceBFS(intervention=intervention)
        interim_target = eve.interimtarget.Even(
            pathfinder, intervention, resolution=20, threshold=10
        )
        # Observation

        tracking = eve.observation.TrackingDevice2D(
            intervention, device_idx=0, n_points=3, resolution=2
        )
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, intervention
        )
        tracking_device2 = eve.observation.TrackingDevice2D(
            intervention, device_idx=1, n_points=3, resolution=2, name="cath"
        )
        tracking_device2 = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking_device2, intervention
        )
        target_state = eve.observation.Target2D(
            intervention, interim_target=interim_target
        )
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, intervention
        )
        final_target_state = eve.observation.Target2D(intervention)
        final_target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            final_target_state, intervention
        )
        observation = eve.observation.ObsDict(
            {
                "device1": tracking,
                "device2": tracking_device2,
                "target": target_state,
                # "final_target": final_target_state,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(
            intervention,
            factor=0.02,
            interim_target=interim_target,
            final_only_after_all_interim=True,
        )

        path_delta = eve.reward.TipToTargetDistDelta(
            0.001,
            intervention=intervention,
            interim_target=interim_target,
        )
        device_diff = eve.reward.InsertionLengthRelativeDelta(
            intervention,
            device_id=1,
            relative_to_device_id=0,
            factor=-0.001,
            lower_clearance=-50,
            upper_clearance=-10,
        )
        reward = eve.reward.Combination([target_reward, path_delta, device_diff])

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(intervention)
        if mode == "train":
            n_max_steps = 200
        else:
            n_max_steps = 200
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
            visu = eve.visualisation.SofaPygame(intervention, interim_target)
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
            interim_target=interim_target,
        )

import eve
import eve.visualisation

MAX_STEPS_EVAL = 450
MAX_STEPS_TRAIN = 150

STEP_REWARD = -0.005
TARGET_REWARD = 1.0
PATHLENGHT_REWARD_FACTOR = 0.001

DIAMETER_SCALING = 0.75
LAO_RAO_DEG = 25
CRA_CAU_DEG = 5


class AorticArchSingleType(eve.Env):
    def __init__(
        self,
        mode: str = "train",
        archtype: eve.vesseltree.ArchType = eve.vesseltree.ArchType.I,
        visualisation: bool = False,
        mp: bool = True,
    ) -> None:
        self.visualisation = visualisation
        seed_random = None if mode == "train" else 15
        vessel_tree = eve.vesseltree.AorticArchRandom(
            seed_random=seed_random,
            scale_diameter_array=[DIAMETER_SCALING],
            episodes_between_change=1,
            rotate_z_deg_array=[-LAO_RAO_DEG],
            rotate_x_deg_array=[-CRA_CAU_DEG],
            n_coordinate_space_iters=1,
            arch_types_filter=[archtype],
        )

        device = eve.intervention.device.JWire(velocity_limit=(40, 3.14))
        simulation = eve.intervention.Simulation(
            vessel_tree=vessel_tree,
            devices=[device],
            lao_rao_deg=0,
            cra_cau_deg=0,
            sofacore_mp=mp,
            mp_timeout_step=3,
            mp_restart_n_resets=200,
        )

        # start = eve.start.VesselEnd(simulation, vessel_tree)
        start = eve.start.InsertionPoint(simulation)
        target = eve.target.CenterlineRandom(
            vessel_tree,
            simulation,
            threshold=5,
            branches=["rsa", "rcca", "lcca", "lsa", "bct", "co"],
        )
        pathfinder = eve.pathfinder.BruteForceBFS(vessel_tree, simulation, target)

        # Observation

        tracking = eve.observation.Tracking2D(simulation, n_points=3, resolution=2)
        tracking = eve.observation.wrapper.NormalizeTracking2DEpisode(
            tracking, simulation
        )
        tracking = eve.observation.wrapper.Memory(
            tracking, 2, eve.observation.wrapper.MemoryResetMode.FILL
        )
        target_state = eve.observation.Target2D(target)
        target_state = eve.observation.wrapper.NormalizeTracking2DEpisode(
            target_state, simulation
        )
        last_action = eve.observation.LastAction(simulation)
        last_action = eve.observation.wrapper.Normalize(last_action)
        observation = eve.observation.ObsDict(
            {
                "tracking": tracking,
                "target": target_state,
                "last_action": last_action,
            }
        )

        # Reward
        target_reward = eve.reward.TargetReached(target, factor=TARGET_REWARD)
        step_reward = eve.reward.Step(factor=STEP_REWARD)
        path_delta = eve.reward.PathLengthDelta(pathfinder, PATHLENGHT_REWARD_FACTOR)
        reward = eve.reward.Combination([target_reward, step_reward, path_delta])

        # Terminal and Truncation
        terminal = eve.terminal.TargetReached(target=target)
        if mode == "train":
            n_max_steps = MAX_STEPS_TRAIN
        else:
            n_max_steps = MAX_STEPS_EVAL
        max_steps = eve.truncation.MaxSteps(n_max_steps)
        vessel_end = eve.truncation.VesselEnd(simulation, vessel_tree)
        sim_error = eve.truncation.SimError(simulation)

        if mode == "train":
            truncation = eve.truncation.Combination([max_steps, vessel_end, sim_error])
        else:
            truncation = max_steps

        # Info
        target_reached = eve.info.TargetReached(target, name="success")
        path_ratio = eve.info.PathRatio(pathfinder)
        steps = eve.info.Steps()
        trans_speed = eve.info.AverageTranslationSpeed(simulation)
        trajectory_length = eve.info.TrajectoryLength(simulation)
        info = eve.info.Combination(
            [target_reached, path_ratio, steps, trans_speed, trajectory_length]
        )

        if visualisation:
            visu = eve.visualisation.SofaPygame(simulation, target=target)
        else:
            visu = None
        super().__init__(
            vessel_tree,
            simulation,
            target,
            start,
            observation,
            reward,
            terminal,
            truncation=truncation,
            pathfinder=pathfinder,
            visualisation=visu,
            info=info,
        )

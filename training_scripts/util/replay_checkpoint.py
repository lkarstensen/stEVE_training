import eve_rl
import eve
from eve.visualisation import SofaPygame

checkpoint = "/Users/lennartkarstensen/stacie/eve_training/results/eve_paper/cerebral/aorta/arch_generator1/2023-06-08_00-34_archgen/checkpoints/best_checkpoint.everl"

algo = eve_rl.algo.AlgoPlayOnly.from_checkpoint(checkpoint)
env: eve.Env = eve_rl.util.get_env_from_checkpoint(checkpoint, "eval")
env.intervention.make_non_mp()
env.intervention.normalize_action = True
env.truncation.max_steps = 200
visu = SofaPygame(env.intervention, env.interim_target)

env.visualisation = visu

seed = 1

for _ in range(5):
    algo.reset()
    obs = env.reset(seed=seed)
    obs_flat, _ = eve_rl.util.flatten_obs(obs)
    while True:
        action = algo.get_eval_action(obs_flat)
        obs, r, terminal, trunc, info = env.step(action)
        obs_flat, _ = eve_rl.util.flatten_obs(obs)
        env.render()
        if terminal or trunc:
            break
    seed += 1

algo.close()
env.close()

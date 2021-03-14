import ray
import ray.rllib.agents.ppo as ppo

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


import sys
sys.path.append("..")
from game_env.envs.harvest import HarvestEnv
from game_env.envs.cleanup import CleanupEnv



def setup():

    return


def main(num_agents=5, env='harvest'):
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0


    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
        single_env = HarvestEnv()
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)
        single_env = CleanupEnv()

    env_name = env + "_env"
    register_env(env_name, env_creator)
    

    # trainer = ppo.PPOTrainer(config=config, env=HarvestEnv)


if __name__ == '__main__':
    main()
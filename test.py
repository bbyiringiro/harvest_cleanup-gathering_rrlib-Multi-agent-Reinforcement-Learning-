import argparse
import gym
import os
import random

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved


from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy

from ray.tune.registry import register_env
from model import *

import sys
sys.path.append("..")

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv


tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--stop-reward", type=float, default=150)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--simple", action="store_true")
parser.add_argument("--num-cpus", type=int, default=4)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--as-test", action="store_true")
parser.add_argument(
    "--framework", choices=["tf2", "tf", "tfe", "torch"], default="tf")

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name,  VisionNetwork2)
    env = 'harvest'
    num_agents = 4
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

    obs_space = single_env.observation_space
    act_space = single_env.action_space


    filters= [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]


    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                # "custom_model": model_name,

            },
            # "gamma": random.choice([0.95, 0.99]),
        }
        return (None, obs_space, act_space, config)

    policies = {
        "agent-{}".format(i): gen_policy(i)
        for i in range(num_agents)
    }
    policy_ids = list(policies.keys())


    def policy_mapping_fn(agent_id):
            return agent_id
    


    

    

    config = {
        "env": env_name,
        "env_config": {
            "num_agents": num_agents,
        },
        "horizon": 1000,
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(policy_mapping_fn)#(lambda agent_id: random.choice(policy_ids)),
        },
        "model": {
            "custom_model": model_name,
            "conv_filters":filters,
            # "use_attention": True,
            # "post_fcnet_hiddens": [32, 32]
            # "use_lstm": True,
            # "lstm_use_prev_action":True,
            # "lstm_use_prev_reward":True,
            # "lstm_cell_size": 128
        
        },
        "framework": args.framework,
    }
    stop = {
        # "episode_reward_mean": args.stop_reward,
        # "timesteps_total": args.stop_timesteps,
        # "training_iteration": args.stop_iters,
    }

    results = tune.run("A3C", stop=stop, config=config, verbose=3)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()



# # import sys
# # sys.path.append("..")

# # from social_dilemmas.envs.harvest import HarvestEnv
# # from social_dilemmas.envs.cleanup import CleanupEnv


# # env = HarvestEnv(num_agents=5, render=True)

# # # print(env.agents.items())

# # # env2 = CleanupEnv()

# # # print(env2.agents.items())
# # env.reset()

# import os, random
# import ray
# from ray import tune


# from ray.tune.logger import pretty_print


# from ray.rllib.models import ModelCatalog



# from model import *


# import sys
# sys.path.append("..")
# from social_dilemmas.envs.harvest import HarvestEnv
# from social_dilemmas.envs.cleanup import CleanupEnv



# def setup():

#     return


# def main(num_agents=3, env='harvest'):
#     ray.init()

#     # model_name = "conv_to_fc_net"
#     # ModelCatalog.register_custom_model(model_name,  VisionNetwork2)


#     trainer = PPOTrainer(env="control_env", config={
#     "multiagent": {
#         "policy_mapping_fn": policy_mapper,
#         "policy_graphs": {
#             "supervisor_policy":
#                 (PPOTFPolicy, sup_obs_space, sup_act_space, sup_conf),
#             "worker_p1": (
#                 (PPOTFPolicy, work_obs_s, work_act_s, work_p1_conf),
#             "worker_p2":
#                 (PPOTFPolicy, work_obs_s, work_act_s, work_p2_conf),
#         },
#         "policies_to_train": [
#              "supervisor_policy", "worker_p1", "worker_p2"],
#         },
#     })
#     while True:
#         print(trainer.train())  # distributed training step


#     # if env == 'harvest':
#     #     def env_creator(_):
#     #         return HarvestEnv(num_agents=num_agents)
#     #     single_env = HarvestEnv()
#     # else:
#     #     def env_creator(_):
#     #         return CleanupEnv(num_agents=num_agents)
#     #     single_env = CleanupEnv()

#     # env_name = env + "_env"
#     # register_env(env_name, env_creator)

#     # obs_space = single_env.observation_space
#     # act_space = single_env.action_space
#     # print(obs_space.shape)
#     # import sys
#     # sys.exit()


#     # def policy_mapping_fn(agent_id):
#     #     return agent_id
    
#     # def gen_policy(i):
#     #     return (PGTFPolicy,obs_space, act_space, {})
#     #     # config = {
#     #     #     "model": {
#     #     #         "custom_model": model_name,
#     #     #     },
#     #     #     "gamma": random.choice([0.95, 0.99]),
#     #     # }
#     #     # return (None, obs_space, act_space, config)

#     # policies = {}
#     # for i in range(num_agents):
#     #     policies['agent-' + str(i)] = gen_policy(i)

#     # policy_ids = list(policies.keys())
#     # print(policy_ids)


#     # trainer = PGTrainer(env=env_name, config={
#     # "multiagent": {
#     #     "policies": policies,
#     #     "policy_mapping_fn":tune.function(policy_mapping_fn), 
#     #     },
#     #     "model": {"custom_model": model_name}#, "use_lstm": True,
#     #                     #   "lstm_cell_size": 128}
#     # })

#     # while True:
#     #     result=trainer.train()
#     #     print(result['episode_reward_max'])
#     # # tune.run(PGTrainer, config={
#     # 'env':env_name,
#     # # 'env_config':{
#     # #     'func_createVV':tune.function(env_creator),
#     # #     'run':'KJHHKHK'

#     # # },
#     # #  'framework':'torch',
#     # "multiagent": {
#     #     "policies": policies,
#     #     "policy_mapping_fn":tune.function(policy_mapping_fn), 
#     #     },
#     #     "model": {
#     #         # "custom_model": model_name,
#     #         # "conv_filters":[[6, [3,3],1]],
#     #         # "post_fcnet_hiddens": [32, 32]
#     #     "use_lstm": True,
#     #     "lstm_cell_size": 128}
       
#     # })
   

    

#     # trainer = ppo.PPOTrainer(config=config, env=HarvestEnv)


# if __name__ == '__main__':
#     main()
#     # ray.shutdown()



# from ray import tune
# from ray.tune.registry import register_env
# from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
# from pettingzoo.sisl import waterworld_v0

# # Based on code from github.com/parametersharingmadrl/parametersharingmadrl

# if __name__ == "__main__":
#     # RDQN - Rainbow DQN
#     # ADQN - Apex DQN
#     def env_creator(args):
#         return PettingZooEnv(waterworld_v0.env())

#     env = env_creator({})
#     register_env("waterworld", env_creator)

#     obs_space = env.observation_space
#     act_space = env.action_space

#     policies = {"shared_policy": (None, obs_space, act_space, {})}

#     # for all methods
#     policy_ids = list(policies.keys())

#     tune.run(
#         "APEX_DDPG",
#         stop={"episodes_total": 60000},
#         checkpoint_freq=10,
#         config={

#             # Enviroment specific
#             "env": "waterworld",

#             # General
#             "num_gpus": 1,
#             "num_workers": 2,
#             "num_envs_per_worker": 8,
#             "learning_starts": 1000,
#             "buffer_size": int(1e5),
#             "compress_observations": True,
#             "rollout_fragment_length": 20,
#             "train_batch_size": 512,
#             "gamma": .99,
#             "n_step": 3,
#             "lr": .0001,
#             "prioritized_replay_alpha": 0.5,
#             "final_prioritized_replay_beta": 1.0,
#             "target_network_update_freq": 50000,
#             "timesteps_per_iteration": 25000,

#             # Method specific
#             "multiagent": {
#                 "policies": policies,
#                 "policy_mapping_fn": (lambda agent_id: "shared_policy"),
#             },
#         },
#     )
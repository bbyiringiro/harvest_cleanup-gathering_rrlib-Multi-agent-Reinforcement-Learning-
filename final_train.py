import argparse
import gym
import os
import random
import sys
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

from ray.tune import run_experiments
from ray.rllib.agents.registry import get_trainer_class


from ray.rllib.agents.pg import PGTrainer, PGTFPolicy, PGTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy

from ray.tune.registry import register_env
from model import *

import sys
sys.path.append("utils")

from game_env.envs.harvest import HarvestEnv
from game_env.envs.cleanup import CleanupEnv
from game_env.mycallbacks import HarvestCallback, CleanUPCallback

from arg_extractor  import get_args








harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687
    }




cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': 0.00176}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    if env == 'harvest':
        def env_creator(env_config):
            return HarvestEnv(config=env_config, num_agents=num_agents)
        single_env = HarvestEnv(config={'imrl':{'use':False}})
        callback = HarvestCallback
    else:
        def env_creator(env_config):
            return CleanupEnv(config=env_config, num_agents=num_agents)
        single_env = CleanupEnv(config={'imrl':{'use':False}})
        callback = CleanUPCallback

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        config={}
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policies = {}
    for i in range(num_agents):
        policies['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # register the custom model
    # model_name = "conv_to_fc_net"
    # ModelCatalog.register_custom_model(model_name, VisionNetwork2)

    agent_cls = get_trainer_class(algorithm)
    config = agent_cls._default_config.copy()

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - int(gpus_for_driver))
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        print(num_cpus)
        spare_cpus = (int(num_cpus) - int(cpus_for_driver))
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers


    filters= [
    [6, [3, 3], 2],
    [6, [8, 8], 1]
    ]


    # hyperparams
    config.update({
        "env": env_name,
        "env_config": {
            "num_agents": num_agents,
            "env_name":env_name,
            "run":algorithm,
            "func_create":tune.function(env_creator),
            "visual":True,
            "exp":args.exp_index

        },
        "callbacks": callback,
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(policy_mapping_fn)
        },
        "model": {
            # "custom_model": model_name,
            "dim":15,
            "conv_filters":filters,
            # "use_attention": True,
            "post_fcnet_hiddens": [32, 32],
            # "use_lstm": True,
            # "lstm_use_prev_action":True,
            # "lstm_use_prev_reward":True,
            # "lstm_cell_size": 32

        },
        "framework": args.framework,
        "train_batch_size": train_batch_size,
        "horizon": 1000,
        # "rollout_fragment_length":50,
        "lr_schedule":
        [[0, hparams['lr_init']],
            [2000000, hparams['lr_final']]], #20000000
        # "entropy_coeff": hparams['entropy_coeff'],
        "num_workers": 2, #11
        "num_gpus": 0, #8 # The number of GPUs for the driver
        "num_cpus_for_driver": 1,
        "num_envs_per_worker":8,
        # "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
        "num_cpus_per_worker": 1,   # Can be a fraction
    })
    if args.imrl['use']:
        config['env_config'].update({
            'imrl':{"use":False,
                "imrl_reward_alpha":1, 
                "full_obs":False,
                "fairness_gamma":0.99,
                "fairness_alpha": 1,
                "fairness_epsilon":0.05, #change to optimal
                "reward_gamma": 0.99,
                "reward_alpha": 1,
                "aspirational": 0.5,
                "aspiration_beta": 0.5,
                "f_u": 1,
                "g_v": 1,
                "core":'wf', #'wf',
                "wellbeing_fx":'variance'
        }
        })



    return algorithm, env_name, config


def main(args):
    ray.init()
    # ray.init(address="auto")

    # ray.init(log_to_driver=False)
    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('starting experiment', exp_name)
    # run_experiments({
    #     exp_name: {
    #         "run": alg_run,
    #         "env": env_name,
    #         "stop": {
    #             "training_iteration": args.training_iterations
    #         },
    #         'checkpoint_freq': args.checkpoint_frequency,
    #         "config": config,
    #     }
    # }, verbose=args.verbose, resume=args.resume)

    tune.run(alg_run,
             name=exp_name,
             stop= {
                "training_iteration": args.training_iterations
            },
            checkpoint_freq = args.checkpoint_frequency,
            config = config,
            checkpoint_at_end = True,
            verbose=args.verbose,
            resume=args.resume,
            reuse_actors=args.reuse_actors
        )


if __name__ == '__main__':
    args, device = get_args()

    main(args)
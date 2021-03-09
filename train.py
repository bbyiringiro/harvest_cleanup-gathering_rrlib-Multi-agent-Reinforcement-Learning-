import argparse
import gym
import os
import random

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
sys.path.append("..")

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv







parser = argparse.ArgumentParser()



parser.add_argument("--exp_name", default=None, help='Name of the ray_results experiment directory where results are stored.')
parser.add_argument('--env', default='harvest',
    help='Name of the environment to rollout. Can be cleanup or harvest.')
parser.add_argument('--algorithm', default='A3C',
    help='Name of the rllib algorithm to use.')
parser.add_argument('--num_agents', default=5,
    help='Number of agent policies')
parser.add_argument('--train_batch_size', default=30000,
    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint_frequency', default=100,
    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training_iterations', default=10000,
    help='Total number of steps to train for')
parser.add_argument('--num_cpus',type=int, default=2,
    help='Number of available CPUs')
parser.add_argument('--num_gpus',type=int, default=1,
    help='Number of available GPUs')
parser.add_argument("--use_gpus_for_workers", default=False,
    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument("--use_gpu_for_driver", default=False,
    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument('--num_workers_per_device',type=int, default=2,
    help='Number of workers to place on a single device (CPU or GPU)')
parser.add_argument(
        "--resume",
        action='store_true',
        help="Whether to attempt to resume previous Tune experiments.")
parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Whether to attempt to resume previous Tune experiments.")

parser.add_argument(
    "--framework", choices=["tf2", "tf", "tfe", "torch"], default="tf")

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    # 'entropy_coeff': .000687
    }

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': 0.00176}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

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
    model_name = "conv_to_fc_net"
    ModelCatalog.register_custom_model(model_name, VisionNetwork2)

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



    # hyperparams
    config.update({
        "env": env_name,
        "env_config": {
            "num_agents": num_agents,
            "env_name":env_name,
            "run":algorithm,
            "func_create":tune.function(env_creator),

        },
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(policy_mapping_fn)
        },
        "model": {
            "custom_model": model_name,
            # "conv_filters":filters,
            # "use_attention": True,
            # "post_fcnet_hiddens": [32, 32]
            # "use_lstm": True,
            # "lstm_use_prev_action":True,
            # "lstm_use_prev_reward":True,
            # "lstm_cell_size": 128
        
        },
        "framework": args.framework,
        "train_batch_size": train_batch_size,
        "horizon": 1000,
        "lr_schedule":
        [[0, hparams['lr_init']],
            [20000000, hparams['lr_final']]],
        # "entropy_coeff": hparams['entropy_coeff'],
        "num_workers": num_workers,
        "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
        "num_cpus_for_driver": cpus_for_driver,
        "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
        "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
    })



    return algorithm, env_name, config


def main(args):
    ray.init()
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
    print('Commencing experiment', exp_name)
    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
        }
    }, verbose=args.verbose, resume=args.resume)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

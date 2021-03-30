import argparse
import json
import os
import sys
import GPUtil

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
   
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Complex SSDs')

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
        "--framework", choices=["tf2", "tf", "tfe", "torch"], default="tfe")
    parser.add_argument('--json_file', nargs="?", type=str, default=None,
                        help='')

    args = parser.parse_args()

    if args.json_file is not None:
        args = extract_args_from_json(json_file_path=args.json_file, existing_args_dict=args)

    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)

    import torch

    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()
        print("use {} GPU(s)".format(torch.cuda.device_count()), file=sys.stderr)
    else:
        print("use CPU", file=sys.stderr)
        device = torch.device('cpu')  # sets the device to be CPU

    return args, device


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as f:
        arguments_dict = json.load(fp=f)

    for key, value in vars(existing_args_dict).items():
        if key not in arguments_dict:
            arguments_dict[key] = value

    arguments_dict = AttributeAccessibleDict(arguments_dict)

    return arguments_dict
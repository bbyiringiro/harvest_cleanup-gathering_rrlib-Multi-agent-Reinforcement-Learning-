"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil

from game_env.envs.cleanup import CleanupEnv
from game_env.envs.harvest import HarvestEnv
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--vid_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')), help='Path to directory where videos are saved.')
parser.add_argument('--env', default='harvest',
    help='Name of the environment to rollout. Can be cleanup or harvest.')
parser.add_argument('--render_type', default='prettyz',
    help='Name of the environment to rollout. Can be cleanup or harvest.')
parser.add_argument('--fps', default=5, type=int,
    help='Number of frames per second')
parser.add_argument("--render", action='store_true',
    help='whether to render')

parser.add_argument('--horizon', default=50, type=int,
    help='Number of steps')


# args ={}
# args['vid_path']= os.path.abspath(os.path.join(os.path.dirname(__file__), './videos'))
#     # 'Path to directory where videos are saved.'
# args['env']= 'harvest'
#     # 'Name of the environment to rollout. Can be cleanup or harvest.')
# args['render_type'] = 'prettyz'
#     # 'Can be pretty or fast. Implications obvious.')
# args['fps'] = 20
#     # 'Number of frames per second.')


class Controller(object):

    def __init__(self, env_name='cleanup'):
        self.env_name = env_name
        num_agents = 5
        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(num_agents=num_agents, render=True)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=num_agents, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.env.reset()
        plt.ion()

        # TODO: initialize agents here

    def rollout(self, horizon=50, save_path=None, render=False):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """
        rewards = []
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        for i in range(horizon):
            agents = list(self.env.agents.values())
            action_dim = agents[0].action_space.n
            rand_action = np.random.randint(action_dim, size=5)
            obs, rew, dones, info, = self.env.step({'agent-0': rand_action[0],
                                                    'agent-1': rand_action[1],
                                                    'agent-2': rand_action[2],
                                                    'agent-3': rand_action[3],
                                                    'agent-4': rand_action[4]})
            # print(rew)

            sys.stdout.flush()
            if render:
                if save_path is not None:
                    self.env.render(filename=save_path + 'frame' + str(i).zfill(6) + '.png')
                else:
                    self.env.render()

            rgb_arr = self.env.map_to_colors()
            # rgb_arr = obs['agent-0']
            full_obs[i] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])

        return rewards, observations, full_obs

    def render_rollout(self, horizon=50, path=None,
                       render_type='pretty', fps=8, render=False):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            rewards, observations, full_obs = self.rollout(
                horizon=horizon, save_path=image_path, render=render)
            utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                    video_name=video_name)

            # Clean up images
            shutil.rmtree(image_path)
        else:
            
            rewards, observations, full_obs = self.rollout(horizon=horizon,render=render)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                                   video_name=video_name)


def main():
    c = Controller(env_name=args.env)
    c.render_rollout(path=args.vid_path, horizon=args.horizon, render=args.render, render_type=args.render_type,fps=args.fps)


if __name__ == '__main__':
    args = parser.parse_args()
    main()

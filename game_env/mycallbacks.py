from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from typing import Dict
from ray.rllib.policy import Policy
import numpy as np


class CleanUPCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # print("episode {} (env-idx={}) started.".format(
        #     episode.episode_id, env_index))

        episode.user_data["extrinsic_reward"] = []
        episode.user_data["intrinsic_reward"] = []
        episode.user_data["aggresseviness"] = []
        episode.user_data["clean"] = []


    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        
        
        
        in_reward = 0
        ex_reward = 0
        fire = 0
        clean = 0
        for i in range(2):
            agent_key = 'agent-'+str(i)
            
            info = episode.last_info_for(agent_key)
            if not info: return 
            
            ex_reward +=info.get('exR',0)
            in_reward += info.get('inR',0)
            if info['agent_action'] == 7:
                fire += 1
            elif info['agent_action'] == 8:
                clean += 1
        episode.user_data["extrinsic_reward"].append(ex_reward)
        episode.user_data["intrinsic_reward"].append(in_reward)
        episode.user_data["aggresseviness"].append(fire)
        episode.user_data["clean"].append(clean)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        episode.custom_metrics["ExReward"] = np.sum(episode.user_data["extrinsic_reward"])
        episode.custom_metrics["InReward"] = np.sum(episode.user_data["intrinsic_reward"])
        episode.custom_metrics["aggresseviness"] = np.sum(episode.user_data["aggresseviness"])
        episode.custom_metrics["clean"] = np.sum(episode.user_data["clean"])

class HarvestCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # print("episode {} (env-idx={}) started.".format(
        #     episode.episode_id, env_index))

        episode.user_data["extrinsic_reward"] = []
        episode.user_data["intrinsic_reward"] = []
        episode.user_data["aggresseviness"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        
        in_reward = 0
        ex_reward = 0
        tag_action = 0
        for i in range(2):
            agent_key = 'agent-'+str(i)
            
            info = episode.last_info_for(agent_key)
            if not info: return 
            
            ex_reward +=info.get('exR',0)
            in_reward += info.get('inR',0)
            if info['agent_action'] == 7:
                tag_action += 1
        episode.user_data["extrinsic_reward"].append(ex_reward)
        episode.user_data["intrinsic_reward"].append(in_reward)
        episode.user_data["aggresseviness"].append(tag_action)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        episode.custom_metrics["ExReward"] = np.sum(episode.user_data["extrinsic_reward"])
        episode.custom_metrics["InReward"] = np.sum(episode.user_data["intrinsic_reward"])
        episode.custom_metrics["aggresseviness"] = np.sum(episode.user_data["aggresseviness"])
        
        

    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
    #                       result: dict, **kwargs) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #         policy, result["sum_actions_in_train_batch"]))

    # def on_postprocess_trajectory(
    #         self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: str, policy_id: str, policies: Dict[str, Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict[str, SampleBatch], **kwargs):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1
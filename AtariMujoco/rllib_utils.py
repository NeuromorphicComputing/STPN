import copy

import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from typing import Dict, List, Union, Optional

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork  # added by me

import pdb

torch, nn = try_import_torch()

import ray
from ray import tune
from ray.tune import Experiment
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

###########################################


import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import sys
from STPN.Scripts.Nets import SimpleNetSTPN, SimpleNetSTPMLP

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy], episode: Episode,
                         env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs logged so far).
        assert episode.length == 0, "ERROR: `on_episode_start()` callback should be called right after env reset!"
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        episode.user_data["energy"] = []
        episode.hist_data["energy"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode: Episode,
                        env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, "ERROR: `on_episode_step()` callback should not be called right after env reset!"

        # Calculate energy in the Policy net, store it as class attribute (like it's done for states), read it here
        episode.user_data["energy"].append(worker.get_policy().model._energy.sum(-1).cpu().numpy())  # dims (1, hidden)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode,
                       env_index: int, **kwargs):
        # Do not make sure episode is really done.
        print("episode {} (env-idx={}) ended with length {} and energy {}".format(
            episode.episode_id, env_index, episode.length, np.mean(episode.user_data["energy"]))
        )
        episode.custom_metrics["energy"] = episode.user_data["energy"]
        episode.hist_data["energy"] = episode.user_data["energy"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"].numpy())
        print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
            policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str, policy_id: str,
            policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


class MyCallbacksPong(MyCallbacks):
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode,
                        env_index: int, **kwargs):
        # Make sure this episode is ongoing.
        assert episode.length > 0, "ERROR: `on_episode_step()` callback should not be called right after env reset!"
        # Calculate energy in the Policy net, store it as class attribute (like it's done for states), read it here
        episode.user_data["energy"].append(worker.get_policy().model._energy.sum(-1))  # dims are (1, hidden)
import copy
import pdb

import ray
from ray import tune
from ray.tune import Experiment
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from ray.tune.trial import ExportFormat

from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.agents.registry import get_trainer_class

import gym
from STPN.Scripts.utils import merge


torch, nn = try_import_torch()
from rllib_utils import MyCallbacksPong
from rllib_nets import TorchDiagGaussianClipped, TorchCustomModelEnergy

ModelCatalog.register_custom_model("my_torch_model_energy", TorchCustomModelEnergy)
ModelCatalog.register_custom_action_dist("my_dist", TorchDiagGaussianClipped)


ray.init()

# ============================================================ #
# === Set checkpoint to model to evaluate ==================== #
# === Also set relevant config and model_name further below == #
# ============================================================ #
base_path = os.path.expanduser('~/ray_results/A2C/PongNoFrameskip-v4/')
ckp_to_model = base_path + 'nonplastic_experiment/A2C_PongNoFrameskip-v4_bc9f5_00000_0_seed=0_2022-08-12_16-28-02' \
                           '/checkpoint_015626/checkpoint-15626'

seed = 0



common_configs = {
    "run": "A2C",
    "local_dir": os.path.expanduser("~/ray_results/A2C/PongNoFrameskip-v4/eval/"),
    "stop": {"timesteps_total": 1e6, "agent_timesteps_total": 1e6},
    "restore": ckp_to_model,
    "config":{
        ## Train config
        "env": "PongNoFrameskip-v4",
        "seed": seed,  # tune.grid_search([0,1,2,3,42]),
        "use_critic": True,
        "grad_clip": 40,
        "gamma": 0.99,
        "framework": "torch",
        "rollout_fragment_length": 50,
        "clip_rewards": True,
        "num_envs_per_worker": 1,
        # Number of CPUs to allocate per worker.
        "num_cpus_per_worker": 0.3,
        "num_gpus": 0,
        "callbacks": MyCallbacksPong,
        # Training rollouts will be collected using just the learner
        # process, but evaluation will be done in parallel with two
        # workers. Hence, this run will use 3 CPUs total (1 for the
        # learner + 2 more for evaluation workers).
        "num_workers": 0,
        "evaluation_num_workers": 2,
        # Enable evaluation, once per training iteration.
        "evaluation_interval": 1,
        "in_evaluation": True,
        "evaluation_config": {"explore": False},
        "preprocessor_pref": "rllib", # This returns 1 channel (84,84,1) no frame stacking
        "model": {
            # "custom_action_dist": "my_dist",
            "custom_model": "my_torch_model_energy",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "lstm_state_size": 48,
                "use_mlp": False,
                "use_cnn": True,
                "rnn_type": "lstm",
                "energy_eval": True
            },
            "dim": 84,
            "grayscale": True,
        },
    },
}



# ============= STPN =======================
stpn_configs = {
    "config": {
        "model": {
            "custom_model_config": {
                "lstm_state_size": 64,
                "rnn_type": "stpn",
                "stp": {
                    "learn_plastic_weight_params": True,
                    "learn_plastic_weight_params_dims": [0, 1],
                    "plastic_weight_clamp_val": None,
                    "plastic_weight_connections": "all",
                    "which_relative_layer_input_use_postsynaptic": 2,
                    "plastic_bias": False,
                    "plasticity_type": {
                        "weight": "stp",
                        "bias": "stp"
                    },
                    "plasticity_type_kwargs": {
                        "weight_norm": {
                            "ord": 2
                        },
                        "plastic_weight_norm": {
                            "norm": "G",
                            "ord": 2,
                            "time": "pre"
                        },
                        "weight": {
                            "plastic_weight_dependent_factor": {
                                "learn_parameter": False,
                                "fixed_parameter_value": 0.5
                            }
                        },
                        "bias": {
                            "plastic_bias_dependent_factor": {
                                "learn_parameter": False,
                                "fixed_parameter_value": 0.5
                            }
                        }
                    },
                    "learn_plastic_bias_params": True,
                    "learn_plastic_bias_params_dims": [0],
                    "plastic_weights_init_config": {
                        "weight_lambda": {
                            "mode": "uniform", "mean": 0.5, "spread": 0.5, "hidden_weighting": None
                        },
                        "weight_gamma": {
                            "mode": "uniform", "mean": 0, "spread": 0.001, "hidden_weighting": "both"
                        },
                        "bias_lambda": {
                            "mode": "uniform", "mean": 0.5, "spread": 0.5, "hidden_weighting": None
                        },
                        "bias_gamma": {
                            "mode": "uniform", "mean": 0, "spread": 0.001, "hidden_weighting": "both"
                        }
                    }
                }
}}}}
stpn_configs = merge(copy.deepcopy(common_configs), stpn_configs)

# ========================================================================
#  ===============================================================
# ========================================================================

if "nonplastic_experiment" in ckp_to_model:
    config, model_name = common_configs, "nonplastic_experiment"
elif "stpn_configs" in ckp_to_model:
    config, model_name = stpn_configs, "stpn_configs"  #"stpn_experiment"
else:
    raise ValueError


# === Choices for evaluation =============
config['explore_during_inference'] = False
config['num_episodes_during_inference'] = 1
config['energy_reporting'] = 'per_episode_step'  # available options: ['per_episode', 'per_episode_step']

# === Set up evaluation ==================
# Get the last checkpoint from the above training run.
checkpoint = config["restore"]
# Create new Trainer and restore its state from the last checkpoint.
trainer = get_trainer_class(config['run'])(config=config['config'])
trainer.restore(checkpoint)
# Create the env to do inference in.
env = gym.make(config['config']['env'])
obs = env.reset()
# In case the model needs previous-reward/action inputs, keep track of
# these via these variables here (we'll have to pass them into the
# compute_actions methods below).
init_prev_a = prev_a = None
init_prev_r = prev_r = None
# Set LSTM's initial internal state.
lstm_cell_size = config['config']["model"]["custom_model_config"]["lstm_state_size"]
# we can still get local_worker even when not in local mode
init_state = state = trainer.workers._local_worker.get_policy().get_initial_state()
# if model uses previous action or reward as input, initialise it to not None
if config.get('prev_action', False):
    init_prev_a = prev_a = 0
if config.get('prev_reward', False):
    init_prev_r = prev_r = 0.0
# instantiate result storing
num_episodes = 0
energy_all_episodes, reward_all_episodes, length_all_episodes = [], [], []
energy_this_episode, reward_this_episode, episode_step = [], 0, 0
# run evaluation
while num_episodes < config['num_episodes_during_inference']:
    episode_step += 1
    # Compute an action (`a`).
    a, state_out, _ = trainer.compute_single_action(
        observation=obs,
        state=state,
        prev_action=prev_a,
        prev_reward=prev_r,
        explore=config['explore_during_inference'],
        policy_id="default_policy",
    )
    # get energy
    energy = trainer.workers._local_worker.get_policy().model._energy.sum(-1)
    energy_this_episode.append(energy)
    # Send the computed action `a` to the env.
    obs, reward, done, _ = env.step(a)
    reward_this_episode += reward
    # Is the episode `done`? -> Reset.
    if done:
        print(f"=========================== new episode {num_episodes} ==============================")
        # problem is episodes are of diff lengths, which can't be stored in tensor/array
        episode_length = len(energy_this_episode)
        # Store energy per episode step
        if config['energy_reporting'] == 'per_episode':
            # we average
            energy_all_episodes.append(torch.cat(energy_this_episode).mean().tolist())
        elif config['energy_reporting'] == 'per_episode_step':
            # we keep all
            energy_all_episodes.append(torch.cat(energy_this_episode).squeeze(-1).tolist())
        else:
            raise ValueError
        # store episode results
        reward_all_episodes.append(reward_this_episode)
        length_all_episodes.append(episode_step)

        # reset storing variables
        energy_this_episode, reward_this_episode, episode_step = [], 0, 0
        env.seed(episode_step)
        obs = env.reset()
        num_episodes += 1
        state = init_state
        prev_a = init_prev_a
        prev_r = init_prev_r
    # Episode is still ongoing -> Continue.
    else:
        if reward != 0:
            print(f" At episode step {episode_step} r={reward}, all_r={reward_this_episode} a={a}")
        state = state_out
        if init_prev_a is not None:
            prev_a = a
        if init_prev_r is not None:
            prev_r = reward

# path to save based on checkpoint
store_path = os.path.join(
    config["local_dir"],
    model_name,
    os.path.relpath(
        os.path.dirname(checkpoint), os.path.join(os.path.expanduser("~/ray_results/A2C/PongNoFrameskip-v4/"), model_name)
    )
)
if not os.path.isdir(store_path):
    os.makedirs(store_path)

# save energy
with open(
        os.path.join(store_path,f"{config['num_episodes_during_inference']}_episodes_energy_{config['energy_reporting']}.npy"), 'wb'
    ) as energy_results_file:
    if config['energy_reporting'] == 'per_episode':
        # we average
        energy_to_save = np.array(energy_all_episodes)
    elif config['energy_reporting'] == 'per_episode_step':
        # big array with (episodes, max episode_length)
        max_episode_length = 0
        for this_episode in energy_all_episodes:
            if max_episode_length < len(this_episode):
                max_episode_length = len(this_episode)
        # empty episodes stored with -1
        energy_to_save = -1 * np.ones((len(energy_all_episodes), max_episode_length))
        for i_episode, this_episode in enumerate(energy_all_episodes):
            energy_to_save[i_episode, :len(this_episode)] = this_episode
    else:
        raise ValueError
    print("mean energy consumption", energy_to_save.mean())
    np.save(energy_results_file, energy_to_save) # here we save it as (episodes, 1=mean_energy)

# save reward
with open(os.path.join(store_path,f"{config['num_episodes_during_inference']}_episodes_reward.txt"), 'w') as thefile:
    for item in reward_all_episodes:
        thefile.write("%s\n" % item)

# save episode lengths
with open(os.path.join(store_path,f"{config['num_episodes_during_inference']}_episodes_length.txt"), 'w') as thefile:
    for item in length_all_episodes:
        thefile.write("%s\n" % item)

# close ray session
ray.shutdown()

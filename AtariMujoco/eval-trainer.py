from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
from rllib_nets import TorchMLP_RNNModel, TorchDiagGaussianClipped, TorchCustomModelEnergy

# register RLLib classes
ModelCatalog.register_custom_model("my_torch_model_mlp", TorchMLP_RNNModel)
ModelCatalog.register_custom_model("my_torch_model_energy", TorchCustomModelEnergy)
ModelCatalog.register_custom_action_dist("my_dist", TorchDiagGaussianClipped)

import os
import pickle

import ray
from ray.rllib.agents import ppo
import numpy as np

from rllib_utils import MyCallbacks

# Configure what model to evaluate
pendulum_path = os.path.expanduser('~/ray_results/PPO/InvertedPendulum-v2/')
# model_path = 'STPNf/PPO_InvertedPendulum-v2_a4e43_00001_1_num_gpus=0.5,seed=0_2022-01-25_21-21-11/checkpoint_000489/checkpoint-489'  # noqa

# model_path = 'STPNf/PPO_InvertedPendulum-v2_6bf97_00001_1_num_gpus=0.5000,seed=0_2022-08-12_19-53-29/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'STPNf/PPO_InvertedPendulum-v2_6bf97_00003_3_num_gpus=0.5000,seed=1_2022-08-12_19-53-45/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00006_6_rnn_type=mlp,seed=0_2022-08-12_19-54-10/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00009_9_rnn_type=mlp,seed=1_2022-08-12_19-54-30/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00008_8_rnn_type=rnn,seed=1_2022-08-12_19-54-24/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00005_5_rnn_type=rnn,seed=0_2022-08-12_19-54-03/checkpoint_000489/checkpoint-489'  # noqa
# model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00004_4_rnn_type=lstm,seed=0_2022-08-12_19-53-57/checkpoint_000489/checkpoint-489'  # noqa
model_path = 'nonplastic_experiment/PPO_InvertedPendulum-v2_6bf97_00007_7_rnn_type=lstm,seed=1_2022-08-12_19-54-17/checkpoint_000489/checkpoint-489'  # noqa

checkpoint_path = os.path.join(pendulum_path, model_path)
run = 'PPO'
model_name = 'STPNf'

# initialise ray session
ray.init()

# load model config
run_base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
config_path = os.path.join(run_base_dir, 'params.pkl')
with open(config_path, 'rb') as f:
    config = pickle.load(f)

# convert all the training workers to evaluation workers
num_episodes_during_inference = 10  # default, not sure where to specify it
config["evaluation_num_episodes"] = num_episodes_during_inference
energy_reporting = 'per_episode_step'  # reporting energy per episode step
config['evaluation_num_workers'] = config['num_workers']
config['num_workers'] = 0  # no training workers
# set up energy consumption measurement
config['model']['custom_model'] = "my_torch_model_energy"
config['model']['custom_model_config']['energy_eval'] = True

# custom callbacks
config["callbacks"] = MyCallbacks

# instantiate evaluator
trainer = ppo.PPOTrainer(config=config)
trainer.restore(checkpoint_path)
# run evaluation
evaluation_metrics = trainer.evaluate()

# instantiate energy result storing objects
length_all_episodes = evaluation_metrics['evaluation']['hist_stats']['episode_lengths']
energy_all_episodes, reward_all_episodes = [], []
cum_episode_length = 0
# postprocess energy consumption results
for i_episode, episode_length in enumerate(length_all_episodes):
    if config['model']['custom_model_config']['rnn_type'] in ['lstm', 'rnn']:
        # for some reason this returns with a extra useless dim, we just squeeze it
        energy_all_episodes.append(np.array(
            evaluation_metrics['evaluation']['hist_stats']['energy'][
            cum_episode_length:cum_episode_length + episode_length]
        ).squeeze(-1).squeeze(-1))
    elif config['model']['custom_model_config']['rnn_type'] in ['stpmlp', 'mlp']:
        energy_all_episodes.append(np.array(
            evaluation_metrics['evaluation']['hist_stats']['energy'][
            cum_episode_length:cum_episode_length + episode_length]
        ).squeeze(-1))
    else:
        raise Exception(f"Energy concatenation for {config['model']['custom_model_config']['rnn_type']} not supported")
    reward_all_episodes.append(evaluation_metrics['evaluation']['hist_stats']['episode_reward'][i_episode])
    cum_episode_length += episode_length

# report
print("num_episodes_during_inference", num_episodes_during_inference)
print('num_episodes we have data from', len(evaluation_metrics['evaluation']['hist_stats']['episode_lengths']))

#####################################################
############ store results ##########################
#####################################################
# store path
store_path = os.path.join(os.path.expanduser('~/ray_results/'), run, config['env'], 'eval', model_name, os.path.relpath(
    os.path.dirname(checkpoint_path), os.path.join(os.path.expanduser(f"~/ray_results/{run}/{config['env']}/"), model_name)
))
if not os.path.isdir(store_path):
    os.makedirs(store_path)

# save energy consumption
with open(
        os.path.join(store_path, f"{num_episodes_during_inference}_episodes_energy_{energy_reporting}.npy"), 'wb'
) as energy_results_file:
    if energy_reporting == 'per_episode':
        # we average
        energy_to_save = np.array(energy_all_episodes)
    elif energy_reporting == 'per_episode_step':
        # big array with (episodes, max episode_length)
        max_episode_length = 0
        for this_episode in energy_all_episodes:
            if max_episode_length < len(this_episode):
                max_episode_length = len(this_episode)
        # empty episodes stored with -1
        energy_to_save = -1 * np.ones((len(energy_all_episodes), max_episode_length))
        for i_episode, this_episode in enumerate(energy_all_episodes):
            energy_to_save[i_episode, :len(this_episode)] = this_episode
    print("mean energy consumption", energy_to_save.mean())
    np.save(energy_results_file, energy_to_save)  # here we save it as (episodes, 1=mean_energy)

# save reward
with open(os.path.join(store_path, f"{num_episodes_during_inference}_episodes_reward.txt"), 'w') as thefile:
    for item in reward_all_episodes:
        thefile.write("%s\n" % item)

# save episode lengths
with open(os.path.join(store_path, f"{num_episodes_during_inference}_episodes_length.txt"), 'w') as thefile:
    for item in length_all_episodes:
        thefile.write("%s\n" % item)

ray.shutdown()

import copy
import os
# SET GPU constraints here ==============================================#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # needed to set visible gpus
# set GPUs here, but might want to define explicitly all the available GPUs as an environment variable from command line
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# =======================================================================#

from rllib_nets import TorchMLP_RNNModel, TorchDiagGaussianClipped
from STPN.Scripts.utils import merge

import ray
from ray import tune
from ray.tune import Experiment
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()

ModelCatalog.register_custom_model("my_torch_model_mlp", TorchMLP_RNNModel)
ModelCatalog.register_custom_action_dist("my_dist", TorchDiagGaussianClipped)

ray.init()
this_environment = 'InvertedPendulum-v2'
base_path = os.path.expanduser("~/ray_results/PPO")

common_configs = {
    "run": "PPO",
    "local_dir": os.path.join(base_path, this_environment),
    # (int) – Number of checkpoints to keep. A value of None keeps all checkpoints. Defaults to None.
    # If set, need to provide checkpoint_score_attr.
    "keep_checkpoints_num": None,
    # How many training iterations between checkpoints. A value of 0 (default) disables checkpointing.
    # This has no effect when using the Functional Training API.
    "checkpoint_freq": 100,
    # (str) – Specifies by which attribute to rank the best checkpoint. Default is increasing order.
    # If attribute starts with min- it will rank attribute in decreasing order, i.e. min-validation_loss.
    "checkpoint_score_attr": "episode_reward_mean",
    # (bool) – Whether to checkpoint at the end of the experiment regardless of the checkpoint_freq. Default is False.
    # This has no effect when using the Functional Training API.
    "checkpoint_at_end": True,
    "stop": {"timesteps_total": 1000000, "agent_timesteps_total": 1000000},
    "config": {
        # Train config
        "env": this_environment,
        "seed": tune.grid_search([0, 1]), # tune.grid_search([0, 1, 42]),
        "rollout_fragment_length": 2048,
        "lr_schedule": [
            [0, 0.0003],
            [400000000, 0.000000000001],
        ],
        "num_sgd_iter": 10,  # PPO paper: 10 # RLLib tuned: 32 # RLLib default: 30
        "sgd_minibatch_size": 64,  # PPO paper: 64 # RLLib tuned: 4096 # RLLib default: 128
        "gamma": 0.99,
        "lambda": 0.95,
        "num_workers": 1,  # RLLib tuned: 16
        # PPO paper scheduled, but RLLib doesn't support this
        "clip_param": 0.2,  # RLLib tuned: 0.2 # RLLib default: 0.3
        "vf_loss_coeff": 1.0,  # Like PPO paper # RLLib tuned: 0.5 # RLLib default: 1.0
        "entropy_coeff": 0.01,  # Like PPO paper # RLLib tuned: 0 # RLLib default: 0
        # some obvious ones, not given in original paper
        "batch_mode": "truncate_episodes",
        "use_critic": True,  # RLLib default, RLLib tuned: True
        "grad_clip": 40,
        "framework": "torch",
        "clip_rewards": True,
        "num_envs_per_worker": 1,
        "num_gpus": 0, #1,
        "observation_filter": "NoFilter",  # RLLib tuned: "MeanStdFilter" # RLLib default: "NoFilter"
        # The following message might be displayed:
        #  `train_batch_size` (4000) cannot be achieved with your other settings (num_workers=1 num_envs_per_worker=1
        #  rollout_fragment_length=2048)! Auto-adjusting `rollout_fragment_length` to 4000.
        "train_batch_size": 2048,  # RLLib tuned: 65536 # RLlib default: 4000
        "model": {
            "custom_action_dist": "my_dist",  # also use custom action distribution here
            "custom_model": "my_torch_model_mlp",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "lstm_state_size": 64,
                "use_mlp": False,
                "use_cnn": False,
                "rnn_type": tune.grid_search(["lstm", "rnn", "mlp"]),
            },
            "dim": 84,
            "grayscale": True,
        },
    },
}

stpnf_specific_configs = {
    "config": {
        "num_gpus": tune.grid_search([0, 0.5]),
        "model": {
            "custom_model_config": {
                "rnn_type": "stpmlp",
                "stp": {
                    "learn_plastic_weight_params": True,
                    "learn_plastic_weight_params_dims": [0, 1],
                    "plastic_weight_clamp_val": 0.1,
                    "which_relative_layer_input_use_postsynaptic": 2,
                    "plastic_bias": False,
                    "plasticity_type": {
                        "weight": "stp",
                        "bias": "stp"
                    },
                    # usual stpn configs
                    "plasticity_type_kwargs": {
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
                    },
                },
            }
        }
    }
}
# merge specific and general configs
stpnf_configs = merge(copy.deepcopy(common_configs), stpnf_specific_configs)

# instantiate experiment objects
stpnf_experiments = Experiment("STPNf", **stpnf_configs)
nonplastic_experiments = Experiment("nonplastic_experiment", **common_configs)

# run experiments
tune.run_experiments([stpnf_experiments, nonplastic_experiments])

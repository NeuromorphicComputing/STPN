import copy

import ray
from ray import tune
from ray.tune import Experiment
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
from rllib_nets import TorchMLP_RNNModel, TorchDiagGaussianClipped
from STPN.Scripts.utils import merge

ModelCatalog.register_custom_model("my_torch_model_mlp", TorchMLP_RNNModel)
ModelCatalog.register_custom_action_dist("my_dist", TorchDiagGaussianClipped)

ray.init()  # to debug, pass local_mode=True

common_configs = {
    "run": "A2C",
    "local_dir": "~/ray_results/A2C/PongNoFrameskip-v4/",
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
    "stop": {
        "timesteps_total": 1e8,
        "agent_timesteps_total": 1e8,
        "episode_reward_mean": 20.99
    },
    "config": {
        # Train config
        "env": "PongNoFrameskip-v4",
        "seed": tune.grid_search([0, 1]), #tune.grid_search([0, 1, 2, 3, 42]),
        "use_critic": True,
        "grad_clip": 40,
        "gamma": 0.99,
        "framework": "torch",
        "rollout_fragment_length": 50,
        "clip_rewards": True,
        "num_workers": 64,
        "num_envs_per_worker": 1,
        # Number of CPUs to allocate per worker.
        "num_cpus_per_worker": 0.3,
        "num_gpus": 1,
        "lr_schedule": [
            [0, 0.0001],
            [200000000, 0.000000000001],
        ],
        "preprocessor_pref": "rllib",  # This returns 1 channel (84,84,1), no frame stacking
        "model": {
            "custom_model": "my_torch_model_mlp",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "lstm_state_size": 48,
                "use_mlp": False,
                "use_cnn": True,
                "rnn_type": "lstm",
            },
            "dim": 84,
            "grayscale": True,
        },
    },
}

stpn_specific_configs = {
    "config": {
        "lr_schedule": [
            [0, 0.0007],
            [200000000, 0.000000000001],
        ],
        "model": {
            "custom_model_config": {
                "rnn_type": "stpn",
                "lstm_state_size": 64,
                "stp": {
                    "plasticity_type_kwargs": {
                        "weight_norm": {
                            "ord": 2
                        },
                        "plastic_weight_norm": {
                            "norm": "G",
                            "ord": 2,
                            "time": "pre"
                        }
                    },
                    "plastic_weight_clamp_val": None,
                    "plastic_weight_connections": "all",
                    # common STP configs
                    "learn_plastic_weight_params": True,
                    "learn_plastic_weight_params_dims": [0, 1],
                    "which_relative_layer_input_use_postsynaptic": 2,
                    "plastic_bias": False,
                    "plasticity_type": {
                        "weight": "stp",
                        "bias": "stp"
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

# merge common configs and stpn specific ones
stpn_configs = merge(copy.deepcopy(common_configs), stpn_specific_configs)

# instantiate experiments
stpn_experiments = Experiment("stpn_configs", **stpn_configs)
nonplastic_experiments = Experiment("nonplastic_experiment", **common_configs)

# run experiments, in order
tune.run_experiments([stpn_experiments, nonplastic_experiments])

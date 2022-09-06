import copy

import numpy as np
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

class TorchDiagGaussianClipped(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    @override(ActionDistribution)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2):
        super().__init__(inputs, model)
        mean, log_std = torch.chunk(self.inputs, 2, dim=1)
        # print("mean", mean)  # reuiremens is real # this is 'loc'
        # print("log_std", log_std)  # requiremen is larger than zero # this is 'scale'
        # pdb.set_trace(header='After chunking (unpacking) mean and log_std of Normal dist')
        # TODO LogSumExp or whatever should be applied here
        try:
            self.dist = torch.distributions.normal.Normal(mean, torch.nn.functional.softplus(log_std))
            # self.dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        except Exception as e:
            print("mean", mean)  # reuiremens is real # this is 'loc'
            print("log_std", log_std)  # requiremen is larger than zero # this is 'scale'
            print("e:", e)
            pdb.set_trace()

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self.dist.mean
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return super().logp(actions).sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space, #: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        self.last_sample = self.dist.sample()
        # TODO clip here
        return self.last_sample


class TorchMLP_RNNModel(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=256,  # 200,
            lstm_state_size=128,  # 256, # 128 in mujoco with FC of 200 before, 256 with ConvNet in front
            # action_processor=None,
            use_mlp=True,
            use_cnn=False,
            rnn_type="lstm",
            stp=None,
            energy_eval=False,
    ):

        # pdb.set_trace(header="Who inits this. action_space == num_outputs? obs_pace?")
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # env = gym.make("Pendulum-v0")
        # pdb.set_trace()

        # print(env.unwrapped.get_action_meanings())
        # env.get_action_meanings()
        # pdb.set_trace()
        # breakpoint()
        self.energy_eval = energy_eval  # whether we'll collect energy
        self._energy = None  # to store the energy, then picked up by Callbacks

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.lstm_state_size = lstm_state_size
        self.obs_shape = get_preprocessor(obs_space)(obs_space).shape

        ### v2
        assert (not (use_cnn == use_mlp)) or use_mlp is False
        self.use_mlp = use_mlp
        self.use_conv_net = use_cnn
        if self.use_mlp:
            self.fc1 = nn.Linear(self.obs_size, fc_size)
            rnn_input_size = fc_size
            # self.rnn = nn.LSTM(
            #     fc_size, self.lstm_state_size, batch_first=True)
        elif self.use_conv_net:
            self.conv1 = torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=8, stride=4, bias=True)
            self.conv2 = torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=True)
            rnn_input_size = 9 * 9 * 32
            # self.rnn = nn.LSTM(
            #     9*9*32, self.lstm_state_size, batch_first=True)
        else:
            rnn_input_size = self.obs_size
            # self.rnn = nn.LSTM(
            #     self.obs_size, self.lstm_state_size, batch_first=True)

        supported_rnn_types = ["lstm", "stpn", "rnn", "stpmlp", "mlp", "stpnl"]
        assert rnn_type in supported_rnn_types, \
            f"RNN of type {rnn_type} not supported. Supported types are {supported_rnn_types}"

        self.rnn_type = rnn_type
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_size, self.lstm_state_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                rnn_input_size, self.lstm_state_size, batch_first=True)
        elif self.rnn_type == "stpmlp":
            self.rnn = SimpleNetSTPMLP(rnn_input_size, self.lstm_state_size, stp=stp, bias=True)
        elif self.rnn_type == "stpn":
            self.rnn = SimpleNetSTPN(rnn_input_size, self.lstm_state_size, stp=stp, bias=True)
        elif self.rnn_type == "mlp":
            self.rnn = nn.Linear(rnn_input_size, self.lstm_state_size, bias=True)
        else:
            raise ValueError

        # TODO how to use this in config
        # if len(self.obs_shape) == 1:
        #     # only every other action output, as I think they go [mean, std, mean, std, ...] for each continuous action
        #     self.action_std_idx = (torch.arange(num_outputs)*2 + 1)[:int(num_outputs/2)].type(torch.long) #torch.Tensor([1]).type(torch.long)
        # self.action_processor = self.f_action_processor(action_processor)

        # self.action_branch = nn.Sequential(nn.Linear(self.lstm_state_size, num_outputs), self.action_processor)
        # pdb.set_trace()
        # for some reason num_outputs is sometimes float?
        # self.action_branch = nn.Linear(self.lstm_state_size, int(num_outputs))
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    # def f_action_processor(
    #     self,
    #     # std_idx: torch.Tensor,
    #     action_processor: Optional[str]=None
    # ):
    #     if action_processor is None:
    #         return lambda x: x
    #     elif action_processor == "softplus":
    #         return lambda x: torch.nn.functional.softplus(x)
    #     elif action_processor == "logsoftplus":
    #         return lambda x: torch.log(torch.nn.functional.softplus(x))
    #     else:
    #         raise NotImplementedError

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""

        inputs = input_dict["obs"].float()
        # print("inputs.shape", inputs.shape)
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = inputs.shape[0] // seq_lens.shape[0]
        if max_seq_len == 0:
            print("seq_lens", seq_lens)
            print("inputs.shape[0]", inputs.shape[0])
            print("seq_lens.shape[0]", seq_lens.shape[0])
            print("max_seq_len", max_seq_len)
            ray.util.pdb.set_trace()
            breakpoint()
            pdb.set_trace()
            pdb.set_trace()
            print("after pdb")
        self.time_major = self.model_config.get("_time_major", False)
        # pdb.set_trace()
        inputs = add_time_dimension(
            inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        # add_time_dimension(inputs,max_seq_len=max_seq_len,framework="torch",time_major=self.time_major,).shape
        ### CNN
        # if inputs.shape[0]>1:
        #   # pdb.set_trace()
        # # pdb.set_trace()
        # # print("input_dict['obs'].shape", input_dict["obs"].shape)
        # print("inputs.shape", inputs.shape)
        # inputs = inputs.permute(0, 1, 4, 2, 3) # (batch, seq, H, W, C)

        if self.rnn_type in ['stpn', 'stpmlp', 'stpnl']:
            # state = STPN_states2flat_list(state, stp_model = self.lstm.rnn, supposed_batch_size=inputs.shape[0])
            state = self.rnn.rnn.flat_list_to_tuple_states(state)
        elif self.rnn_type in ['lstm', 'rnn', 'mlp']:
            pass
        else:
            raise NotImplementedError
        if self.use_conv_net is True:
            inputs = inputs.permute(0, 1, 4, 2, 3)

        outputs, actions_out = [], []
        for seq_idx in range(inputs.shape[1]):
            # print(f"inputs.shape at {seq_idx}", inputs.shape)
            # print(f"state[0].shape at {seq_idx}", state[0].shape)
            output, state = self.forward_rnn(inputs[:, seq_idx], state, seq_lens)
            # pdb.set_trace(header="self.num_outputs and output.shape")
            # num_ouputs here is 4, not sure why
            # however I think the torch LSTM returns output (L,N,D *Hout), so we should concatenate these
            # I'd look into why output is 4, and what happens if we don't concat
            # also the fact that i
            # output = torch.reshape(output, [-1, self.num_outputs])
            action_out = self.action_branch(output)
            # pdb.set_trace()
            # # (batch, seq, action)
            # if self.action_std_idx is not None:
            #     # pdb.set_trace()
            #     action_out[...,  self.action_std_idx] = self.action_processor(action_out[...,  self.action_std_idx])

            outputs.append(output)
            actions_out.append(action_out)

        # concatenate the actions and hidden outputs at each timestep, along dim 1 seqlen
        self._features = torch.cat(outputs, dim=1)
        # pdb.set_trace()
        actions_out = torch.cat(actions_out, dim=1)
        # flatten actions from (batch, seq, features) or (batch, seq, C, H, W) -->  (batch * seq, flat_features)
        # pdb.set_trace()
        actions_out = actions_out.reshape(actions_out.shape[0] * actions_out.shape[1], actions_out.shape[-1])
        if self.rnn_type in ["lstm", 'rnn', 'mlp']:
            pass
        elif self.rnn_type in ['stpn', 'stpmlp', 'stpnl']:
            # turn states again to list, to be handled by rllib
            state = self.rnn.rnn.tuple_states_to_flat_list(state, inputs.shape[0])
        else:
            raise NotImplementedError
        # if not isinstance(state, list):
        #     # pdb.set_trace()

        # pdb.set_trace(header='actions_out same as the ones passed to ActionDistribution?')
        return actions_out, state

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.

        if self.rnn_type == "lstm":
            h = [
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
            ]
        elif self.rnn_type == 'rnn':
            h = [
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
            ]
        elif self.rnn_type == 'mlp':
            h = [
                self.rnn.weight.new(1, 1).zero_().squeeze(0),
            ]
        elif self.rnn_type in ['stpn', 'stpmlp', 'stpnl']:
            states = self.rnn.rnn.states_init(batch_size=1, device=self.rnn.rnn.weight.device)
            # squeeze batch dim and flatten to list
            h = []
            for states_for_each_type in states:
                for state in states_for_each_type:
                    h.append(state.squeeze(0))
        else:
            raise NotImplementedError

        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # for some reason it wants value and action flat
        value_out = torch.reshape(self.value_branch(self._features), [-1])
        return value_out

    # @override(ModelV2)
    # def forward_energy_consumtion(
    #         self,
    #         last_observation: TensorType,
    #         # input_dict: Dict[str, TensorType],
    #         state_features: TensorType, # oh no but features only has the hidden state!
    #         # state: List[TensorType],
    #         seq_lens: TensorType
    #     ) -> (TensorType, List[TensorType]):
    #     """Adds time dimension to batch before sending inputs to forward_rnn().
    #     You should implement forward_rnn() in your subclass."""
    #     with torch.no_grad():
    #         self.eval()
    #
    #         # inputs = input_dict["obs"].float()
    #         inputs = last_observation
    #         state=state_features
    #         # print("inputs.shape", inputs.shape)
    #         if isinstance(seq_lens, np.ndarray):
    #             seq_lens = torch.Tensor(seq_lens).int()
    #         max_seq_len = inputs.shape[0] // seq_lens.shape[0]
    #         if max_seq_len == 0:
    #             print("seq_lens", seq_lens)
    #             print("inputs.shape[0]", inputs.shape[0])
    #             print("seq_lens.shape[0]", seq_lens.shape[0])
    #             print("max_seq_len", max_seq_len)
    #             ray.util.pdb.set_trace()
    #             breakpoint()
    #             pdb.set_trace()
    #             pdb.set_trace()
    #             print("after pdb")
    #         self.time_major = self.model_config.get("_time_major", False)
    #         # pdb.set_trace()
    #         inputs = add_time_dimension(
    #             inputs,
    #             max_seq_len=max_seq_len,
    #             framework="torch",
    #             time_major=self.time_major,
    #         )
    #         # add_time_dimension(inputs,max_seq_len=max_seq_len,framework="torch",time_major=self.time_major,).shape
    #         ### CNN
    #         # if inputs.shape[0]>1:
    #         #   # pdb.set_trace()
    #         # # pdb.set_trace()
    #         # # print("input_dict['obs'].shape", input_dict["obs"].shape)
    #         # print("inputs.shape", inputs.shape)
    #         # inputs = inputs.permute(0, 1, 4, 2, 3) # (batch, seq, H, W, C)
    #
    #         if self.rnn_type in ['stpn', 'stpmlp']:
    #             # state = STPN_states2flat_list(state, stp_model = self.lstm.rnn, supposed_batch_size=inputs.shape[0])
    #             state = self.rnn.rnn.flat_list_to_tuple_states(state)
    #         if self.use_conv_net is True:
    #             inputs = inputs.permute(0, 1, 4, 2, 3)
    #
    #         outputs, actions_out = [], []
    #         for seq_idx in range(inputs.shape[1]):
    #             output, state = self.forward_rnn_energy_consumption(inputs[:, seq_idx], state, seq_lens)
    #             # # pdb.set_trace(header="self.num_outputs and output.shape")
    #             # # num_ouputs here is 4, not sure why
    #             # # however I think the torch LSTM returns output (L,N,D *Hout), so we should concatenate these
    #             # # I'd look into why output is 4, and what happens if we don't concat
    #             # # also the fact that i
    #             # # output = torch.reshape(output, [-1, self.num_outputs])
    #             # action_out = self.action_branch(output)
    #             # # pdb.set_trace()
    #             # # # (batch, seq, action)
    #             # # if self.action_std_idx is not None:
    #             # #     # pdb.set_trace()
    #             # #     action_out[...,  self.action_std_idx] = self.action_processor(action_out[...,  self.action_std_idx])
    #
    #             outputs.append(output)
    #             # actions_out.append(action_out)
    #
    #         # concatenate the actions and hidden outputs at each timestep, along dim 1 seqlen
    #         self._features = torch.cat(outputs, dim=1)
    #         # pdb.set_trace()
    #         actions_out = torch.cat(actions_out, dim=1)
    #         # flatten actions from (batch, seq, features) or (batch, seq, C, H, W) -->  (batch * seq, flat_features)
    #         # pdb.set_trace()
    #         actions_out = actions_out.reshape(actions_out.shape[0] * actions_out.shape[1], actions_out.shape[-1])
    #         if self.rnn_type in ["lstm", 'rnn', 'mlp']:
    #             pass
    #         elif self.rnn_type in ['stpn', 'stpmlp']:
    #             # turn states again to list, to be handled by rllib
    #             state = self.rnn.rnn.tuple_states_to_flat_list(state, inputs.shape[0])
    #         else:
    #             raise NotImplementedError
    #         # if not isinstance(state, list):
    #         #     # pdb.set_trace()
    #
    #         # pdb.set_trace(header='actions_out same as the ones passed to ActionDistribution?')
    #         # if we are training, you should do self.train()
    #         return actions_out, state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # pdb.set_trace()
        if self.use_mlp:
            # pdb.set_trace()
            inputs = torch.tanh(self.fc1(inputs))
        elif self.use_conv_net:
            # pdb.set_trace()
            # inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 84, 84)
            inputs = torch.relu(self.conv2(torch.relu(self.conv1(inputs))))
            inputs = torch.flatten(inputs, start_dim=1)  # .unsqueeze(1)

        # flatten the channel, H_out and W_out dims
        # add seq dim for LSTM
        if self.rnn_type == "lstm":
            inputs = inputs.unsqueeze(1)
            # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
            state = [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        elif self.rnn_type == "rnn":
            # inputs = torch.flatten(inputs, start_dim=1).unsqueeze(1)
            # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
            # pdb.set_trace()
            inputs = inputs.unsqueeze(1)
            if len(state) == 0:
                ray.util.pdb.set_trace()
                breakpoint()
                pdb.set_trace()
            state = torch.unsqueeze(state[0], 0)
        elif self.rnn_type in ['stpn', 'stpmlp', 'stpnl']:
            # inputs = torch.flatten(inputs, start_dim=1)
            pass
        elif self.rnn_type == "mlp":
            pass
        else:
            raise NotImplementedError

        if self.rnn_type == 'mlp':
            # pdb.set_trace()
            y = torch.tanh(self.rnn(inputs))
        else:
            # pdb.set_trace()
            y, state = self.rnn(inputs, state)

        if self.rnn_type == "lstm":
            # squeeze the Layer dim at 0
            state = [torch.squeeze(state[0], 0), torch.squeeze(state[1], 0)]
        elif self.rnn_type == "rnn":
            # squeeze the Layer dim at 0
            state = [torch.squeeze(state, 0)]
        elif self.rnn_type in ['stpn', 'stpmlp', 'mlp', 'stpnl']:
            # pass
            y = y.unsqueeze(1)  # following pytorch, return (batch, seq, layer*hidden) when batch_first=True
        # elif self.rnn_type == 'mlp':
        #     pass
        else:
            raise NotImplementedError

        return y, state

    # def forward_rnn_energy_consumption(self, inputs, state, seq_lens):
    #     """Feeds `inputs` (B x T x ..) through the Gru Unit.
    #     Returns the resulting outputs as a sequence (B x T x ...).
    #     Values are stored in self._cur_value in simple (B) shape (where B
    #     contains both the B and T dims!).
    #     Returns:
    #         NN Outputs (B x T x ...) as sequence.
    #         The state batches as a List of two items (c- and h-states).
    #     """
    #     # pdb.set_trace()
    #     if self.use_mlp:
    #         # pdb.set_trace()
    #         inputs = torch.tanh(self.fc1(inputs))
    #     elif self.use_conv_net:
    #         # pdb.set_trace()
    #         # inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 84, 84)
    #         inputs = torch.relu(self.conv2(torch.relu(self.conv1(inputs))))
    #         inputs = torch.flatten(inputs, start_dim=1)#.unsqueeze(1)
    #
    #     # flatten the channel, H_out and W_out dims
    #     # add seq dim for LSTM
    #     if self.rnn_type == "lstm":
    #         inputs = inputs.unsqueeze(1)
    #         # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
    #         state = [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
    #     elif self.rnn_type == "rnn":
    #         # inputs = torch.flatten(inputs, start_dim=1).unsqueeze(1)
    #         # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
    #         # pdb.set_trace()
    #         inputs = inputs.unsqueeze(1)
    #         if len(state) == 0:
    #             ray.util.pdb.set_trace()
    #             breakpoint()
    #             pdb.set_trace()
    #         state = torch.unsqueeze(state[0], 0)
    #     elif self.rnn_type in ['stpn', 'stpmlp']:
    #         # inputs = torch.flatten(inputs, start_dim=1)
    #         pass
    #     elif self.rnn_type == "mlp":
    #         pass
    #     else:
    #         raise NotImplementedError
    #
    #     if self.rnn_type == 'mlp':
    #         # pdb.set_trace()
    #         energy = self.rnn.forward_energy_consumption(inputs)
    #     else:
    #         # pdb.set_trace()
    #         energy, state = self.rnn.forward_energy_consumption(inputs, state)
    #
    #     if self.rnn_type == "lstm":
    #         # squeeze the Layer dim at 0
    #         state = [torch.squeeze(state[0], 0), torch.squeeze(state[1], 0)]
    #     elif self.rnn_type == "rnn":
    #         # squeeze the Layer dim at 0
    #         state = [torch.squeeze(state, 0)]
    #     elif self.rnn_type in ['stpn', 'stpmlp', 'mlp']:
    #         # pass
    #         energy = energy.unsqueeze(1) # following pytorch, return (batch, seq, layer*hidden) when batch_first=True
    #     # elif self.rnn_type == 'mlp':
    #     #     pass
    #     else:
    #         raise NotImplementedError
    #
    #     return energy, state # should I return state or not?


class TorchCustomModelEnergy(TorchModelV2, nn.Module):
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=256,  # 200,
            lstm_state_size=128,  # 256, # 128 in mujoco with FC of 200 before, 256 with ConvNet in front
            # action_processor=None,
            use_mlp=True,
            use_cnn=False,
            rnn_type="lstm",
            stp=None,
            energy_eval=False,
    ):

        # pdb.set_trace(header="Who inits this. action_space == num_outputs? obs_pace?")
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # env = gym.make("Pendulum-v0")
        # pdb.set_trace()

        # print(env.unwrapped.get_action_meanings())
        # env.get_action_meanings()
        # pdb.set_trace()
        # breakpoint()
        self.energy_eval = energy_eval  # whether we'll collect energy
        self._energy = None  # to store the energy, then picked up by Callbacks

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.lstm_state_size = lstm_state_size
        self.obs_shape = get_preprocessor(obs_space)(obs_space).shape

        ### v2
        assert (not (use_cnn == use_mlp)) or use_mlp is False
        self.use_mlp = use_mlp
        self.use_conv_net = use_cnn
        if self.use_mlp:
            self.fc1 = nn.Linear(self.obs_size, fc_size)
            rnn_input_size = fc_size
            # self.rnn = nn.LSTM(
            #     fc_size, self.lstm_state_size, batch_first=True)
        elif self.use_conv_net:
            self.conv1 = torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=8, stride=4, bias=True)
            self.conv2 = torch.nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=True)
            rnn_input_size = 9 * 9 * 32
            # self.rnn = nn.LSTM(
            #     9*9*32, self.lstm_state_size, batch_first=True)
        else:
            rnn_input_size = self.obs_size
            # self.rnn = nn.LSTM(
            #     self.obs_size, self.lstm_state_size, batch_first=True)

        supported_rnn_types = ["lstm", "stpn", "rnn", "stpmlp", "mlp"]
        assert rnn_type in supported_rnn_types, \
            f"RNN of type {rnn_type} not supported. Supported types are {supported_rnn_types}"

        self.rnn_type = rnn_type
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_size, self.lstm_state_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                rnn_input_size, self.lstm_state_size, batch_first=True)
        elif self.rnn_type == "stpmlp":
            self.rnn = SimpleNetSTPMLP(rnn_input_size, self.lstm_state_size, stp=stp, bias=True)
        elif self.rnn_type == "stpn":
            self.rnn = SimpleNetSTPN(rnn_input_size, self.lstm_state_size, stp=stp, bias=True)
        elif self.rnn_type == "mlp":
            self.rnn = nn.Linear(rnn_input_size, self.lstm_state_size, bias=True)
        else:
            raise ValueError

        # TODO how to use this in config
        # if len(self.obs_shape) == 1:
        #     # only every other action output, as I think they go [mean, std, mean, std, ...] for each continuous action
        #     self.action_std_idx = (torch.arange(num_outputs)*2 + 1)[:int(num_outputs/2)].type(torch.long) #torch.Tensor([1]).type(torch.long)
        # self.action_processor = self.f_action_processor(action_processor)

        # self.action_branch = nn.Sequential(nn.Linear(self.lstm_state_size, num_outputs), self.action_processor)
        # pdb.set_trace()
        # for some reason num_outputs is sometimes float?
        # self.action_branch = nn.Linear(self.lstm_state_size, int(num_outputs))
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    # def f_action_processor(
    #     self,
    #     # std_idx: torch.Tensor,
    #     action_processor: Optional[str]=None
    # ):
    #     if action_processor is None:
    #         return lambda x: x
    #     elif action_processor == "softplus":
    #         return lambda x: torch.nn.functional.softplus(x)
    #     elif action_processor == "logsoftplus":
    #         return lambda x: torch.log(torch.nn.functional.softplus(x))
    #     else:
    #         raise NotImplementedError

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""

        inputs = input_dict["obs"].float()
        # print("inputs.shape", inputs.shape)
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = inputs.shape[0] // seq_lens.shape[0]
        if max_seq_len == 0:
            print("seq_lens", seq_lens)
            print("inputs.shape[0]", inputs.shape[0])
            print("seq_lens.shape[0]", seq_lens.shape[0])
            print("max_seq_len", max_seq_len)
            ray.util.pdb.set_trace()
            breakpoint()
            pdb.set_trace()
            pdb.set_trace()
            print("after pdb")
        self.time_major = self.model_config.get("_time_major", False)
        # pdb.set_trace()
        inputs = add_time_dimension(
            inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        # add_time_dimension(inputs,max_seq_len=max_seq_len,framework="torch",time_major=self.time_major,).shape
        ### CNN
        # if inputs.shape[0]>1:
        #   # pdb.set_trace()
        # # pdb.set_trace()
        # # print("input_dict['obs'].shape", input_dict["obs"].shape)
        # print("inputs.shape", inputs.shape)
        # inputs = inputs.permute(0, 1, 4, 2, 3) # (batch, seq, H, W, C)

        if self.rnn_type in ['stpn', 'stpmlp']:
            # state = STPN_states2flat_list(state, stp_model = self.lstm.rnn, supposed_batch_size=inputs.shape[0])
            state = self.rnn.rnn.flat_list_to_tuple_states(state)
        if self.use_conv_net is True:
            inputs = inputs.permute(0, 1, 4, 2, 3)

        outputs, actions_out, energies = [], [], []
        for seq_idx in range(inputs.shape[1]):
            # print(f"inputs.shape at {seq_idx}", inputs.shape)
            # print(f"state[0].shape at {seq_idx}", state[0].shape)

            # # Option 1: is calculate energy separatedly, we repeat the ocmputation for other layers
            # if self.energy_eval is True:
            #     with torch.no_grad():
            #         energy = self.forward_rnn_energy_consumption(inputs[:, seq_idx], state, seq_lens)
            #         energies.append(energy)
            # output, state = self.forward_rnn(inputs[:, seq_idx], state, seq_lens)

            # Option 2: is calculate energy together, we don't repeat the ocmputation for other layers
            if self.energy_eval is True:
                output, state, energy = self.forward_rnn(inputs[:, seq_idx], state, seq_lens)
                energies.append(energy)
            else:
                output, state = self.forward_rnn(inputs[:, seq_idx], state, seq_lens)

            # pdb.set_trace(header="self.num_outputs and output.shape")
            # num_ouputs here is 4, not sure why
            # however I think the torch LSTM returns output (L,N,D *Hout), so we should concatenate these
            # I'd look into why output is 4, and what happens if we don't concat
            # also the fact that i
            # output = torch.reshape(output, [-1, self.num_outputs])
            action_out = self.action_branch(output)
            # pdb.set_trace()
            # # (batch, seq, action)
            # if self.action_std_idx is not None:
            #     # pdb.set_trace()
            #     action_out[...,  self.action_std_idx] = self.action_processor(action_out[...,  self.action_std_idx])

            outputs.append(output)
            actions_out.append(action_out)

        # concatenate the actions and hidden outputs at each timestep, along dim 1 seqlen
        self._features = torch.cat(outputs, dim=1)
        # pdb.set_trace()
        actions_out = torch.cat(actions_out, dim=1)
        if self.energy_eval is True:
            # pdb.set_trace()
            self._energy = torch.cat(energies, dim=1)
        # flatten actions from (batch, seq, features) or (batch, seq, C, H, W) -->  (batch * seq, flat_features)
        # pdb.set_trace()
        actions_out = actions_out.reshape(actions_out.shape[0] * actions_out.shape[1], actions_out.shape[-1])
        if self.rnn_type in ["lstm", 'rnn', 'mlp']:
            pass
        elif self.rnn_type in ['stpn', 'stpmlp']:
            # turn states again to list, to be handled by rllib
            state = self.rnn.rnn.tuple_states_to_flat_list(state, inputs.shape[0])
        else:
            raise NotImplementedError
        # if not isinstance(state, list):
        #     # pdb.set_trace()

        # pdb.set_trace(header='actions_out same as the ones passed to ActionDistribution?')
        return actions_out, state

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.

        if self.rnn_type == "lstm":
            h = [
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
            ]
        elif self.rnn_type == 'rnn':
            h = [
                self.rnn.weight_ih_l0.new(1, self.lstm_state_size).zero_().squeeze(0),
            ]
        elif self.rnn_type == 'mlp':
            h = [
                self.rnn.weight.new(1, 1).zero_().squeeze(0),
            ]
        elif self.rnn_type in ['stpn', 'stpmlp']:
            states = self.rnn.rnn.states_init(batch_size=1, device=self.rnn.rnn.weight.device)

            # squeeze batch dim and flatten to list
            h = []
            for states_for_each_type in states:
                for state in states_for_each_type:
                    h.append(state.squeeze(0))
        else:
            raise NotImplementedError

        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        # for some reason it wants value and action flat
        value_out = torch.reshape(self.value_branch(self._features), [-1])
        return value_out

    # @override(ModelV2)
    # def forward_energy_consumtion(
    #         self,
    #         last_observation: TensorType,
    #         # input_dict: Dict[str, TensorType],
    #         state_features: TensorType, # oh no but features only has the hidden state!
    #         # state: List[TensorType],
    #         seq_lens: TensorType
    #     ) -> (TensorType, List[TensorType]):
    #     """Adds time dimension to batch before sending inputs to forward_rnn().
    #     You should implement forward_rnn() in your subclass."""
    #     with torch.no_grad():
    #         self.eval()
    #
    #         # inputs = input_dict["obs"].float()
    #         inputs = last_observation
    #         state=state_features
    #         # print("inputs.shape", inputs.shape)
    #         if isinstance(seq_lens, np.ndarray):
    #             seq_lens = torch.Tensor(seq_lens).int()
    #         max_seq_len = inputs.shape[0] // seq_lens.shape[0]
    #         if max_seq_len == 0:
    #             print("seq_lens", seq_lens)
    #             print("inputs.shape[0]", inputs.shape[0])
    #             print("seq_lens.shape[0]", seq_lens.shape[0])
    #             print("max_seq_len", max_seq_len)
    #             ray.util.pdb.set_trace()
    #             breakpoint()
    #             pdb.set_trace()
    #             pdb.set_trace()
    #             print("after pdb")
    #         self.time_major = self.model_config.get("_time_major", False)
    #         # pdb.set_trace()
    #         inputs = add_time_dimension(
    #             inputs,
    #             max_seq_len=max_seq_len,
    #             framework="torch",
    #             time_major=self.time_major,
    #         )
    #         # add_time_dimension(inputs,max_seq_len=max_seq_len,framework="torch",time_major=self.time_major,).shape
    #         ### CNN
    #         # if inputs.shape[0]>1:
    #         #   # pdb.set_trace()
    #         # # pdb.set_trace()
    #         # # print("input_dict['obs'].shape", input_dict["obs"].shape)
    #         # print("inputs.shape", inputs.shape)
    #         # inputs = inputs.permute(0, 1, 4, 2, 3) # (batch, seq, H, W, C)
    #
    #         if self.rnn_type in ['stpn', 'stpmlp']:
    #             # state = STPN_states2flat_list(state, stp_model = self.lstm.rnn, supposed_batch_size=inputs.shape[0])
    #             state = self.rnn.rnn.flat_list_to_tuple_states(state)
    #         if self.use_conv_net is True:
    #             inputs = inputs.permute(0, 1, 4, 2, 3)
    #
    #         outputs, actions_out = [], []
    #         for seq_idx in range(inputs.shape[1]):
    #             output, state = self.forward_rnn_energy_consumption(inputs[:, seq_idx], state, seq_lens)
    #             # # pdb.set_trace(header="self.num_outputs and output.shape")
    #             # # num_ouputs here is 4, not sure why
    #             # # however I think the torch LSTM returns output (L,N,D *Hout), so we should concatenate these
    #             # # I'd look into why output is 4, and what happens if we don't concat
    #             # # also the fact that i
    #             # # output = torch.reshape(output, [-1, self.num_outputs])
    #             # action_out = self.action_branch(output)
    #             # # pdb.set_trace()
    #             # # # (batch, seq, action)
    #             # # if self.action_std_idx is not None:
    #             # #     # pdb.set_trace()
    #             # #     action_out[...,  self.action_std_idx] = self.action_processor(action_out[...,  self.action_std_idx])
    #
    #             outputs.append(output)
    #             # actions_out.append(action_out)
    #
    #         # concatenate the actions and hidden outputs at each timestep, along dim 1 seqlen
    #         self._features = torch.cat(outputs, dim=1)
    #         # pdb.set_trace()
    #         actions_out = torch.cat(actions_out, dim=1)
    #         # flatten actions from (batch, seq, features) or (batch, seq, C, H, W) -->  (batch * seq, flat_features)
    #         # pdb.set_trace()
    #         actions_out = actions_out.reshape(actions_out.shape[0] * actions_out.shape[1], actions_out.shape[-1])
    #         if self.rnn_type in ["lstm", 'rnn', 'mlp']:
    #             pass
    #         elif self.rnn_type in ['stpn', 'stpmlp']:
    #             # turn states again to list, to be handled by rllib
    #             state = self.rnn.rnn.tuple_states_to_flat_list(state, inputs.shape[0])
    #         else:
    #             raise NotImplementedError
    #         # if not isinstance(state, list):
    #         #     # pdb.set_trace()
    #
    #         # pdb.set_trace(header='actions_out same as the ones passed to ActionDistribution?')
    #         # if we are training, you should do self.train()
    #         return actions_out, state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        # pdb.set_trace()
        if self.use_mlp:
            # pdb.set_trace()
            inputs = torch.tanh(self.fc1(inputs))
        elif self.use_conv_net:
            # pdb.set_trace()
            # inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 84, 84)
            inputs = torch.relu(self.conv2(torch.relu(self.conv1(inputs))))
            inputs = torch.flatten(inputs, start_dim=1)  # .unsqueeze(1)

        # flatten the channel, H_out and W_out dims
        # add seq dim for LSTM
        if self.rnn_type == "lstm":
            inputs = inputs.unsqueeze(1)
            # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
            state = [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        elif self.rnn_type == "rnn":
            # inputs = torch.flatten(inputs, start_dim=1).unsqueeze(1)
            # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
            # pdb.set_trace()
            inputs = inputs.unsqueeze(1)
            if len(state) == 0:
                ray.util.pdb.set_trace()
                breakpoint()
                pdb.set_trace()
            state = torch.unsqueeze(state[0], 0)
        elif self.rnn_type in ['stpn', 'stpmlp']:
            # inputs = torch.flatten(inputs, start_dim=1)
            pass
        elif self.rnn_type == "mlp":
            pass
        else:
            raise NotImplementedError

        # evaluate energy if needed
        if self.energy_eval is True:
            with torch.no_grad():
                if self.rnn_type == "lstm":
                    x_2 = torch.mul(inputs, inputs)
                    h_2 = torch.mul(state[0][0], state[0][0])
                    # pdb.set_trace()

                    energy = torch.nn.functional.linear(x_2, self.rnn.weight_ih_l0.abs()) + \
                             torch.nn.functional.linear(h_2, self.rnn.weight_hh_l0.abs())
                elif self.rnn_type == "rnn":
                    # TODO: same as LSTM, don't duplicate code
                    x_2 = torch.mul(inputs, inputs)
                    h_2 = torch.mul(state[0][0], state[0][0])
                    # pdb.set_trace()
                    energy = torch.nn.functional.linear(x_2, self.rnn.weight_ih_l0.abs()) + \
                             torch.nn.functional.linear(h_2, self.rnn.weight_hh_l0.abs())
                    # raise NotImplementedError
                elif self.rnn_type in ['stpn', 'stpmlp']:
                    energy = self.rnn.forward_energy_consumption(inputs, state)
                elif self.rnn_type == "mlp":
                    x_2 = torch.mul(inputs, inputs)
                    energy = torch.nn.functional.linear(x_2, self.rnn.weight.abs())
                else:
                    raise Exception(f'Energy consumption measurement fo rnn_type {self.rnn_type} not supported')

        # make the actual forward pass
        if self.rnn_type == 'mlp':
            # pdb.set_trace()
            y = torch.tanh(self.rnn(inputs))
        else:
            # pdb.set_trace()
            y, state = self.rnn(inputs, state)

        if self.rnn_type == "lstm":
            # squeeze the Layer dim at 0
            state = [torch.squeeze(state[0], 0), torch.squeeze(state[1], 0)]
        elif self.rnn_type == "rnn":
            # squeeze the Layer dim at 0
            state = [torch.squeeze(state, 0)]
        elif self.rnn_type in ['stpn', 'stpmlp', 'mlp']:
            # pass
            y = y.unsqueeze(1)  # following pytorch, return (batch, seq, layer*hidden) when batch_first=True
        # elif self.rnn_type == 'mlp':
        #     pass
        else:
            raise NotImplementedError

        if self.energy_eval is True:
            return y, state, energy
        else:
            return y, state

    # def forward_rnn_energy_consumption(self, inputs, state, seq_lens):
    #     """Feeds `inputs` (B x T x ..) through the Gru Unit.
    #     Returns the resulting outputs as a sequence (B x T x ...).
    #     Values are stored in self._cur_value in simple (B) shape (where B
    #     contains both the B and T dims!).
    #     Returns:
    #         NN Outputs (B x T x ...) as sequence.
    #         The state batches as a List of two items (c- and h-states).
    #     """
    #     # pdb.set_trace()
    #     if self.use_mlp:
    #         # pdb.set_trace()
    #         inputs = torch.tanh(self.fc1(inputs))
    #     elif self.use_conv_net:
    #         # pdb.set_trace()
    #         # inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], 84, 84)
    #         inputs = torch.relu(self.conv2(torch.relu(self.conv1(inputs))))
    #         inputs = torch.flatten(inputs, start_dim=1)#.unsqueeze(1)
    #
    #     # flatten the channel, H_out and W_out dims
    #     # add seq dim for LSTM
    #     if self.rnn_type == "lstm":
    #         inputs = inputs.unsqueeze(1)
    #         # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
    #         state = [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
    #     elif self.rnn_type == "rnn":
    #         # inputs = torch.flatten(inputs, start_dim=1).unsqueeze(1)
    #         # add Layer dimension at 0, pytorch doesn't care if batch_first=True when it comes to hidden states
    #         # pdb.set_trace()
    #         inputs = inputs.unsqueeze(1)
    #         if len(state) == 0:
    #             ray.util.pdb.set_trace()
    #             breakpoint()
    #             pdb.set_trace()
    #         state = torch.unsqueeze(state[0], 0)
    #     elif self.rnn_type in ['stpn', 'stpmlp']:
    #         # inputs = torch.flatten(inputs, start_dim=1)
    #         pass
    #     elif self.rnn_type == "mlp":
    #         pass
    #     else:
    #         raise NotImplementedError
    #
    #     if self.rnn_type == 'mlp':
    #         # pdb.set_trace()
    #         energy = self.rnn.forward_energy_consumption(inputs)
    #     else:
    #         # pdb.set_trace()
    #         energy, state = self.rnn.forward_energy_consumption(inputs, state)
    #
    #     if self.rnn_type == "lstm":
    #         # squeeze the Layer dim at 0
    #         state = [torch.squeeze(state[0], 0), torch.squeeze(state[1], 0)]
    #     elif self.rnn_type == "rnn":
    #         # squeeze the Layer dim at 0
    #         state = [torch.squeeze(state, 0)]
    #     elif self.rnn_type in ['stpn', 'stpmlp', 'mlp']:
    #         # pass
    #         energy = energy.unsqueeze(1) # following pytorch, return (batch, seq, layer*hidden) when batch_first=True
    #     # elif self.rnn_type == 'mlp':
    #     #     pass
    #     else:
    #         raise NotImplementedError
    #
    #     return energy, state # should I return state or not?
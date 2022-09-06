import argparse
import math
from typing import Optional, Tuple

import torch

from STPN.AssociativeRetrievalTask.data_art import load_data
from STPN.Scripts.utils import DATA


class STPNR(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
    ):
        super().__init__()

        # --- Network sizes ---
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        # --- Parameters ---
        init_k = 1 / math.sqrt(self._hidden_size)
        # STP parameters: lambda (decay rate) and gamma (meta-learning rate)
        self.weight_lambda = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size).uniform_(0, 1))
        self.weight_gamma = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size).uniform_(
            -0.001 * init_k, 0.001 * init_k))
        # RNN weight and bias
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size).uniform_(-init_k, init_k))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size).uniform_(-init_k, init_k))
        # Readout layer
        self.hidden2tag = torch.nn.Linear(hidden_size, output_size)

    def forward(self, sentence: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # Batch first assumption, seq second
        batch_size, seq_len = sentence.size()[:2]
        device = sentence.device

        if states is None:
            # neuronal activations and synaptic states (short-term component F), respectively
            states = (torch.zeros((batch_size, self._hidden_size), device=device),
                      torch.zeros((batch_size, self._hidden_size, self._hidden_size + self._input_size), device=device))

        for seq_element_idx in range(seq_len):
            seq_element = sentence[:, seq_element_idx]
            # treat all presynaptic inputs (neuronal and environment) equally
            total_input = torch.cat((seq_element, states[0]), dim=1)
            # combine fixed and plastic weights into one efficacy matrix G, which is normalised
            total_weights = self.weight + states[1]
            # affine transformation, with optional biases, and activation function
            h_tp1 = torch.tanh(torch.einsum('bf,bhf->bh', total_input, total_weights) + self.bias)

            # compute norm per row (add small number to avoid division by 0)
            norm = torch.linalg.norm(total_weights, ord=2, dim=2, keepdim=True) + 1e-16
            # norm output instead of total_weights for lower memory footprint
            h_tp1 = h_tp1 / norm.squeeze(-1)
            # normalise synaptic weights
            f_tp1 = states[1] / norm

            # STP update: scale current memory, add a scaled association between presynaptic and posynaptic activations
            f_tp1 = self.weight_lambda * f_tp1 + self.weight_gamma * torch.einsum('bf,bh->bhf', total_input, h_tp1)
            # update neuronal memories
            states = (h_tp1, f_tp1)

        # readout to get classes scores
        tag_space = self.hidden2tag(h_tp1)
        return tag_space, states


def main(config):
    batch_size, output_size = 128, 37
    # Dataloaders, will generate datasets if not found in DATA path
    train_dataloader, validation_dataloader, test_dataloader = load_data(
        batch_size=batch_size,
        data_path=DATA,
        onehot=True,
        train_size=100000,
        valid_size=10000,
        test_size=20000,
    )

    device = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1 else torch.device('cpu')

    net = STPNR(
        input_size=37,  # alphabet character + digits + '?' sign
        hidden_size=11,
        output_size=output_size,  # predict possibly any character
    )

    # set up learning
    net.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    n_epochs, best_validation_acc = 200, -float('inf')
    for i_epoch in range(1, n_epochs + 1):
        for sentence, tags in train_dataloader:
            if sentence.size()[0] != batch_size:
                # drop last batch which isn't full
                continue
            net.zero_grad()
            sentence, tags = sentence.to(device), tags.to(device)
            tag_scores, _ = net(sentence, states=None)
            # flatten the batch and sequences, so all frames are treated equally
            loss = loss_function(tag_scores.view(-1, output_size), tags.view(-1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc = 0
            for sentence, tags in validation_dataloader:
                if sentence.size()[0] != batch_size:
                    continue  # drop last batch which isn't full
                sentence, tags = sentence.to(device), tags.to(device)
                tag_scores, _ = net(sentence, states=None)
                # add correct predictions
                acc += (tag_scores.argmax(dim=1) == tags.to(device)).float().sum().cpu().item()
            # normalise to number of samples in validation set
            last_validation_acc = acc / math.prod(list(validation_dataloader.dataset.tensors[1].size()))
        if last_validation_acc > best_validation_acc:
            best_validation_acc = last_validation_acc
        if i_epoch % 10 == 0:
            print('-' * 45, 'Epoch', i_epoch, '-' * 45)
            print("Best validation accuracy", best_validation_acc)
            print("Last validation accuracy", last_validation_acc)
            print("Last loss", f"{loss.cpu().item():4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, help="id of gpu used for gradient backpropagation", default=0)
    parsed_args = parser.parse_args()
    main(vars(parsed_args))

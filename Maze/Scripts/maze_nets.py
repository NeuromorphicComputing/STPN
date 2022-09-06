from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F  # noqa
from torch import no_grad, Tensor

from STPN.Scripts.Nets import NetParallelEnergy

NBDA = 1  # Number of different DA output neurons. Code assumes NBDA=1 and will NOT WORK if you change this # noqa

np.set_printoptions(precision=4)

ADDINPUT = 4  # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs # noqa

NBACTIONS = 4  # Up, Down, Left, Right # noqa

RFSIZE = 3  # Receptive Field # noqa

TOTALNBINPUTS = RFSIZE * RFSIZE + ADDINPUT + NBACTIONS  # noqa


class MiconiNetwork(nn.Module):
    """
    Network just like Miconi's, but with some extra options like weight-norm, energy consumption measurement and some
    speed-ups
    """

    def __init__(self, params):
        super(MiconiNetwork, self).__init__()
        self.clamp = float(params['clamp'])
        print("self.clamp", self.clamp)
        self.type = params['type']
        self.softmax = torch.nn.functional.softmax
        self.activ = F.tanh
        if params['type'] == 'rnn':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).to(params['device'])
            self.w = torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).to(params['device']),
                                        requires_grad=True)
        elif params['type'] == 'lstm':
            self.lstm = torch.nn.LSTMCell(TOTALNBINPUTS, params['hs'], bias=True).to(params['device'])
        elif params['type'] == 'modplast':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).to(params['device'])
            self.w = torch.nn.Parameter((.01 * torch.t(torch.rand(params['hs'], params['hs']))).to(params['device']),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter(
                (.01 * torch.t(torch.rand(params['hs'], params['hs']))).to(params['device']), requires_grad=True)
            self.h2DA = torch.nn.Linear(params['hs'], NBDA).to(params['device'])
        elif params['type'] == 'plastic':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).to(params['device'])
            self.w = torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).to(params['device']),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).to(params['device']),
                                            requires_grad=True)
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).to(params['device']),
                                          requires_grad=True)  # Everyone has the same eta
        elif params['type'] == 'modul' or params['type'] == 'modul2':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hs']).to(params['device'])
            self.w = torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).to(params['device']),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter((.01 * torch.rand(params['hs'], params['hs'])).to(params['device']),
                                            requires_grad=True)
            self.etaet = torch.nn.Parameter((.01 * torch.ones(1)).to(params['device']),
                                            requires_grad=True)  # Everyone has the same etaet
            self.h2DA = torch.nn.Linear(params['hs'], NBDA).to(params['device'])
        else:
            raise ValueError("Which network type?")
        self.h2o = torch.nn.Linear(params['hs'], NBACTIONS).to(params['device'])
        self.h2v = torch.nn.Linear(params['hs'], 1).to(params['device'])
        self.params = params

        self.eval_energy = params['eval_energy']
        if self.eval_energy is True:
            self.zero_idx_energy = params['energy'].get('zero_idx', None)
        else:
            self.zero_idx_energy = None
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order,
        # following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

    def forward(self, inputs, hidden, hebb, et, pw):
        BATCHSIZE = self.params['bs']  # noqa
        HS = self.params['hs']  # noqa

        if self.type == 'rnn':
            if self.eval_energy is True:
                # evaluate energy consumption
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    # remove some input indeces for energy calculation, like reward
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    # P =  V^2 * G <-- x^2 * W
                    x_energy_2 = torch.mul(x_energy, x_energy)
                    h_2 = torch.mul(hidden, hidden)
                    total_inputs = torch.cat((x_energy_2, h_2), dim=1)
                    weight = torch.cat((self.i2h.weight, self.w), dim=1).abs()
                    if self.params['wN'] is not None:
                        norm = torch.linalg.norm(
                            input=weight,
                            ord=2,  # specify type of norm, frobenius, euclidean =2, etc
                            dim=1,  # normalise across batch (dims are bhf)
                        ).unsqueeze(-1)
                        weight = weight / (norm + 1e-15)
                    energy = torch.nn.functional.linear(total_inputs, weight)

            total_inputs = torch.cat((inputs, hidden), dim=1)
            weight = torch.cat((self.i2h.weight, self.w), dim=1)
            # weight norm option, not in Miconi's work
            if self.params['wN'] is not None:
                norm = torch.linalg.norm(
                    input=weight,
                    ord=2,  # specify type of norm, frobenius, euclidean =2, etc
                    dim=1,  # normalise across batch (dims are bhf)
                ).unsqueeze(-1)
                weight = weight / (norm + 1e-15)
            hactiv = self.activ(torch.nn.functional.linear(total_inputs, weight, bias=self.i2h.bias))
            hidden = hactiv
            activout = self.h2o(hactiv)  # Linear! To be softmax'ed outside the function
            valueout = self.h2v(hactiv)

        elif self.type == 'lstm':
            # measure energy consumption, P=V^2 * G <-- x^2 * W
            if self.eval_energy is True:
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    x_2 = torch.mul(x_energy, x_energy)
                    h_2 = torch.mul(hidden[0], hidden[0])
                    weight_ih = torch.empty_like(self.lstm.weight_ih).copy_(self.lstm.weight_ih)
                    weight_ih = weight_ih.abs()
                    weight_hh = torch.empty_like(self.lstm.weight_hh).copy_(self.lstm.weight_hh)
                    weight_hh = weight_hh.abs()
                    energy = x_2 @ weight_ih.t() + h_2 @ weight_hh.t()
            hactiv, c = self.lstm(inputs, hidden)
            hidden = [hactiv, c]
            activout = self.h2o(hactiv)  # Linear! To be softmax'ed outside the function
            valueout = self.h2v(hactiv)

        elif self.type == 'plastic':
            if self.eval_energy is True:
                raise NotImplementedError
            # Each row of w and hebb contains the input weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul((self.w + torch.mul(self.alpha, hebb)), hidden.view(BATCHSIZE, HS, 1))
                                ).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # batched outer product...should it be other way round?
            deltahebb = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))

            if self.params['addpw'] == 3:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Hard clamp
                clamp_val = self.params.get('extra_config', {}).get('clamp', 1.0)
                hebb = torch.clamp(hebb + self.eta * deltahebb, min=-clamp_val, max=clamp_val)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                # Soft clamp
                hebb = torch.clamp(hebb + torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +
                                   torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1), min=-1.0, max=1.0)
            elif self.params['addpw'] == 1:  # Purely additive, tends to make the meta-learning diverge. No decay/clamp.
                hebb = hebb + self.eta * deltahebb
            elif self.params['addpw'] == 0:
                # We do it the normal way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient.
                hebb = (1 - self.eta) * hebb + self.eta * deltahebb

            hidden = hactiv

        elif self.type == 'modplast':
            # measure energy consumption P = V^2 * G = x^2 * W
            if self.eval_energy is True:
                with no_grad():
                    x_energy = torch.empty_like(inputs).copy_(inputs)
                    if self.zero_idx_energy is not None:
                        x_energy[:, self.zero_idx_energy] = 0
                    i2h_w = torch.empty_like(self.i2h.weight).copy_(self.i2h.weight)
                    h2h_w = self.w + torch.mul(self.alpha, hebb)
                    weight = torch.cat((i2h_w.expand(h2h_w.shape[0], *i2h_w.shape), h2h_w), dim=2)
                    if self.params['wN'] is not None and 'weight_norm' in self.params['wN']:
                        norm = torch.linalg.norm(input=weight, ord=2, dim=2).unsqueeze(-1)
                        weight = weight / (norm + 1e-15)
                    x_energy_2 = torch.mul(x_energy, x_energy)
                    h_2 = torch.mul(hidden, hidden)

                    total_inputs = torch.cat((x_energy_2, h_2), dim=1)
                    energy = torch.einsum('bhf,bf->bh', weight.abs(), total_inputs)
            # Here we compute the same deltahebb for the whole network, and use
            # the same addpw for the whole network too.

            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y

            i2h_w = torch.empty_like(self.i2h.weight).copy_(self.i2h.weight)
            h2h_w = self.w + torch.mul(self.alpha, hebb)  # torch.empty_like(self.i2h.weight).copy_(self.i2h.weight)
            weight = torch.cat((i2h_w.expand(h2h_w.shape[0], *i2h_w.shape), h2h_w), dim=2)
            norm = None
            # normalise total effective weights
            if self.params['wN'] is not None and 'weight_norm' in self.params['wN']:
                norm = torch.linalg.norm(input=weight, ord=2, dim=2).unsqueeze(-1)
                weight = weight / (norm + 1e-15)
            total_inputs = torch.cat((inputs, hidden), dim=1)
            hactiv = torch.einsum('bhf,bf->bh', weight, total_inputs)
            if self.i2h.bias is not None:
                hactiv += self.i2h.bias
            hactiv = self.activ(hactiv)
            # normalise plastic weights separatedly prior to plastic update
            if self.params['wN'] is not None and 'plastic_weight_norm' in self.params['wN'] and \
                    self.params['wN']['plastic_weight_norm']['time'] == 'pre':
                # use total effective weight norm
                if self.params['wN']['plastic_weight_norm']['norm'] == 'G':
                    if norm is not None:
                        hebb = hebb / (norm + 1e-15)
                    else:
                        norm = torch.linalg.norm(input=weight, ord=2, dim=2).unsqueeze(-1)
                        hebb = hebb / (norm + 1e-15)
                # use plastic weight norm
                elif self.params['wN']['plastic_weight_norm']['norm'] == 'F':
                    norm = torch.linalg.norm(input=hebb, ord=2, dim=2).unsqueeze(-1)
                    hebb = hebb / (norm + 1e-15)
                else:
                    raise Exception(f"Invalid self.params['wN']['plastic_weight_norm']['norm'] "
                                    f"{self.params['wN']['plastic_weight_norm']['norm']}")

            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA,
            # but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))  # noqa
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))  # noqa
            elif self.params['da'] == 'lin':
                DAout = self.h2DA(hactiv)  # noqa
            else:
                raise ValueError("Which transformation for DAout ?")

            # deltahebb has shape BS x HS x HS
            # Each row of hebb contain the input weights to a neuron
            # batched outer product...should it be other way round?
            deltahebb = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))

            if self.params['addpw'] == 3:  # Hard clamp, purely additive
                # Note that we do the same for Hebb and Oja's rule
                hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-self.clamp,
                                    max=self.clamp)  # , min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                # Note that there is no decay, even in the Hebb-rule case : additive only!
                hebb1 = torch.clamp(hebb + torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=0.0) * (1 - hebb) +
                                    torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, max=0.0) * (hebb + 1),
                                    min=-1.0, max=1.0)
            elif self.params['addpw'] == 1:  # Purely additive. This will almost certainly diverge, don't use it!
                hebb1 = hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb

            elif self.params['addpw'] == 0:
                # We do it the old way. Note that here, Hebb-rule is decaying.
                # There is probably a way to make it more efficient
                # NOTE: THIS WILL GO AWRY if DAout is allowed to go outside [0,1]!
                # Note 2: For Oja's rule, there is no difference between addpw 0 and addpw1
                hebb1 = (1 - DAout.view(BATCHSIZE, 1, 1)) * hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
            else:
                raise ValueError("Which additive form for plastic weights?")

            hebb = hebb1
            hidden = hactiv

            # normalise plastic weights after plastic update
            if self.params['wN'] is not None and 'plastic_weight_norm' in self.params['wN'] and \
                    self.params['wN']['plastic_weight_norm']['time'] == 'post':
                # using total effective weight norm
                if self.params['wN']['plastic_weight_norm']['norm'] == 'G':
                    if norm is not None:
                        hebb = hebb / (norm + 1e-15)
                    else:
                        norm = torch.linalg.norm(input=weight, ord=2, dim=2).unsqueeze(-1)
                        hebb = hebb / (norm + 1e-15)
                elif self.params['wN']['plastic_weight_norm']['norm'] == 'F':
                    norm = torch.linalg.norm(input=hebb, ord=2, dim=2).unsqueeze(-1)
                    hebb = hebb / (norm + 1e-15)
                else:
                    raise Exception(f"Invalid self.params['wN']['plastic_weight_norm']['norm'] "
                                    f"{self.params['wN']['plastic_weight_norm']['norm']}")

        elif self.type == 'modul':
            if self.eval_energy is True:
                raise NotImplementedError
            # The rows of w and hebb are the inputs weights to a single neuron
            # hidden = x, hactiv = y
            hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) +
                                torch.matmul((self.w + torch.mul(self.alpha, pw)), hidden.view(BATCHSIZE, HS, 1))
                                ).view(BATCHSIZE, HS)
            activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
            valueout = self.h2v(hactiv)

            # Now computing the Hebbian updates...

            # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA,
            # but we assume NBDA=1 for now in the deltahebb multiplication below)
            if self.params['da'] == 'tanh':
                DAout = F.tanh(self.h2DA(hactiv))  # noqa
            elif self.params['da'] == 'sig':
                DAout = F.sigmoid(self.h2DA(hactiv))  # noqa
            elif self.params['da'] == 'lin':
                DAout = self.h2DA(hactiv)  # noqa
            else:
                raise ValueError("Which transformation for DAout ?")

            # We need to select the order of operations; network update,
            # e.t. update, neuromodulated incorporation into plastic weights
            # One possibility (for now go with this one):
            #    - computing all outputs from current inputs, including DA
            #    - incorporating neuromodulated Hebb/eligibility trace into plastic weights
            #    - computing updated hebb/eligibility traces
            # Another possibility (modul2):
            #    - computing all outputs from current inputs, including DA
            #    - computing updated Hebb/eligibility traces
            #    - incorporating this modified Hebb into plastic weights through neuromodulation

            # In modul2 we would compute deltaet and update et here too; here we compute them later

            if self.params['addpw'] == 3:
                # Hard clamp
                # From modplast/addpw=3:
                # hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)
            elif self.params['addpw'] == 2:
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                # This constrains the pw to stay within [-1, 1] (we could also do that by putting a tanh on top of it,
                # but instead we want pw itself to remain within that range, to avoid large gradients and facilitate
                # movement back to 0)
                # The outer clamp is there for safety. In theory the expression within that clamp is "softly"
                # constrained to stay within [-1, 1], but finite-size effects might throw it off.
                pw1 = torch.clamp(
                    pw + torch.clamp(deltapw, min=0.0) * (1 - pw) + torch.clamp(deltapw, max=0.0) * (pw + 1),
                    min=-.99999, max=.99999)
            elif self.params['addpw'] == 1:  # Purely additive, tends to make the meta-learning diverge
                deltapw = DAout.view(BATCHSIZE, 1, 1) * et
                pw1 = pw + deltapw
            elif self.params['addpw'] == 0:
                # We do it the old way, with a decay term.
                # This will FAIL if DAout is allowed to go outside [0,1]
                # Note: this makes the plastic weights decaying!
                pw1 = (1 - DAout.view(BATCHSIZE, 1, 1)) * pw + DAout.view(BATCHSIZE, 1, 1) * et

            pw = pw1  # noqa: F823

            # Updating the eligibility trace - always a simple decay term.
            # batched outer product...should it be other way round?
            deltaet = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS))
            et = (1 - self.etaet) * et + self.etaet * deltaet

            hidden = hactiv

        else:
            raise ValueError("Must select network type")

        if self.eval_energy is False:
            energy = None
        return activout, valueout, hidden, hebb, et, pw, energy  # noqa: F823

    def initialZeroHebb(self):  # noqa
        if self.type == 'lstm':
            return None
        else:
            return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']),
                            requires_grad=False).to(self.params['device'])

    def initialZeroPlasticWeights(self):  # noqa
        if self.type == 'lstm':
            return None
        else:
            return Variable(torch.zeros(self.params['bs'], self.params['hs'], self.params['hs']),
                            requires_grad=False).to(self.params['device'])

    def initialZeroState(self):  # noqa
        BATCHSIZE = self.params['bs']  # noqa
        if self.type == 'lstm':
            return [
                Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False).to(self.params['device'])
                for _ in range(2)
            ]
        else:
            return Variable(torch.zeros(BATCHSIZE, self.params['hs']), requires_grad=False).to(self.params['device'])


class RandomPolicyNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x_t, hidden, hebb, et, pw):  # noqa
        """
        Dummy function. Random action won't be taken here.
        et and pw are legacy arguments from original code, are unused
        """
        valueout = torch.zeros(self.params['bs'])
        activout = torch.zeros(self.params['bs'], self.params['hs'])
        return activout, valueout, hidden, hebb, et, pw

    def initialZeroHebb(self):  # noqa
        return None

    def initialZeroPlasticWeights(self):  # noqa
        return None

    def initialZeroState(self):  # noqa
        return None


class STPNNetwork(nn.Module):

    def __init__(self, params):
        super().__init__()

        hidden_size = params['hs']
        input_size = TOTALNBINPUTS
        self.params = params

        self.h2o = torch.nn.Linear(hidden_size, NBACTIONS)  # From recurrent to outputs (action probabilities)
        self.h2v = torch.nn.Linear(hidden_size, 1)  # From recurrent to value-prediction (used for A2C)

        # store network architectural parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hsize, self.isize = hidden_size, input_size
        self.rnn = NetParallelEnergy(
            input_size, hidden_size, stp=params['stp'], rnn_type=params['type'],
            eval_energy=params['eval_energy'], **params['extra_config']
        )
        self.eval_energy = params['eval_energy']

    def init_weights(self):
        # should implement this for the softmax to actions and value
        raise NotImplementedError

    def forward(self, x_t, hidden, hebb, et, pw):  # noqa
        """
        Assumes x is of shape (batch, sequence, feature)
        et and pw are legacy arguments from original code, are unused
        """
        if self.eval_energy is True:
            # This implementation includes all input for energy calculation, including past reward.
            # Therefore, models with higher reward can have higher energy consumption for this input.
            h_tp1, states, energy = self.rnn(x=x_t, states=hebb)
        else:
            energy = None
            h_tp1, states = self.rnn(x=x_t, states=hebb)

        activout = self.h2o(h_tp1)  # Pure linear, raw scores - to be softmaxed later, outside the function
        valueout = self.h2v(h_tp1)

        return activout, valueout, h_tp1, states, et, pw, energy

    def initialZeroState(self):  # noqa
        return Variable(torch.zeros(self.params['bs'], self.hidden_size), requires_grad=False).to(self.params['device'])

    # In plastic networks, we must also initialize the Hebbian state:
    def initialZeroHebb(self):  # noqa
        return self.rnn.rnn.states_init(batch_size=self.params['bs'], device=self.params['device'])

    def initialZeroPlasticWeights(self):  # noqa
        return Variable(torch.zeros(
            self.params['bs'], self.params['hs'], self.params['hs']), requires_grad=False).to(self.params['device'])

import itertools
import math

import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import scipy.special as sps

from STPN.HebbFF.net_utils import StatefulBase, check_dims, random_weight_init, binary_classifier_accuracy


# %%############
### Hebbian ###
###############

class HebbNet(StatefulBase):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbNet, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            W, b = random_weight_init([Nx, Nh, Ny], bias=True)
        else:
            W, b = init
        #            check_dims(W,b)

        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.g1 = nn.Parameter(torch.tensor(float('nan')),
                               requires_grad=False)  # Can add this to network post-init. Then, freeze W1 and only train its gain
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut

        self.register_buffer('A', None)
        try:
            self.reset_state()
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbFF.__init__'.format(e))

        self.register_buffer('plastic', torch.tensor(True))
        self.register_buffer('forceAnti', torch.tensor(False))
        self.register_buffer('forceHebb', torch.tensor(False))
        self.register_buffer('reparamLam', torch.tensor(False))
        self.register_buffer('reparamLamTau', torch.tensor(False))
        self.register_buffer('groundTruthPlast', torch.tensor(False))

        self.init_hebb(**hebbArgs)  # parameters of Hebbian rule

    def load(self, filename):
        super(HebbNet, self).load(filename)
        self.update_hebb(torch.tensor([0.]),
                         torch.tensor([0.]))  # to get self.eta right if forceHebb/forceAnti used

    def reset_state(self):
        self.A = torch.zeros_like(self.w1)

    def init_hebb(self, eta=None, lam=0.99):
        if eta is None:
            eta = -5. / self.w1.shape[1]  # eta*d = -5

        """ A(t+1) = lam*A(t) + eta*h(t)x(t)^T """
        if self.reparamLam:
            self._lam = nn.Parameter(torch.tensor(np.log(lam / (1. - lam))))
            if self.lam:
                del self.lam
            self.lam = torch.sigmoid(self._lam)
        elif self.reparamLamTau:
            self._lam = nn.Parameter(torch.tensor(1. / (1 - lam)))
            if self.lam:
                del self.lam
            self.lam = 1. - 1 / self._lam
        else:
            self._lam = nn.Parameter(torch.tensor(lam))  # Hebbian decay
            self.lam = self._lam.data

        # Hebbian learning rate
        if self.forceAnti:
            if self.eta:
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta))))  # eta = exp(_eta)
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb:
            if self.eta:
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta))))  # eta = exp(_eta)
            self.eta = torch.exp(self._eta)
        else:
            self._eta = nn.Parameter(torch.tensor(eta))
            self.eta = self._eta.data

    def update_hebb(self, pre, post, isFam=False):
        if self.reparamLam:
            self.lam = torch.sigmoid(self._lam)
        elif self.reparamLamTau:
            self.lam = 1. - 1 / self._lam
        else:
            self._lam.data = torch.clamp(self._lam.data, min=0., max=1.)  # if lam>1, get exponential explosion
            self.lam = self._lam

        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb:
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta

        if self.plastic:
            if self.groundTruthPlast and isFam:
                self.A = self.lam * self.A
            else:
                self.A = self.lam * self.A + self.eta * torch.ger(post, pre)

    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""

        w1 = self.g1 * self.w1 if not torch.isnan(self.g1) else self.w1
        if len(x.shape) > 1:
            assert len(x.shape) == 2
            assert x.shape[0] == 1
            x = x.squeeze(0)
        a1 = torch.addmv(self.b1, w1 + self.A, x)  # hidden layer activation
        h = self.f(a1)
        self.update_hebb(x, h, isFam=isFam)

        if self.w2.numel() == 1:
            w2 = self.w2.expand(1, h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, h)  # output layer activation
        y = self.fOut(a2)

        if debug:
            return a1, h, a2, y
        return y

    def evaluate(self, batch):
        self.reset_state()
        out = torch.empty_like(batch[1])
        for t, (x, y) in enumerate(zip(*batch)):
            if len(x.shape) > 1:
                assert len(x.shape) == 2
                assert x.shape[0] == 1
                x = x.squeeze(0)
            out[t] = self(x, isFam=bool(y))
        return out

    @torch.no_grad()
    def evaluate_debug(self, batch, recogOnly=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        self.reset_state()

        Nh, d = self.w1.shape
        T = len(batch[1])
        db = {'a1': torch.empty(T, Nh),
              'h': torch.empty(T, Nh),
              'Wxb': torch.empty(T, Nh),
              'Ax': torch.empty(T, Nh),
              'a2': torch.empty_like(batch[1]),
              'out': torch.empty_like(batch[1])}
        for t, (x, y) in enumerate(zip(*batch)):
            db['Ax'][t] = torch.mv(self.A, x.squeeze(0))
            try:
                isFam = bool(y)
            except:
                isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x.squeeze(0), isFam=isFam, debug=True)
            w1 = self.g1 * self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, x.squeeze(0))
        db['acc'] = self.accuracy(batch).item()

        if recogOnly and len(db['out'].shape) > 1:
            db['data'] = TensorDataset(batch[0], batch[1][:, 0].unsqueeze(1))
            db['out'] = db['out'][:, 0].unsqueeze(1)
            db['a2'] = db['a2'][:, 0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()
        return db

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):
            if self.hist['iter'] % 10 == 0:
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta, self.hist['iter'])

                if self.w2.numel() == 1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel() == 1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel() == 1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class uSTPNfNet(StatefulBase):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs):
        super(uSTPNfNet, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            W, b = random_weight_init([Nx, Nh, Ny], bias=True)
        else:
            W, b = init
        #            check_dims(W,b)

        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        self.g1 = nn.Parameter(torch.tensor(float('nan')),
                               requires_grad=False)  # Can add this to network post-init. Then, freeze W1 and only train its gain
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut
        self.register_buffer('A', None)
        try:
            self.reset_state()
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbFF.__init__'.format(e))

        self.register_buffer('plastic', torch.tensor(True))
        self.register_buffer('forceAnti', torch.tensor(False))
        self.register_buffer('forceHebb', torch.tensor(False))
        self.register_buffer('reparamLam', torch.tensor(False))
        self.register_buffer('reparamLamTau', torch.tensor(False))
        self.register_buffer('groundTruthPlast', torch.tensor(False))

        self.init_hebb(**hebbArgs)  # parameters of Hebbian rule

    def load(self, filename):
        super(uSTPNfNet, self).load(filename)

    def reset_state(self):
        self.A = torch.zeros_like(self.w1)

    def init_hebb(self, eta=None, lam=None):
        self._lam = nn.Parameter(torch.rand(1))
        # rescale to [-0.001/h , 0.001/h]
        self._eta = nn.Parameter((torch.rand(1) - 0.5) * ((0.001 / math.sqrt(self.b1.shape[0])) / 0.5))
        self.lam = self._lam.data
        self.eta = self._eta.data

    def update_hebb(self, pre, post, isFam=False):
        self.lam = self._lam
        self.eta = self._eta

        if self.plastic:
            if self.groundTruthPlast and isFam:
                self.A = self.lam * self.A
            else:
                self.A = self.lam * self.A + self.eta * torch.ger(post, pre)

    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A).
        Don't call twice in a row unless you want to update self.A twice!"""
        if len(x.shape) > 1:
            assert len(x.shape) == 2
            assert x.shape[0] == 1
            x = x.squeeze(0)

        assert torch.isnan(self.g1)
        w1 = self.g1 * self.w1 if not torch.isnan(self.g1) else self.w1
        G1 = w1 + self.A
        norm = torch.linalg.norm(
            input=G1,
            ord=2,
            dim=1,  # (hidden, feature) , no batching here
        ).unsqueeze(-1)
        candidate_a1 = G1 @ x
        a1 = candidate_a1 / norm.squeeze(-1) + self.b1
        self.A = self.A / norm

        h = self.f(a1)
        self.update_hebb(x, h, isFam=isFam)

        if self.w2.numel() == 1:
            w2 = self.w2.expand(1, h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, h)  # output layer activation
        y = self.fOut(a2)

        if debug:
            return a1, h, a2, y
        return y

    def evaluate(self, batch):
        self.reset_state()
        out = torch.empty_like(batch[1])
        for t, (x, y) in enumerate(zip(*batch)):
            if len(x.shape) > 1:
                assert len(x.shape) == 2
                assert x.shape[0] == 1
                x = x.squeeze(0)
            out[t] = self(x, isFam=bool(y))
        return out

    @torch.no_grad()
    def evaluate_debug(self, batch, recogOnly=True):
        raise NotImplementedError
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the
        recognition part of the task, set recogOnly=True
        """
        self.reset_state()

        Nh, d = self.w1.shape
        T = len(batch[1])
        db = {'a1': torch.empty(T, Nh),
              'h': torch.empty(T, Nh),
              'Wxb': torch.empty(T, Nh),
              'Ax': torch.empty(T, Nh),
              'a2': torch.empty_like(batch[1]),
              'out': torch.empty_like(batch[1])}
        for t, (x, y) in enumerate(zip(*batch)):
            if len(x.shape) > 1:
                assert len(x.shape) == 2
                assert x.shape[0] == 1
                x = x.squeeze(0)
            db['Ax'][t] = torch.mv(self.A, x)
            try:
                isFam = bool(y)
            except:
                isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)
            w1 = self.g1 * self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, x)
        db['acc'] = self.accuracy(batch).item()

        if recogOnly and len(db['out'].shape) > 1:
            db['data'] = TensorDataset(batch[0], batch[1][:, 0].unsqueeze(1))
            db['out'] = db['out'][:, 0].unsqueeze(1)
            db['a2'] = db['a2'][:, 0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()
        return db

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        super(uSTPNfNet, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):
            if self.hist['iter'] % 10 == 0:
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta, self.hist['iter'])

                if self.w2.numel() == 1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel() == 1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel() == 1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class STPNfNet(uSTPNfNet):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs):
        super(STPNfNet, self).__init__(init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs)

    def init_hebb(self, eta=None, lam=None):
        self._lam = nn.Parameter(torch.rand(*self.w1.shape))
        # rescale to [-0.001/h , 0.001/h]
        assert self.b1.shape[0] == self.w1.shape[0]
        self._eta = nn.Parameter((torch.rand(*self.w1.shape) - 0.5) * ((0.001 / math.sqrt(self.b1.shape[0])) / 0.5))
        self.lam = self._lam.data
        self.eta = self._eta.data

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        # monitor grandparent directly, as otherwise it will monitor lambda and eta (gamma) as scalars
        super(uSTPNfNet, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):
            if self.hist['iter'] % 10 == 0:
                self.writer.add_scalar('params/lambda', self.lam.abs().mean(), self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta.abs().mean(), self.hist['iter'])

                if self.w2.numel() == 1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel() == 1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel() == 1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class uSTPNrNet(StatefulBase):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs):
        super(uSTPNrNet, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            W, b = random_weight_init([Nx + Nh, Nh, Ny], bias=True)
        else:
            W, b = init

        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        # Can add this to network post-init. Then, freeze W1 and only train its gain
        self.g1 = nn.Parameter(torch.tensor(float('nan')), requires_grad=False)
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut

        self.register_buffer('A', None)
        try:
            self.reset_state()
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbFF.__init__'.format(e))

        self.register_buffer('plastic', torch.tensor(True))
        self.register_buffer('forceAnti', torch.tensor(False))
        self.register_buffer('forceHebb', torch.tensor(False))
        self.register_buffer('reparamLam', torch.tensor(False))
        self.register_buffer('reparamLamTau', torch.tensor(False))
        self.register_buffer('groundTruthPlast', torch.tensor(False))

        self.init_hebb(**hebbArgs)  # parameters of Hebbian rule

    def load(self, filename):
        super(uSTPNrNet, self).load(filename)

    def reset_state(self):
        self.A = torch.zeros_like(self.w1, device=self.w1.device)
        self.h = torch.zeros_like(self.b1, device=self.w1.device)

    def init_hebb(self, eta=None, lam=None):
        self._lam = nn.Parameter(torch.rand(1))
        # rescale to [-0.001/h , 0.001/h]
        self._eta = nn.Parameter((torch.rand(1) - 0.5) * ((0.001 / self.b1.shape[0]) / 0.5))
        self.lam = self._lam.data
        self.eta = self._eta.data

    def update_hebb(self, pre, post, isFam=False):
        self.lam = self._lam
        self.eta = self._eta

        if self.plastic:
            if self.groundTruthPlast and isFam:
                self.A = self.lam * self.A
            else:
                self.A = self.lam * self.A + self.eta * torch.ger(post, pre)

    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A).
        Don't call twice in a row unless you want to update self.A twice!"""

        if len(x.shape) > 1:
            assert len(x.shape) == 2
            assert x.shape[0] == 1
            x = x.squeeze(0)
        assert torch.isnan(self.g1)
        w1 = self.g1 * self.w1 if not torch.isnan(self.g1) else self.w1
        G1 = w1 + self.A
        norm = torch.linalg.norm(
            input=G1,
            ord=2,
            dim=1,  # (hidden, features) , no batching
        ).unsqueeze(-1)
        pre = torch.cat([x, self.h], dim=0)
        candidate_a1 = G1 @ pre
        a1 = candidate_a1 / norm.squeeze(-1) + self.b1
        self.A = self.A / norm

        self.h = self.f(a1)
        self.update_hebb(pre, self.h, isFam=isFam)

        if self.w2.numel() == 1:
            w2 = self.w2.expand(1, self.h.shape[0])
        else:
            w2 = self.w2
        a2 = torch.addmv(self.b2, w2, self.h)  # output layer activation
        y = self.fOut(a2)

        if debug:
            return a1, self.h, a2, y
        return y

    def evaluate(self, batch):
        self.reset_state()
        out = torch.empty_like(batch[1])
        for t, (x, y) in enumerate(zip(*batch)):
            if len(x.shape) > 1:
                assert len(x.shape) == 2
                assert x.shape[0] == 1
                x = x.squeeze(0)
            out[t] = self(x, isFam=bool(y))
        return out

    @torch.no_grad()
    def evaluate_debug(self, batch, recogOnly=True):
        raise NotImplementedError
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the
        recognition part of the task, set recogOnly=True
        """
        self.reset_state()

        Nh, d = self.w1.shape
        T = len(batch[1])
        db = {'a1': torch.empty(T, Nh),
              'h': torch.empty(T, Nh),
              'Wxb': torch.empty(T, Nh),
              'Ax': torch.empty(T, Nh),
              'a2': torch.empty_like(batch[1]),
              'out': torch.empty_like(batch[1])}
        for t, (x, y) in enumerate(zip(*batch)):
            if len(x.shape) > 1:
                assert len(x.shape) == 2
                assert x.shape[0] == 1
                x = x.squeeze(0)
            db['Ax'][t] = torch.mv(self.A, x)
            try:
                isFam = bool(y)
            except:
                isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)
            w1 = self.g1 * self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, x)
        db['acc'] = self.accuracy(batch).item()

        if recogOnly and len(db['out'].shape) > 1:
            db['data'] = TensorDataset(batch[0], batch[1][:, 0].unsqueeze(1))
            db['out'] = db['out'][:, 0].unsqueeze(1)
            db['a2'] = db['a2'][:, 0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()
        return db

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        # raise NotImplementedError
        super(uSTPNrNet, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):
            if self.hist['iter'] % 10 == 0:
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta, self.hist['iter'])

                if self.w2.numel() == 1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel() == 1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel() == 1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class STPNrNet(uSTPNrNet):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs):
        super(STPNrNet, self).__init__(init, f=torch.tanh, fOut=torch.sigmoid, **hebbArgs)

    def init_hebb(self, eta=None, lam=None):
        self._lam = nn.Parameter(torch.rand(*self.w1.shape))
        # rescale to [-0.001/h , 0.001/h]
        assert self.b1.shape[0] == self.w1.shape[0]
        self._eta = nn.Parameter((torch.rand(*self.w1.shape) - 0.5) * ((0.001 / math.sqrt(self.b1.shape[0])) / 0.5))
        self.lam = self._lam.data
        self.eta = self._eta.data

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None):
        # monitor grandparent directly, as otherwise it will monitor lambda and eta (gamma) as scalars
        super(uSTPNrNet, self)._monitor(trainBatch, validBatch, out, loss, acc)

        if hasattr(self, 'writer'):
            if self.hist['iter'] % 10 == 0:
                self.writer.add_scalar('params/lambda', self.lam.abs().mean(), self.hist['iter'])
                self.writer.add_scalar('params/eta', self.eta.abs().mean(), self.hist['iter'])

                if self.w2.numel() == 1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel() == 1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel() == 1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class HebbFeatureLayer(HebbNet):
    def __init__(self, init, Nx, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbFeatureLayer, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        _, d = self.w1.shape
        self.featurizer = nn.Linear(Nx, d)

    def forward(self, x, isFam=False, debug=False):
        xFeat = self.featurizer(x)
        return super(HebbFeatureLayer, self).forward(xFeat, isFam, debug)

    @torch.no_grad()
    def evaluate_debug(self, batch, recogOnly=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        Nh, d = self.w1.shape
        T = len(batch[1])
        db = {'a1': torch.empty(T, Nh),
              'h': torch.empty(T, Nh),
              'Wxb': torch.empty(T, Nh),
              'Ax': torch.empty(T, Nh),
              'a2': torch.empty_like(batch[1]),
              'out': torch.empty_like(batch[1])}
        for t, (x, y) in enumerate(zip(*batch)):
            db['Ax'][t] = torch.mv(self.A, self.featurizer(x))
            try:
                isFam = bool(y)
            except:
                isFam = bool(y[0])
            db['a1'][t], db['h'][t], db['a2'][t], db['out'][t] = self(x, isFam=isFam, debug=True)
            w1 = self.g1 * self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1
            db['Wxb'][t] = torch.addmv(self.b1, w1, self.featurizer(x))
        db['acc'] = self.accuracy(batch).item()

        if recogOnly and len(db['out'].shape) > 1:
            db['data'] = TensorDataset(batch[0], batch[1][:, 0].unsqueeze(1))
            db['out'] = db['out'][:, 0].unsqueeze(1)
            db['a2'] = db['a2'][:, 0].unsqueeze(1)
            db['acc'] = self.accuracy(batch).item()
        return db


class HebbClassify(HebbNet):
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        super(HebbClassify, self).__init__(init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs)
        self.onlyRecogAcc = False
        self.w2c = nn.Parameter(torch.tensor(float('nan')), requires_grad=False)  # add this to network post-init

    def forward(self, x, isFam=False, debug=False):
        """This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""

        w1 = self.g1 * self.w1 if not torch.isnan(self.g1) else self.w1
        a1 = torch.addmv(self.b1, w1 + self.A, x)  # hidden layer activation
        h = self.f(a1)
        self.update_hebb(x, h, isFam=isFam)

        if self.w2.numel() == 1:
            w2 = self.w2.expand(1, h.shape[0])
        else:
            w2 = self.w2
        if not torch.isnan(self.w2c).any():
            w2 = torch.cat((w2, self.w2c.data))
        a2 = torch.addmv(self.b2, w2, h)  # output layer activation
        y = self.fOut(a2)

        if debug:
            return a1, h, a2, y
        return y

    def evaluate(self, batch):
        return super(HebbNet, self).evaluate(batch)

    def average_loss(self, batch, out=None):
        if out is None:
            out = self.evaluate(batch)

        recogLoss = self.loss_fn(out[:, 0], batch[1][:, 0])
        classLoss = self.loss_fn(out[:, 1], batch[1][:, 1])
        loss = recogLoss + classLoss

        currentIter = self.hist['iter'] + 1  # +1 b/c this gets called before hist['iter'] gets incremented
        if currentIter % 10 == 0 and self.training:
            self.hist['recog_loss'].append(recogLoss.item())
            self.hist['class_loss'].append(classLoss.item())
            if hasattr(self, 'writer'):
                self.writer.add_scalars(
                    'train/loss_breakdown', {'total': loss, 'recog': recogLoss, 'class': classLoss}, currentIter)
            print('     {} recog_loss:{:.3f} class_loss:{:.3f}'.format(currentIter, recogLoss, classLoss))
        return loss

    def accuracy(self, batch, out=None):
        if out is None:
            out = self.evaluate(batch)

        recogAcc = self.acc_fn(out[:, 0], batch[1][:, 0])
        classAcc = self.acc_fn(out[:, 1], batch[1][:, 1])
        if self.onlyRecogAcc:
            acc = recogAcc
        else:
            acc = (recogAcc + classAcc) / 2.

        if self.training and self.hist['iter'] % 10 == 0:
            self.hist['recog_acc'].append(recogAcc.item())
            self.hist['class_acc'].append(classAcc.item())
            if hasattr(self, 'writer'):
                self.writer.add_scalars('train/acc_breakdown', {'average': acc,
                                                                'recog': recogAcc,
                                                                'class': classAcc},
                                        self.hist['iter'])
            print('     {} recog_acc:{:.3f} class_acc:{:.3f}'.format(self.hist['iter'], recogAcc, classAcc))
        return acc

    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None):
        if self.hist is None:
            self.hist = {'epoch': 0,
                         'iter': -1,  # gets incremented when _monitor() is called
                         'train_loss': [],
                         'train_acc': [],
                         'grad_norm': [],
                         'recog_loss': [],
                         'class_loss': [],
                         'recog_acc': [],
                         'class_acc': []}
            if validBatch:
                self.hist['valid_loss'] = []
                self.hist['valid_acc'] = []
            self._monitor(trainBatch, validBatch=validBatch)
        else:
            print('Network already partially trained. Continuing from iter {}'.format(self.hist['iter']))


class DetHebb():
    def __init__(self, D, n, f=0.5, Pfp=0.01, Ptp=0.99):
        '''
        D: plastic input dim (total input dim is d=D+n)
        n: log_2(hidden dim)
        f: fraction of novel stimuli
        Pfp: desired probability of false positive
        Ptp: desired probability of true positive
        '''
        self.D = D  # plastic input dim
        self.n = n
        self.d = self.D + self.n
        self.N = 2 ** n  # hidden dim

        self.Pfp = Pfp  # true and false positive probabilities determine decay rate and bias
        self.Ptp = Ptp
        self.f = f
        self.a = (sps.erfcinv(2 * Pfp) - sps.erfcinv(2 * Ptp)) * np.sqrt(2 * np.e)
        self.gam = 1 - (np.square(self.a) * f) / (2 * self.D * self.N)  # decay rate

        self.W = D * np.array(list(itertools.product([-1, 1], repeat=n)))  # static, shape Nxn
        self.reset_A()  # plastic
        self.b = sps.erfcinv(2 * Pfp) * np.sqrt(2) / self.a - n
        self.B = D * self.b  # bias, such that exactly one unit active for novel

    def reset_A(self):
        self.A = np.zeros((self.N, self.D))  # plastic

    def forward(self, x, debug=False):
        xW = x[:self.n]  # length n
        xA = x[self.n:]  # length D
        Ax = self.A.dot(xA)
        Wxb = self.W.dot(xW) + self.B
        a = Wxb + Ax  # pre-activation
        h = np.heaviside(a, 0)  # h=1 if a>0, h=0 if a<=0
        yHat = np.all(h == 0)

        self.A = self.gam * self.A - np.outer(h, xA)

        if debug:
            return a, h, Ax, Wxb, yHat
        return yHat

    def evaluate(self, X):
        Yhat = np.zeros((X.shape[0], 1))
        for t, x in enumerate(X):
            Yhat[t] = self.forward(x)
        return Yhat

    def evaluate_debug(self, batch):
        T = len(batch[1])
        db = {'a1': np.empty((T, self.N)),
              'h': np.empty((T, self.N)),
              'Wxb': np.empty((T, self.N)),
              'Ax': np.empty((T, self.N)),
              'a2': np.empty_like((batch[1])),
              'out': np.empty_like((batch[1]))}
        for t, (x, y) in enumerate(zip(*batch)):
            db['a1'][t], db['h'][t], db['Ax'][t], db['Wxb'][t], db['a2'][t] = self.forward(x, debug=True)
            db['out'][t] = db['a2'][t]
        db['acc'] = self.accuracy(db['out'], Yhat=batch[1].numpy())
        db['data'] = TensorDataset(batch[0], batch[1])
        return db

    def accuracy(self, Y, Yhat=None, X=None):
        # alternatively, acc = f*(1-Pfp)+(1-f)*Ptp
        if Yhat is None:
            assert X is not None
            Yhat = self.evaluate(X)
        acc = float((Y == Yhat).sum()) / len(Y)
        return acc

    def true_false_pos(self, Y, Yhat=None, X=None):
        if Yhat is None:
            assert X is not None
            Yhat = self.evaluate(X)
        posOutIdx = Yhat == 1

        totPos = Y.sum()
        totNeg = len(Y) - totPos

        falsePos = (1 - Y)[posOutIdx].sum()
        truePos = Y[posOutIdx].sum()

        falsePosRate = falsePos / totNeg
        truePosRate = truePos / totPos
        return truePosRate, falsePosRate

    def true_false_pos_analytic(self, R, corrected=False):
        if not corrected:
            Pfp = 0.5 * sps.erfc(self.a * (self.n + self.b) / np.sqrt(2)) * np.ones(len(R))
            Ptp = 0.5 * sps.erfc(self.a * (self.n + self.b - np.power(self.gam, R - 1)) / np.sqrt(2))
        else:
            PfpOld = 0.5 * sps.erfc(self.a * (self.n + self.b) / np.sqrt(2)) * np.ones(len(R))
            PtpOld = 0.5 * sps.erfc(self.a * (self.n + self.b - np.power(self.gam, R - 1)) / np.sqrt(2))
            while True:
                fEff = (1 - PfpOld) * self.f + (1 - PtpOld) * (1 - self.f)  # fraction of items *reported* as novel
                a = self.a * np.sqrt(self.f / fEff)
                Pfp = 0.5 * sps.erfc(a * (self.n + self.b) / np.sqrt(2)) * np.ones(len(R))
                Ptp = 0.5 * sps.erfc(a * (self.n + self.b - np.power(self.gam, R - 1)) / np.sqrt(2))

                if np.all(np.abs(PfpOld - Pfp) < 0.0001) and np.all(np.abs(PtpOld - Ptp) < 0.0001):
                    break
                PfpOld, PtpOld = Pfp, Ptp
                print('{} {}'.format(np.max(np.abs(PfpOld - Pfp)), np.max(np.abs(PtpOld - Ptp))))
        return Ptp, Pfp


class DetHebbNoSplit(DetHebb):
    def __init__(self, d, n, f=0.5, Pfp=0.01, Ptp=0.99):
        '''
        d:  input dim
        n: log_2(hidden dim)
        f: fraction of novel stimuli
        Pfp: desired probability of false positive
        Ptp: desired probability of true positive
        '''
        self.d = d  # plastic input dim
        self.n = n
        self.N = 2 ** n  # hidden dim

        self.Pfp = Pfp  # true and false positive probabilities determine decay rate and bias
        self.Ptp = Ptp
        self.f = f
        self.a = (sps.erfcinv(2 * Pfp) - sps.erfcinv(2 * Ptp)) * np.sqrt(2 * np.e)
        self.gam = 1 - (np.square(self.a) * f) / (2 * self.d * self.N)  # decay rate

        self.W = d * np.array(list(itertools.product([-1, 1], repeat=n)))  # static, shape Nxn
        self.reset_A()  # plastic
        self.b = sps.erfcinv(2 * Pfp) * np.sqrt(2) / self.a - n
        self.B = d * self.b  # bias, such that exactly one unit active for novel

    def reset_A(self):
        self.A = np.zeros((self.N, self.d))  # plastic

    def forward(self, x, debug=False):
        xW = x[:self.n]  # length n
        Ax = self.A.dot(x)
        Wxb = self.W.dot(xW) + self.B
        a = Wxb + Ax  # pre-activation
        h = np.heaviside(a, 0)  # h=1 if a>0, h=0 if a<=0
        yHat = np.all(h == 0)

        self.A = self.gam * self.A - np.outer(h, x)

        if debug:
            return a, h, Ax, Wxb, yHat
        return yHat


class HebbNetBatched(StatefulBase):
    def __init__(self, init, batchSize=1, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):
        """
        NOTE: self.w2 is stored transposed so that I don't have to transpose it every time in batched version of forward()
        """
        super(HebbNetBatched, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            W, b = random_weight_init([Nx, Nh, Ny], bias=True)
        else:
            W, b = init
            check_dims(W, b)

        self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))  # shape=[Nh,Nx]
        self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float).unsqueeze(1))  # shape=[Nh,1] for broadcasting
        self.w2 = nn.Parameter(
            torch.tensor(W[1], dtype=torch.float).t())  # shape=[Nh,Ny] pre-transposed for faster matmul
        self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))  # shape=[Ny]

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy

        self.f = f
        self.fOut = fOut

        self.init_hebb(**hebbArgs)  # parameters of Hebbian rule

        self.register_buffer('A', None)
        self.reset_state(batchSize=batchSize)

    def reset_state(self, batchSize=None):
        if batchSize is None:
            batchSize, _, _ = self.A.shape
        self.A = torch.zeros(batchSize, *self.w1.shape, device=self.w1.device)  # shape=[B,Nh,Nx]

    def init_hebb(self, eta=None, lam=0.99):
        if eta is None:
            eta = -5. / self.w1.shape[1]  # eta*d = -5
        self.lam = nn.Parameter(torch.tensor(lam))  # Hebbian decay
        self.eta = nn.Parameter(torch.tensor(eta))  # Hebbian learning rate

    def update_hebb(self, pre, post):
        """Updates A using a (batched) outer product, i.e. torch.ger(post, pre)
        for each of the elements in the batch
            
        pre.shape = [B,Nx] (pre.unsq.shape=[B,1,Nx])
        post.shape = [B,Nh,1]
        """
        self.lam.data = torch.clamp(self.lam.data, max=1.)
        self.A = self.lam * self.A + self.eta * torch.bmm(post, pre.unsqueeze(1))  # shape=[B,Nh,Nx]

    def forward(self, x, debug=False):
        """
        x.shape = [B,Nx]
        
        NOTE: This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!"""
        # b1.shape=[Nh,1], w1.shape=[Nh,Nx], A.shape=[B,Nh,Nx], x.unsq.shape=[B,Nx,1]
        a1 = torch.baddbmm(self.b1, self.w1 + self.A, x.unsqueeze(2))  # shape=[B,Nh,1] (broadcast)
        h = self.f(a1)  # hidden layer activation
        self.update_hebb(x, h)

        # b2.shape=[Ny], h.sq.shape=[B,Nh] w2.shape=[Nh,Ny]
        a2 = torch.addmm(self.b2, h.squeeze(dim=2), self.w2)  # shape=[B,Ny]
        y = self.fOut(a2)  # output layer activation

        if debug:
            return a1, h, a2, y
        return y


# %%##############
### Recurrent ###
#################

class VanillaRNN(StatefulBase):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid):
        super(VanillaRNN, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            W, b = random_weight_init([Nx, Nh, Nh, Ny], bias=True)
        else:
            W, b = init

        self.Wx = nn.Parameter(torch.tensor(W[0], dtype=torch.float))  # input weights
        self.Wh = nn.Parameter(torch.tensor(W[1], dtype=torch.float))  # recurrent weights
        self.b = nn.Parameter(torch.tensor(b[1], dtype=torch.float))  # recurrent neuron bias

        self.Wy = nn.Parameter(torch.tensor(W[2], dtype=torch.float))  # output weights
        self.bY = nn.Parameter(torch.tensor(b[2], dtype=torch.float))  # output neuron bias

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy
        self.f = f
        self.fOut = fOut

        self.reset_state()

    def reset_state(self):
        self.h = torch.zeros_like(self.b)

    def forward(self, x):
        # TODO: concat into single matrix for faster matmul
        a1 = torch.addmv(torch.addmv(self.b, self.Wx, x), self.Wh, self.h)  # W*x(t) + W*h(t-1) + b
        self.h = self.f(a1)
        a2 = torch.addmv(self.bY, self.Wy, self.h)
        y = self.fOut(a2)
        return y


class LSTM(VanillaRNN):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid):
        super(VanillaRNN, self).__init__()

        if all([type(x) == int for x in init]):
            Nx, Nh, Ny = init
            Wi, bi = random_weight_init([Nx, Nh, Nh], bias=True)
            Wf, bf = random_weight_init([Nx, Nh, Nh], bias=True)
            Wo, bo = random_weight_init([Nx, Nh, Nh], bias=True)
            Wc, bc = random_weight_init([Nx, Nh, Nh], bias=True)
            Wy, by = random_weight_init([Nh, Ny], bias=True)
        else:
            W, b = init

        self.Wix = nn.Parameter(torch.tensor(Wi[0], dtype=torch.float))
        self.Wih = nn.Parameter(torch.tensor(Wi[1], dtype=torch.float))
        self.bi = nn.Parameter(torch.tensor(bi[1], dtype=torch.float))

        self.Wfx = nn.Parameter(torch.tensor(Wf[0], dtype=torch.float))
        self.Wfh = nn.Parameter(torch.tensor(Wf[1], dtype=torch.float))
        self.bf = nn.Parameter(torch.tensor(bf[1], dtype=torch.float))

        self.Wox = nn.Parameter(torch.tensor(Wo[0], dtype=torch.float))
        self.Woh = nn.Parameter(torch.tensor(Wo[1], dtype=torch.float))
        self.bo = nn.Parameter(torch.tensor(bo[1], dtype=torch.float))

        self.Wcx = nn.Parameter(torch.tensor(Wc[0], dtype=torch.float))
        self.Wch = nn.Parameter(torch.tensor(Wc[1], dtype=torch.float))
        self.bc = nn.Parameter(torch.tensor(bc[1], dtype=torch.float))

        self.Wy = nn.Parameter(torch.tensor(Wy[0], dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0], dtype=torch.float))

        self.f = f
        self.fOut = fOut

        self.reset_state()

    def reset_state(self):
        self.h = torch.zeros_like(self.bc)
        self.c = torch.zeros_like(self.bc)

    def forward(self, x):
        # TODO: concat into single matrix for faster matmul
        ig = torch.sigmoid(torch.addmv(torch.addmv(self.bi, self.Wih, self.h), self.Wix, x))  # input gate
        fg = torch.sigmoid(torch.addmv(torch.addmv(self.bf, self.Wfh, self.h), self.Wfx, x))  # forget gate
        og = torch.sigmoid(torch.addmv(torch.addmv(self.bo, self.Woh, self.h), self.Wox, x))  # output gate
        cIn = self.f(torch.addmv(torch.addmv(self.bc, self.Wch, self.h), self.Wcx, x))  # cell input
        self.c = fg * self.c + ig * cIn  # cell state
        self.h = og * torch.tanh(self.c)  # hidden layer activation i.e. cell output

        y = self.fOut(torch.addmv(self.by, self.Wy, self.h))
        return y


class nnLSTM(VanillaRNN):
    """Should be identical to implementation above, but uses PyTorch internals for LSTM layer instead"""

    def __init__(self, init, f=None, fOut=torch.sigmoid):  # f is ignored. Included to have same signature as VanillaRNN
        super(VanillaRNN, self).__init__()

        Nx, Nh, Ny = init
        self.lstm = nn.LSTMCell(Nx, Nh)

        Wy, by = random_weight_init([Nh, Ny], bias=True)
        self.Wy = nn.Parameter(torch.tensor(Wy[0], dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0], dtype=torch.float))

        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy
        self.fOut = fOut

        self.reset_state()

    def reset_state(self):
        self.h = torch.zeros(1, self.lstm.hidden_size, device=self.lstm.weight_hh.device)
        self.c = torch.zeros(1, self.lstm.hidden_size, device=self.lstm.weight_hh.device)

    def forward(self, x):
        self.h, self.c = self.lstm(x.unsqueeze(0), (self.h, self.c))
        y = self.fOut(F.linear(self.h, self.Wy, self.by))
        return y

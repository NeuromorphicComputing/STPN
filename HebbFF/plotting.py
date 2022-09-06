"""
Plotting utils
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import erfc, erfcinv
import joblib
import torch
from torch import nn

import STPN
from STPN.HebbFF.net_utils import load_from_file
from STPN.HebbFF.data import generate_recog_data, recog_chance

try:
    from networks import nnLSTM
except:
    from STPN.HebbFF.networks import nnLSTM


def plot_loss_acc(net, chance=None):
    if type(net) == str:
        net = load_from_file(net)
    monitor_interval = 10
    iters = np.arange(len(net.hist['train_loss'])) * monitor_interval

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(iters, net.hist['train_loss'], label='Training')
    ax[0].plot(iters, net.hist['valid_loss'], label='Validation')
    ax[0].set_ylabel('Loss')

    ax[1].plot(iters, net.hist['train_acc'], label='Training')
    ax[1].plot(iters, net.hist['valid_acc'], label='Validation')
    if chance is not None:
        if np.isscalar(chance):
            chance = chance * np.ones_like(iters)
        ax[1].plot(iters, chance, 'k:', linewidth=0.75)
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iteration')
    ax[1].legend()

    return ax


def plot_generalization(testR, testAcc, truePosRate, falsePosRate, chance=None, label='', xscale='log', ax=None,
                        **kwargs):
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[1].plot([], 'k-', label='True positive')
        ax[1].plot([], 'k--', label='False positive')
        ax[1].legend()

    ax[0].plot(testR, testAcc, label=label, **kwargs)
    if chance is not None:
        if np.isscalar(chance):
            chance = chance * np.ones_like(testR)
        ax[0].plot(testR, chance, 'k:', linewidth=0.75)
    ax[0].set_xscale(xscale)
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    line = ax[1].plot(testR, truePosRate, **kwargs)
    color = kwargs.pop('color', line[0].get_color())
    ls = kwargs.pop('ls', '--')
    ax[1].plot(testR, falsePosRate, ls=ls, color=color, **kwargs)
    ax[1].set_ylabel('Probability')
    ax[1].set_xlabel('$R_{test}$')

    return ax


def plot_multiseed_multimodel_generalization(
        gen, chance=None,
        labels=None,  # label='',
        xscale='log', ax=None, colors=None, alpha=0.1, **kwargs):
    # gen is a dict with model_name: (testR, testAcc, truePosRate, falsePosRate) where these are for all seeds
    if ax is None:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[1].plot([], 'k-', label='True positive')
        ax[1].plot([], 'k--', label='False positive')
        ax[1].legend()

    for model_name, this_gen in gen.items():
        (testR, testAcc, truePosRate, falsePosRate) = this_gen
        testAcc_line, testAcc_ci = np.mean(testAcc, axis=0), np.std(testAcc, axis=0)
        testAcc_ci_top, testAcc_ci_bot = testAcc_line + testAcc_ci, testAcc_line - testAcc_ci
        ax[0].plot(testR, testAcc_line, label=labels[model_name], **kwargs, color=colors[model_name])
        ax[0].fill_between(testR, testAcc_ci_bot, testAcc_ci_top, color=colors[model_name], alpha=alpha)
        if chance is not None:
            if np.isscalar(chance):
                chance = chance * np.ones_like(testR)
            ax[0].plot(testR, chance, 'k:', linewidth=0.75)
        ax[0].set_xscale(xscale)
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        truePosRate_line, truePosRate_ci = np.mean(truePosRate, axis=0), np.std(truePosRate, axis=0)
        truePosRate_ci_top, truePosRate_ci_bot = truePosRate_line + truePosRate_ci, truePosRate_line - truePosRate_ci
        falsePosRate_line, falsePosRate_ci = np.mean(falsePosRate, axis=0), np.std(falsePosRate, axis=0)
        falsePosRate_ci_top = falsePosRate_line + falsePosRate_ci
        falsePosRate_ci_bot = falsePosRate_line - falsePosRate_ci

        line = ax[1].plot(testR, truePosRate_line, color=colors[model_name], label=labels[model_name], **kwargs)
        ax[1].fill_between(testR, truePosRate_ci_bot, truePosRate_ci_top, color=colors[model_name], alpha=alpha)
        color = kwargs.pop('color', line[0].get_color())
        ls = kwargs.pop('ls', '--')
        ax[1].plot(testR, falsePosRate_line, ls=ls, color=colors[model_name], label=labels[model_name], **kwargs)
        ax[1].fill_between(testR, falsePosRate_ci_bot, falsePosRate_ci_top, color=colors[model_name], alpha=alpha)

        ax[1].set_ylabel('Probability')
        ax[1].set_xlabel('$R_{test}$')

    return ax


def plot_acc_vs_T(Tlist, Rlist, acc, tp, fp, chance=None):
    fig, ax = plt.subplots(2, 1, sharex=True)

    lines = ax[0].semilogx(Tlist, acc.T)
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(lines, ['$R_{{test}}={}$'.format(R) for R in Rlist])
    if chance is not None:
        if np.isscalar(chance):
            chance = chance * np.ones_like(Tlist)
        ax[0].plot(Tlist, chance, 'k:', linewidth=0.75)

    linesTP = ax[1].semilogx(Tlist, tp.T)
    linesFP = ax[1].semilogx(Tlist, fp.T, '--')
    for i, line in enumerate(linesTP):
        linesFP[i].set_color(line.get_color())
    ax[1].set_xlabel('$T$ (dataset length)')
    ax[1].set_ylabel('Probability')
    ax[1].plot([], 'k-', label='True positive')
    ax[1].plot([], 'k--', label='False positive')
    ax[1].legend()

    return ax


def plot_and_fit(X, Y, ax=None, color=None, marker='.', linestyle='--', label='', xlabel='', title=''):
    if ax is None:
        _, ax = plt.subplots()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']

    X = X[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]
    if len(Y) <= 1:
        return ax

    if marker is not None:
        ax.loglog(X, Y, color=color, linestyle='', marker=marker)
    k, c, _, _, _ = stats.linregress(np.log(X), np.log(Y))
    x = np.linspace(min(X), max(X))
    ax.loglog(x, np.exp(c) * np.power(x, k), linestyle=linestyle, color=color,
              label=label + ' (k={:.2f}, c={:.2f})'.format(k, c))

    ax.set_xlabel(xlabel)
    ax.set_ylabel('$R_{max}$')
    ax.set_title(title)
    ax.legend()
    ax.get_figure().tight_layout()
    return ax


def plot_hidden_activity(h, isFam, out, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    T = h.shape[0]
    # insert col of nan's in between each col to force whitespace
    h = interleave(h, torch.full(h.shape, float('nan'))).T
    im = ax.matshow(h, cmap='Reds', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    for i in range(2 * T):
        if i % 2 == 0 and isFam[i // 2]:
            ax.axvspan(i - 0.5, i + 0.5, ec='k', fill=False, linewidth=0.75)

    ax.set_xticks(range(2 * T))
    isError = (isFam != out.round())
    xticklabels = ['*' if isError[t] else '' for t in range(T)]
    xticklabels = [xticklabels[i // 2] if i % 2 == 0 else '' for i in range(2 * len(xticklabels))]
    ax.set_xticklabels(xticklabels)
    ax.xaxis.tick_bottom()

    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron')
    ax.set_frame_on(False)
    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        left=False,
        bottom=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelright=False)
    return ax


def plot_corr_per_timepoint(Wxb, Ax, isFam, out, cov=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    T = len(Wxb)

    co = np.zeros(Wxb.shape[0])
    for t in range(T):
        if cov:
            co[t] = np.cov(Wxb[t], Ax[t])[0, 1]
        else:
            co[t] = np.corrcoef(Wxb[t], Ax[t])[0, 1]

    ax.plot(co, marker='.')
    ax.plot(torch.nonzero(isFam)[:, 0], co[isFam.flatten()], marker='o', markerfacecolor=(1, 1, 1, 0),
            markeredgecolor='k', ls='')
    ax.set_ylabel('Covariance' if cov else 'Correlation')

    isError = (isFam != out.round())
    xticklabels = ['*' if isError[t] else '' for t in range(T)]
    ax.set_xticks(range(T))
    ax.set_xticklabels(xticklabels)
    ax.xaxis.tick_bottom()

    ax.set_xlabel('Time')

    return ax


def plot_weight(W, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    v = W.abs().max()
    im = ax.matshow(W, cmap='RdBu_r', vmin=-v, vmax=v)
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_frame_on(False)
    ax.tick_params(
        axis='both',
        which='both',
        top=False,
        left=False,
        bottom=False,
        right=False,
        labeltop=False,
        labelleft=False,
        labelbottom=False,
        labelright=False)

    return ax


def plot_distribution(data, isFam, mean=False, xlabel='x', ax=None, bins=50):
    try:
        data = data.detach()
    except:
        pass
    if ax is None:
        fig, ax = plt.subplots()

    values, bins, patches = ax.hist(data[isFam == 0], bins=bins, density=True, histtype='step', align='mid',
                                    color='red')
    ax.plot([], color=patches[0].get_edgecolor(), label='p($\cdot$|y=0)')
    if np.abs(bins - bins.mean()).sum() < 0.01:
        ax.set_xlim(bins.mean() - 0.01, bins.mean() + 0.01)

    values, bins, patches = ax.hist(data[isFam == 1], bins=bins, density=True, histtype='step', align='mid',
                                    color='green')
    ax.plot([], color=patches[0].get_edgecolor(), label='p($\cdot$|y=1)')
    if np.abs(bins - bins.mean()).sum() < 0.01:
        ax.set_xlim(bins.mean() - 0.01, bins.mean() + 0.01)

    if mean:
        ax.axvline(data[isFam == 0].mean(), linestyle='--', linewidth=0.5, color='red', )
        ax.axvline(data[isFam == 1].mean(), linestyle='--', linewidth=0.5, color='green')

    ax.set_xlabel(xlabel)
    return ax


def plot_R_curriculum(iters, Rs, label='', ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.step(iters, Rs, label=label, where='post', **kwargs)
    ax.set_xlabel('Iter')
    ax.set_ylabel('R')
    return ax


def plot_familiar_output_distr(fname, gen_data, net=None, Rs=None, plotNov=False):
    if Rs is None:
        Rs = np.unique(np.logspace(0, 2, 4, dtype=int))

    _, ax = plt.subplots()
    for i, R in enumerate(Rs):
        print('Plot distr R={}'.format(R))
        res = get_evaluation_result(fname, gen_data(R), R=R, net=net)
        isFam = res['data'].tensors[1].bool()
        values, bins, patches = ax.hist(res['a2'][isFam].numpy(), bins=50, density=True, histtype='step', align='mid')
        ax.plot([], color=patches[0].get_edgecolor(), label='R={}'.format(R))

    if plotNov:
        values, bins, patches = ax.hist(res['a2'][~isFam].numpy(), bins=50, density=True, histtype='step', align='mid')
        for p in patches:
            p.set_linestyle(':')
            p.set_color('k')
        ax.plot([], color=patches[0].get_edgecolor(), linestyle=patches[0].get_linestyle(), label='Novel')

        ax.axvline(0, linestyle='-', linewidth=1, color='black')

    ax.set_xlabel('$W_2h(t)+b_2$')
    ax.set_ylabel('Probability')
    ax.legend()
    return ax


def plot_RT_curves(fname, gen_data, Rs=None, net=None):
    if Rs is None:
        Rs = np.unique(np.logspace(0, 2, 10, dtype=int))

    pFP = np.empty(len(Rs) + 1)  # p(nov|fam,R)
    for i, R in enumerate(Rs):
        print('Plot RT R={}'.format(R))
        res = get_evaluation_result(fname, gen_data(R), R=R, net=net)
        isFam = res['data'].tensors[1].bool()
        values, bins = np.histogram(res['a2'][isFam], bins=50, density=True)
        pFP[i] = values[bins[:-1] < 0].sum() * np.diff(bins).mean()

    values, bins = np.histogram(res['a2'][~isFam], bins=50, density=True)
    pFN = values[bins[:-1] > 0].sum() * np.diff(bins).mean()  # p(fam|nov)

    pError = pFP
    pError[-1] = pFN  # p(fam|nov) or p(nov|fam,R)
    pCorrect = 1 - pError  # p(fam|fam,R) or p(nov|nov)

    rtCorrect = 115 - 45 * pCorrect
    rtError = 115 - 45 * pError
    Rs = np.concatenate((Rs, [Rs[-1] * 2]))

    _, ax = plt.subplots()
    lines = ax.semilogx(Rs[:-1], rtCorrect[:-1], marker='.', label='Correct trials')
    ax.semilogx(Rs[-1], rtCorrect[-1], color=lines[0].get_color(), marker='.')

    lines = ax.semilogx(Rs[:-1], rtError[:-1], marker='.', label='Error trials')
    ax.semilogx(Rs[-1], rtError[-1], color=lines[0].get_color(), marker='.')
    ax.set_xlabel('$R_{test}$')
    ax.set_ylabel('Reation time (ms)')

    ax.set_xticks([1, 10, 100, Rs[-1]])
    ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', 'Novel'])
    ax.legend()
    return ax


###############
### Helpers ###
###############
def get_pop_size_acc(fname, net=None, gen_data=None, R=None):
    resultSaveFile = fname + '_popSizeAcc.pkl'
    try:
        res = joblib.load(resultSaveFile)
        popSizeList, acc = res['popSizeList'], res['acc']
    except:
        if R is None:
            R = np.unique(np.logspace(0, 2, dtype=int))
        data = gen_data(R)

        if net is None:
            net = load_from_file(fname)
        N = net.w1.shape[0]

        db = net.evaluate_debug(data.tensors, recogOnly=True)
        h = db['h']
        isFam = (db['data'].tensors[1] == 1).bool().squeeze()

        rankIdx = compute_FLD_weights(h, isFam)[0].argsort()
        popSizeList = range(1, N)
        acc = {d: np.zeros(len(popSizeList)) for d in ['FLD', 'SCC']}
        for decoder in acc.keys():
            print(decoder)
            for i, popSize in enumerate(popSizeList):
                neuronsMask = np.ones(N)
                neuronsMask[rankIdx[popSize:]] = 0
                w2, b2 = compute_FLD_weights(h, isFam, neuronsMask=neuronsMask, SCC=(decoder == 'SCC'))
                net.w2, net.b2 = nn.Parameter(torch.tensor(w2, dtype=torch.float).unsqueeze(0)), nn.Parameter(
                    torch.tensor(b2, dtype=torch.float))

                data = gen_data(R)
                acc[decoder][i] = net.accuracy(data.tensors)

        joblib.dump({'popSizeList': popSizeList, 'acc': acc}, resultSaveFile)
    return popSizeList, acc


def compute_FLD_weights(h, isFam, neuronsMask=None, SCC=False, dPrime=False):
    N = h.shape[1]

    if neuronsMask is None:
        neuronsMask = torch.ones(N, dtype=bool)
    else:  # select subset of neurons to compute decoder
        neuronsMask = neuronsMask.astype(bool)

    hFam = h[isFam, :][:, neuronsMask]
    hNov = h[~isFam, :][:, neuronsMask]
    mN = hNov.mean(dim=0).numpy()
    mF = hFam.mean(dim=0).numpy()

    if SCC:
        w = -np.ones(N) * neuronsMask / neuronsMask.sum()
    else:
        SN = np.cov(hNov, rowvar=False)
        SF = np.cov(hFam, rowvar=False)
        S = (SN + SF) / 2.
        if dPrime and not np.isscalar(S):
            S = S * np.eye(S.shape[0])
        Sinv = np.linalg.inv(S) if not np.isscalar(S) else np.array([1. / S])
        w = np.zeros(N)
        w[neuronsMask] = -np.matmul(Sinv, mN - mF)

    b = -np.dot(w[neuronsMask], (mN + mF) / 2.)
    return w, b


def get_corr_Wxb_Ax(fname, net=None, Rs=None):
    resultSaveFile = fname + '_corrWxbAx.pkl'
    try:
        res = joblib.load(resultSaveFile)
    except:
        assert net is not None
        if Rs is None:
            Rs = np.unique(np.logspace(0, np.log10(3000), 10, dtype=int))
        corr = {nf: {stat: np.zeros(len(Rs)) for stat in ['mean', 'std']}
                for nf in ['novel', 'familiar']}
        for i, R in enumerate(Rs):
            print('R={}'.format(R))
            try:
                N, d = net.w1.shape
            except:
                N, d = net.N, net.D + net.n
            data = generate_recog_data(T=max(R * 20, 5000), d=d, R=R, P=0.5, multiRep=False)
            db = net.evaluate_debug(data.tensors)
            Wxb, Ax = db['Wxb'], db['Ax']
            for nf in ['novel', 'familiar']:
                if nf == 'novel':
                    idx = (data.tensors[1] == 0).squeeze()
                elif nf == 'familiar':
                    idx = (data.tensors[1] == 1).squeeze()

                _C = np.corrcoef(Ax[idx, :], Wxb[idx, :], rowvar=False)
                C = np.diag(_C[:N, N:, ])
                corr[nf]['mean'][i] = C.mean()
                corr[nf]['std'][i] = C.std()
        res = {'Rs': Rs, 'corr': corr}
        joblib.dump(res, resultSaveFile)
    return res


def fit_idealized(R, Pfp, Ptp, D, n, f=2. / 3):
    N = 2 ** n

    def fit_me(R2, a0, b):
        R = R2[:len(R2) // 2]
        lam = 1 - (np.square(a0) * f) / (2 * D * N)  # decay rate
        PfpOld = 0.5 * erfc(a0 * (n + b) / np.sqrt(2)) * np.ones(len(R))
        PtpOld = 0.5 * erfc(a0 * (n + b - np.power(lam, R - 1)) / np.sqrt(2))
        while True:
            fEff = (1 - PfpOld) * f + (1 - PtpOld) * (1 - f)  # fraction of items *reported* as novel
            a = a0 * np.sqrt(f / fEff)
            lam = 1 - (np.square(a) * f) / (2 * D * N)  # decay rate

            Pfp = 0.5 * erfc(a * (n + b) / np.sqrt(2)) * np.ones(len(R))
            Ptp = 0.5 * erfc(a * (n + b - np.power(lam, R - 1)) / np.sqrt(2))

            if np.all(np.abs(PfpOld - Pfp) < 0.0001) and np.all(np.abs(PtpOld - Ptp) < 0.0001):
                break
            PfpOld, PtpOld = Pfp, Ptp
        return np.concatenate([Pfp, Ptp])

    PfpInit = 0.01
    PtpInit = 0.99
    aInit = (erfcinv(2 * PfpInit) - erfcinv(2 * PtpInit)) * np.sqrt(2 * np.e)
    bInit = erfcinv(2 * PfpInit) * np.sqrt(2) / aInit - n
    (aOpt, bOpt), _ = optimize.curve_fit(fit_me, xdata=np.concatenate([R, R]), ydata=np.concatenate([Pfp, Ptp]),
                                         p0=[aInit, bInit], bounds=([1, -np.inf], np.inf))

    Pfp, Ptp = np.split(fit_me(np.concatenate([R, R]), aOpt, bOpt), 2)

    return aOpt, bOpt, Pfp, Ptp


def get_generalization(fname, d, net=None, T=5000, gen_data=None, xscale='log', upToR=float('-inf'),
                       stopAtR=float('inf'), multiRep=False):
    resultSaveFile = '{}_generalization.pkl'.format(fname)
    try:  # load result if it exists
        res = joblib.load(resultSaveFile)
        testR, testAcc, truePosRate, falsePosRate = res['testR'], res['testAcc'], res['truePosRate'], res[
            'falsePosRate']
        print('Loaded {}'.format(resultSaveFile))
    except:  # generate result
        if net is None:
            net = load_from_file(fname)
        if gen_data is None:
            gen_data = lambda R: generate_recog_data(T=max(T, R * 20), R=R, d=d, P=0.5 if not multiRep else 1. / 3,
                                                     multiRep=multiRep)
        testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data, xscale=xscale, upToR=upToR,
                                                                             stopAtR=stopAtR)
        res = {'testR': testR,
               'testAcc': testAcc,
               'truePosRate': truePosRate,
               'falsePosRate': falsePosRate}
        joblib.dump(res, resultSaveFile)

    return testR, testAcc, truePosRate, falsePosRate


def get_recog_positive_rates(net, gen_recog_data, xscale='log', upToR=float('-inf'), stopAtR=float('inf')):
    testR = []
    testAcc = []
    truePosRate = []
    falsePosRate = []

    acc = float('inf')
    truePos = 1;
    falsePos = 0
    R = 1
    while (truePos > falsePos or R < upToR) and R < stopAtR:
        testData = gen_recog_data(R)
        # pdb.set_trace()
        # device = net.Wy.device if isinstance(net, (nnLSTM, STPN.HebbFF.networks.nnLSTM)) else net.w1.device
        device = net.Wy.device if isinstance(net, nnLSTM) else net.w1.device
        testData.tensors = tuple(t.to(device) for t in testData.tensors)
        with torch.no_grad():
            try:
                db = net.evaluate_debug(testData.tensors)
                out, acc, testData = db['out'], db['acc'], db['data']
            except NotImplementedError:
                out = net.evaluate(testData.tensors)
                acc = net.accuracy(testData.tensors, out)
            falsePos, truePos = error_breakdown(out, testData.tensors[1], th=0.5)
        testDataCpu = copy.deepcopy(testData)
        testDataCpu.tensors = tuple(t.cpu() for t in testDataCpu.tensors)
        chance = recog_chance(testDataCpu)
        fracNov = (testData.tensors[1].round() == 0).float().sum() / len(testData)
        acc2 = (1 - falsePos) * fracNov + truePos * (1 - fracNov)
        truePosRate.append(truePos)
        falsePosRate.append(falsePos)
        testAcc.append(acc)
        testR.append(R)
        print('R={}, truePos={:.3f}, falsePos={:.3f}, acc={:.3f}={:.3f}, (chance={:.3f})'.format(R, truePos, falsePos,
                                                                                                 acc, acc2, chance))
        if xscale == 'log':
            R = int(np.ceil(R * 1.3))
        elif xscale == 'linear':
            R += 1
        else:
            raise ValueError
    return testR, testAcc, truePosRate, falsePosRate


def error_breakdown(out, y, th=0.5):
    y = y.round()

    posOutIdx = out > th

    totPos = y.sum().item()
    totNeg = len(y) - totPos

    falsePos = (1 - y)[posOutIdx].sum().item()
    truePos = y[posOutIdx].sum().item()

    falsePosRate = falsePos / totNeg
    truePosRate = truePos / totPos

    return falsePosRate, truePosRate


def get_evaluation_result(fname, data, R, net=None):
    resultSaveFile = '{}_eval_R={}.pkl'.format(fname, R)
    try:  # load result if it exists
        res = joblib.load(resultSaveFile)
        print('Loading {}'.format(resultSaveFile))
    except:  # generate result
        if net is None:
            net = load_from_file(fname)
        res = net.evaluate_debug(data.tensors)
        joblib.dump(res, resultSaveFile)
    return res


def interleave(*tensors):
    '''Each element is tensor, all must be of same length and dimension.
    The first element of the first tensor is first, first element of second is second, etc.'''
    for i in range(len(tensors) - 1):
        if len(tensors[i - 1]) != len(tensors[i]):
            raise ValueError('All tensors must be same length')

    s = list(tensors[0].shape)
    s[0] = s[0] * len(tensors)
    x = np.zeros(s)
    for i, tensor in enumerate(tensors):
        x[range(i, s[0], len(tensors))] = tensor
    return x

"""
Script to generate plots.
Harcoded parameters to chose what to plot
"""
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch
from torch.utils.data import TensorDataset
from torch import nn

from net_utils import load_from_file
import plotting
import networks
from data import generate_recog_data, GenRecogClassifyData

import seaborn as sns
sns.set(font='Arial',
        font_scale=7/12., #default size is 12pt, scale down to 7pt
        palette='Set1',
        rc={'axes.axisbelow': True,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.top': False,
            'text.color': 'dimgrey', #e.g. legend

            'lines.solid_capstyle': 'round',
            'legend.facecolor': 'white',
            'legend.framealpha':0.8,

            'xtick.bottom': True,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',

            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': True,

             'xtick.major.size': 2,
             'xtick.major.width': .5,
             'xtick.minor.size': 1,
             'xtick.minor.width': .5,

             'ytick.major.size': 2,
             'ytick.major.width': .5,
             'ytick.minor.size': 1,
             'ytick.minor.width': .5
            }
        )

import os
os.chdir(os.path.expanduser(os.path.join(RESULTS, 'HebbFF/publish')))

def mm2inch(*args):
    return [x/25.4 for x in args]

def format_and_save(fig, fname, w=None, h=None):
    if w is not None or h is not None:
        fig.set_size_inches(*mm2inch(w, h))
    fig.tight_layout()
    fig.savefig(fname)

which_models = [
    'nnLSTM',
    'HebbNet',
    'uSTPNrNet',
    'STPNfNet',
    'STPNrNet',
    'uSTPNfNet',
]

# TODO: be able to pass seeds to plot
which_figures = [
    # '1a',
    '1b',
]

global_colors = {
    'HebbFF': 'k',
    'STPNr': 'C0',
    'uSTPNr': 'y',
    'STPNf': 'g',
    'uSTPNf': 'm',

    # 'HebbNet': 'k',
    # 'STPNrNet': 'C0',
    # 'uSTPNrNet': 'y',
    # 'STPNfNet': 'g',
    # 'uSTPNfNet': 'm',

    'HebbNet': 'k',
    'STPNrNet': 'tab:brown',
    'uSTPNrNet': 'tab:pink',
    'STPNfNet': 'tab:gray',
    'uSTPNfNet': 'tab:olive',
    'nnLSTM': 'tab:cyan',
}

global_labels = {
    'uSTPNfNet': 'uSTPNf',
    'STPNfNet': 'STPNf',
    'uSTPNrNet': 'uSTPNr',
    'STPNrNet': 'STPNr',
    'HebbNet': 'HebbFF',
    'nnLSTM': 'LSTM',
}

n_exp_runs = 5 # 3

if bool(set(which_models) & {'uSTPNrNet', 'STPNfNet', 'STPNrNet', 'uSTPNfNet', 'HebbNet', "nnLSTM"}):
    all_gens = {3: {}, 6: {}}
    for model_name in which_models:
        if model_name in ['uSTPNrNet', 'STPNfNet', 'STPNrNet', 'uSTPNfNet', 'HebbNet', "nnLSTM"]:
            #%% Fig 2: RNN
            #(a) Train on single dataset
            # we have developed our own for this, which supports multiple seeds and is more flexible
            N = {'uSTPNrNet': 62, 'STPNfNet': 34, 'STPNrNet': 27, 'uSTPNfNet': 100, 'HebbNet': 100, 'nnLSTM': 21}[model_name]
            alt_model_paths = {'HebbNet': 'antiHebb', 'nnLSTM': 'RNN'}
            model_path = alt_model_paths.get(model_name, model_name)
            d = 100
            if '1a' in which_figures:
                # # single seed
                if n_exp_runs == 1:
                    ax = plotting.plot_loss_acc('{}/{}[{},{},1]_train=dat3_T=5000.pkl'.format(model_path,model_name, d,N), chance=2./3)
                    format_and_save(ax[0].get_figure(), f'{model_path}/single.pdf', w=58, h=70)
                else:
                    raise NotImplementedError
            #(b)
            if '1b' in which_figures:
                ax = None
                for R in [3,6]:
                    # all_gens[R] = {}
                    # # for single seed
                    if n_exp_runs == 1:
                        fname = '{}/{}[{},{},1]_train=inf{}.pkl'.format(model_path,model_name,d,N,R)
                        gen = plotting.get_generalization(fname, d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
                        ax = plotting.plot_generalization(*gen, label='$R_{{train}}={}$'.format(R), chance=2./3, xscale='linear', ax=ax)

                    # for multiple seeds
                    else:
                        for n_seed in range(n_exp_runs):
                            if n_seed > 0:
                                fname = '{}/{}[{},{},1]_train=inf{}_({}).pkl'.format(model_path, model_name, d, N, R, n_seed+1)
                            else:
                                fname = '{}/{}[{},{},1]_train=inf{}.pkl'.format(model_path, model_name, d, N, R)
                            gen = plotting.get_generalization(fname, d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
                            if n_seed == 0:
                                # skip
                                # all_gens[R][model_name] = tuple([this_metric] for this_metric in gen)
                                all_gens[R][model_name] = tuple([this_metric] if i_m > 0 else this_metric for i_m, this_metric in enumerate(gen))
                            else:
                                # skip the first one, which is just R
                                all_gens[R][model_name] = tuple(all_gens[R][model_name][i_m] + [this_metric] if i_m > 0 else this_metric for i_m, this_metric in enumerate(gen))

            # #(c)
            # gen = plotting.get_generalization(
            #     '{}/{}[{},{},1]_train=inf[3,6].pkl'.format(model_name,model_name,d,N),
            #     d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
            # ax = plotting.plot_generalization(*gen, label='$R_{{train}}=[3,6]$', chance=2./3, xscale='linear')
            #
            # format_and_save(ax[0].get_figure(), f'{model_name}/twoR.pdf', w=58, h=70)

    if '1b' in which_figures and n_exp_runs > 1:
        for R in [3, 6]:
            ax = plotting.plot_multiseed_multimodel_generalization(
                all_gens[R],
                labels=global_labels, # label='$R_{{train}}={}$'.format(R),
                chance=2. / 3, xscale='linear',
                ax=None, # don't pass to keep R 3 and 6 separated
                colors = global_colors,
                alpha=0.1, # in kwargs
            )
            format_and_save(ax[0].get_figure(), f'infinite_R{R}_seeds_{n_exp_runs}_' + '_'.join(all_gens[R].keys()) + '.pdf', w=58, h=70)

if 'LSTM' in which_models:
    #%% Fig 2: RNN
    #(a) Train on single dataset
    d = N = 100
    ax = plotting.plot_loss_acc('RNN/nnLSTM[{},{},1]_train=dat3_T=5000.pkl'.format(d,N), chance=2./3)

    format_and_save(ax[0].get_figure(), 'RNN/single.pdf', w=58, h=70)

    #(b)
    ax = None
    for R in [3,6]:
        fname = 'RNN/nnLSTM[{},{},1]_train=inf{}.pkl'.format(d,N,R)
        gen = plotting.get_generalization(fname, d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
        ax = plotting.plot_generalization(*gen, label='$R_{{train}}={}$'.format(R), chance=2./3, xscale='linear', ax=ax)

    format_and_save(ax[0].get_figure(), 'RNN/infinite.pdf', w=58, h=70)

    #(c)
    gen = plotting.get_generalization('RNN/nnLSTM[{},{},1]_train=inf[3,6].pkl'.format(d,N), d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
    ax = plotting.plot_generalization(*gen, label='$R_{{train}}=[3,6]$', chance=2./3, xscale='linear')

    format_and_save(ax[0].get_figure(), 'RNN/twoR.pdf', w=58, h=70)

if 'HebbFF' in which_models:
    #%% Fig 3: anti-Hebbian and continual
    #(a) Train on single dataset
    d = N = 100
    ax = plotting.plot_loss_acc('antiHebb/HebbNet[{},{},1]_train=dat3_T=5000.pkl'.format(d,N), chance=2./3)
    format_and_save(ax[0].get_figure(), 'antiHebb/single.pdf', w=58, h=70)

    #(b) Hebb inf, (c) Anti inf
    for force in ['Hebb', 'Anti']:
        ax = None
        for R in [3,6]:
            fname = 'antiHebb/HebbNet[{},{},1]_train=inf{}_force{}.pkl'.format(d,N,R,force)
            net = load_from_file(fname)
            label = '$R_{{train}}={}$ \n $\lambda={:.2f}$, $\eta={:.2f}$'.format(R, net.lam, net.eta)
            gen = plotting.get_generalization(fname, d=d, T=5000, upToR=50, stopAtR=50)
            ax = plotting.plot_generalization(*gen, chance=2./3,  xscale='linear', label=label, ax=ax)
        format_and_save(ax[0].get_figure(), 'antiHebb/{}.pdf'.format(force), w=58, h=70)

    #(d), (e)
    for resultFile in ['bogacz/bogacz_continual.pkl', 'bogacz/hebbff_continual.pkl']: #from bogacz/bogacz.py
        res = joblib.load(resultFile)
        ax = plotting.plot_acc_vs_T(chance=2./3, **res )
        format_and_save(ax[0].get_figure(), resultFile[:-4]+'.pdf', w=58, h=70)

if '4_Mechanism' in which_figures:
    #%% Fig 4: Mechanism
    # Hidden activity, weight matrix, W1x+b1 histogram for various R_train
    ax = [plt.subplots(3,1, sharex='col')[1] for _ in range(3)]
    data1 = generate_recog_data(T=5000, R=1, d=25, P=0.5, multiRep=False)
    for i,Rtrain in enumerate([1,7,14]):
        fname = 'mechanism/HebbNet[25,25,1]_R={}.pkl'.format(Rtrain)
        net = load_from_file(fname)

        res = plotting.get_evaluation_result(fname, data1, R=1)
        h = res['h'][:20]
        out = res['out'][:20]
        isFam = res['data'].tensors[1].bool()[:20]
        plotting.plot_hidden_activity(h, isFam, out, ax=ax[0][i])
        ax[0][i].set_ylabel('$R_{{train}}={}$\n\n Neuron'.format(Rtrain))

        plotting.plot_weight(net.w1.detach(), ax=ax[1][i])

        data = generate_recog_data(T=5000, R=Rtrain, d=25, P=0.5, multiRep=False)
        Wxb = plotting.get_evaluation_result(fname, data, R=Rtrain)['Wxb']
        ax[2][i].hist(Wxb.flatten().numpy(), bins=50, density=True, histtype='step', align='mid', color='black')
        ax[2][i].set_ylabel('Probability')
    ax[2][i].set_xlabel('$W_1x(t)+b_1$')
    [a.set_xlabel('') for a in ax[0][0:2]]
    ax[1][-1].set_xlabel(' ')

    format_and_save(ax[0][0].get_figure(), 'mechanism/mechanism_Rtrain_h.pdf', w=74, h=100)
    format_and_save(ax[1][0].get_figure(), 'mechanism/mechanism_Rtrain_W1.pdf', w=45, h=100)
    format_and_save(ax[2][0].get_figure(), 'mechanism/mechanism_Rtrain_Wxb.pdf', w=58, h=100)

    #%% Ax and (W1+A)x+b1 histograms and corr(Wx, Ax)(t) for various R, from Rtrain=Rmax
    fname = 'mechanism/HebbNet[25,25,1]_R=14.pkl'
    ax = [plt.subplots(3,1, sharex='col')[1] for _ in range(2)] + [plt.subplots(3,1)[1]]

    for i, R in enumerate([14, 40, 100]):
        data = generate_recog_data(T=5000, R=R, d=25, P=0.5, multiRep=False)
        res = plotting.get_evaluation_result(fname, data, R)
        for j, (var, label) in enumerate([(res['Ax'],   '$A(t)x(t)$'),
                                          (res['a1'], '$(W_1+A(t))x(t)+b_1$')]):
            isFam = res['data'].tensors[1].expand(-1, var.shape[1]).flatten()
            plotting.plot_distribution(var.flatten().numpy(), isFam, mean=False, xlabel=label, ax=ax[j][i])
        ax[1][i].set_ylabel('Probability')
        ax[0][i].set_ylabel('$R_{{test}}={}$\n\n Probability'.format(R))

        Wxb = res['Wxb'][1:]
        Ax = res['Ax'][1:]
        isFam = res['data'].tensors[1].bool()[1:]
        corr = np.zeros(Wxb.shape[0])
        for t in range(len(Wxb)):
            corr[t] = np.corrcoef(Wxb[t], Ax[t])[0,1]
        print(np.corrcoef(isFam.flatten(), corr)[0,1])

        Wxb = res['Wxb'][R:R+20]
        Ax = res['Ax'][R:R+20]
        isFam = res['data'].tensors[1].bool()[R:R+20]
        out = res['out'][R:R+20]
        plotting.plot_corr_per_timepoint(Wxb, Ax, isFam, out, ax=ax[2][i])

    [a.set_ylim([-0.9, -0.2]) for a in ax[2]]
    [[a.set_xlabel('') for a in ax[j][0:2]] for j in range(3)]
    ax[0][0].legend(['Novel', 'Familiar'])
    format_and_save(ax[0][0].get_figure(), 'mechanism/mechanism_Rtest_Ax.pdf', w=61, h=100)
    format_and_save(ax[1][0].get_figure(), 'mechanism/mechanism_Rtest_WAxb.pdf', w=58, h=100)
    format_and_save(ax[2][0].get_figure(), 'mechanism/mechanism_Wx_Ax_time.pdf', w=58, h=100)


if 'S4_Weighted_vs_simple' in which_figures:
    #%% Fig S4: Weighted vs simple average comparison
    files = [('mechanism/HebbNet[25,25,1]_train=cur2_incr=plus1.pkl', 'Weighted'),
             ('mechanism/HebbNet[25,25,1]_R=14.pkl', 'Uniform')
    ]

    #(a) Generalization performance
    ax=None
    for fname, label in files:
        net = load_from_file(fname)
        N,d = net.w1.shape
        gen = plotting.get_generalization(fname, d, net=net, multiRep=False)
        ax = plotting.plot_generalization(*gen, label=label, ax=ax)
    format_and_save(ax[0].get_figure(), 'mechanism/gen_supp.pdf'.format(label), w=58, h=70)

    #%%
    # (b) Weight matrices
    for fname, label in files:
        net = load_from_file(fname)
        if label == 'Simple':
            w2T = net.w2.detach().repeat(25,1)
            b1 = net.b1.detach().repeat(25,1)
        else:
            w2T = net.w2.detach().t()
            b1 = net.b1.detach().unsqueeze(1)
        ax = plotting.plot_weight(w2T)
        ax.set_title('$W_2^T$')
        format_and_save(ax.get_figure(), 'mechanism/{}_w2_supp.pdf'.format(label), w=10, h=50)

        ax = plotting.plot_weight(b1)
        ax.set_title('$b_1$')
        format_and_save(ax.get_figure(), 'mechanism/{}_b1_supp.pdf'.format(label), w=10, h=50)

    #%%
    # (c) Readout distribution
    gen_data = lambda R: generate_recog_data(T=max(10000, R*20), d=d, R=R, P=0.5, multiRep=False)
    for fname, label in files:
        net = load_from_file(fname)
        Rmax = net.hist['increment_R'][-1][1]
        Rs = [1, Rmax, 2*Rmax, 4*Rmax]
        ax = plotting.plot_familiar_output_distr(fname, gen_data, Rs=Rs, net=net, plotNov=True)
        ax.set_xlabel('$W_2h(t)+b_2$')
        ax.set_xlim((-30,12))
        ax.set_title(label)
        format_and_save(ax.get_figure(), 'mechanism/out_distr_supp_{}.pdf'.format(label), w=58, h=50)

if '5_Empirical_capacity' in which_figures:
    #%% Fig 5: Empirical capacity
    ds = np.array([25, 50, 100, 200, 400])
    Ns = np.array([  1,   2,   4,   8,  16,  32,  64, 128, 256])
    Rmax = np.full((len(ds), len(Ns)), np.nan)
    for i,d in enumerate(ds):
        for j,N in enumerate(Ns):
            fname = 'capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N)
            try:
                net = load_from_file(fname)
                Rmax[i,j] = net.hist['increment_R'][-1][1]
            except IOError:
                pass
    #
    fig,ax = plt.subplots(2,2)
    # selected nets to plot R over time
    netDims = [(50,8), (50,16), (100,16), (100,32)]
    Riter = []
    for i,(d,N) in enumerate(netDims):
        fname = 'capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N)
        net = load_from_file(fname)
        iters, Rs = map(list, zip(*net.hist['increment_R']))
        iters.append( net.hist['iter'] )
        Rs.append( net.hist['increment_R'][-1][1] )
        label = 'd={}, N={}'.format(d,N)
        ax[0,0] = plotting.plot_R_curriculum(iters, Rs, label=label, ax=ax[0,0])
        ax[0,0].legend()

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls = '-'
    mkr = '.'

    #% Plot Rmax(N) for each value of d
    # ax[1,1].set_prop_cycle('color', plt.cm.Reds(np.linspace(0,1,5)))
    for i,d in enumerate(ds):
        print(d)
        ax[1,1] = plotting.plot_and_fit(Ns, Rmax[i,:], ax=ax[1,1], marker=mkr, linestyle=ls, label='d={}'.format(d))
    ax[1,1].set_xlabel('Number of hidden units ($N$)')

    #% Plot Rmax(d) for selected values of N
    # ax[1,0].set_prop_cycle('color', plt.cm.Greens(np.linspace(0,1,5)))
    for j,N in enumerate(Ns):
        if N not in [8,  16,  32,  64, 128]:
            continue
        print(j,N)
        if N == 128:
            print(Rmax[:,j])
        ax[1,0] = plotting.plot_and_fit(ds, Rmax[:,j], ax=ax[1,0], marker=mkr, linestyle=ls, label='N={}'.format(N))
    ax[1,0].set_xlabel('Number of input units ($d$)')

    #% Plot Rmax(N*d)
    Nsyn = np.outer(ds,Ns)[~np.isnan(Rmax)]
    idx = np.argsort(Nsyn)
    Nsyn = Nsyn[idx]
    _Rmax = Rmax[~np.isnan(Rmax)]
    _Rmax = _Rmax[idx]
    ax[0,1] = plotting.plot_and_fit(Nsyn, _Rmax, color='k', linestyle=ls, marker=None, ax=ax[0,1], label='Color corresp. to $d$')
    ax[0,1].set_xlabel('Number of synapses ($N \cdot d$)')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,d in enumerate(ds):
        ax[0,1].loglog(d*Ns, Rmax[i,:], color=cycle[i%len(cycle)], linestyle='', marker=mkr)

    fig.tight_layout()
    format_and_save(fig, 'capacity/capacity.pdf', w=174, h=120)

if '6_Idealized_model' in which_figures:
    #%% Fig 6: Idealized model
    #(b) Simulated and analytic model generalization
    ax = None
    evens=True
    for D,n in [(50,6), (100,7)]:
        d = D+n
        f = 2./3
        net = networks.DetHebb(D, n, f=f, Pfp=0.005, Ptp=0.995)
        fname = 'idealized/idealized_{}x{}'.format(D,2**n)

        Rs, acc, tp, fp = plotting.get_generalization(fname, d, net=net)
        if evens:
            ax = plotting.plot_generalization(Rs[::2], acc[::2], tp[::2], fp[::2], ax=ax, label='Sim {}x{}'.format(D,2**n), ls='', marker='x', markersize=3)
            evens=False
        else:
             ax = plotting.plot_generalization(Rs[1::2], acc[1::2], tp[1::2], fp[1::2], ax=ax, label='Sim {}x{}'.format(D,2**n), ls='', marker='x', markersize=3)

        tp, fp = net.true_false_pos_analytic(np.array(Rs), corrected=True)
        acc = (1-f)*tp+f*(1-fp)
        print(ax[0].get_lines()[-1].get_color())
        plotting.plot_generalization(Rs, acc, tp, fp, ax=ax, label='Analytic'.format(D,2**n), color=ax[0].get_lines()[-1].get_color(), chance=2./3)

    format_and_save(ax[0].get_figure(), 'idealized/sim_vs_ana.pdf', w=58, h=70)

    #%% (c) Fit idealized to HebbFF
    ax = None
    for d,N in [(100,32)]:#,(200,32)]:
        n = int(np.log2(N))
        D = d-n
        f = 2./3
        fname = 'capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N)
        net = load_from_file(fname)
        Rs, accHebb, PtpHebb, PfpHebb = plotting.get_generalization(fname, d)
        aOpt, bOpt, Pfp, Ptp = plotting.fit_idealized(np.array(Rs), np.array(PfpHebb), np.array(PtpHebb), D, n, f=f)
        acc = (1-f)*Ptp+f*(1-Pfp)
        lam = 1 - (np.square(aOpt)*f)/(2*D*N) #decay rate

        ax = plotting.plot_generalization(Rs, accHebb, PtpHebb, PfpHebb, ax=ax, label='HebbFF ({}x{})'.format(d,N), ls='', marker='.', markersize=3)
        plotting.plot_generalization(Rs, acc, Ptp, Pfp, ax=ax, color=ax[0].get_lines()[-1].get_color(), label='Idealized fit', chance=2./3)
    format_and_save(ax[0].get_figure(), 'idealized/fit_to_hebbff.pdf', w=58, h=70)


    #%% (d) Wx+b, (e) Ax (f) WAxb histogram
    D = 400
    n = 5
    fname = 'idealized/idealized_hist_{}x{}.pkl'.format(D,2**n)
    try:
        res = joblib.load(fname)
    except:
        d = D+n
        f = 2./3
        net = networks.DetHebb(D, n, f=f, Pfp=0.005, Ptp=0.995)
        data = generate_recog_data(T=200000, d=d, R=300, P=0.5, multiRep=False)
        res = net.evaluate_debug(data.tensors)
        Wxb, Ax, WAxb = res['Wxb'], res['Ax'], res['a1']
        joblib.dump(res, fname)

    fig, ax = plt.subplots()
    ax.hist(Wxb.flatten(), bins=50, density=True, histtype='step', align='mid', color='black')
    ax.set_xlabel('$Wx_w(t)+B$')
    ax.set_ylabel('Probability')
    format_and_save(ax.get_figure(), 'idealized/Wxb.pdf', w=58, h=45)

    isFam = data.tensors[1].expand(-1, Ax.shape[1]).flatten()
    ax = plotting.plot_distribution(Ax.flatten(), isFam, mean=False, xlabel='$A(t)x_A(t)$')
    ax.set_ylabel('Probability')
    ax.legend(['Novel', 'Familiar'])
    format_and_save(ax.get_figure(), 'idealized/Ax.pdf', w=58, h=45)

    ax = plotting.plot_distribution(WAxb.flatten(), isFam, mean=False, xlabel='$Wx_w(t)+A(t)x_A(t)+B$')
    ax.set_ylabel('Probability')
    format_and_save(ax.get_figure(), 'idealized/WAxb.pdf', w=58, h=45)

if '7_S7_IT_data' in which_figures:
    #%% Fig 7, S7: IT data
    files = ['inferotemporal/HebbClassify[25,50,2]_train=cur1_incr=plus1.pkl',
              'inferotemporal/HebbNet[25,50,1]_train=cur2_incr=plus1.pkl']

    for fname in files:
        netStr = fname[fname.find('/')+1:fname.find('[')]
        net = load_from_file(fname)
        N,d = net.w1.shape
        net.onlyRecogAcc=True
        def gen_data(R):
            T = max(R*100, 10000) if np.isscalar(R) else max(max(R)*100, 10000)
            data = generate_recog_data(T=T, d=d, R=R, P=0.5, multiRep=False)
            if type(net) == networks.HebbClassify:
                data.tensors = (data.tensors[0], torch.cat((data.tensors[1], float('nan')*data.tensors[1]), dim=1))
            return data

    # (a) Acc vs pop size
        fig, ax = plt.subplots()
        popSizeList, acc = plotting.get_pop_size_acc(fname, gen_data=gen_data)
        for decoder in ['SCC', 'FLD']:
            ax.plot(popSizeList, acc[decoder], label=decoder)
        ax.set_xlabel('Population size')
        ax.set_ylabel('Accuracy')
    #    ax.set_title(netStr)
        ax.legend()
        format_and_save(ax.get_figure(), 'inferotemporal/acc_v_pop_{}.pdf'.format(netStr), w=58, h=50)

    # (b) distribution of readout weight
        data = gen_data(np.unique(np.logspace(0,2, dtype=int)))
        db = net.evaluate_debug(data.tensors, recogOnly=True)
        isFam = (db['data'].tensors[1]==1).bool().squeeze()
        w2, b2 = plotting.compute_FLD_weights(db['h'], isFam)

        print('{} W2FLD outliers: {}'.format(netStr, w2[np.abs(w2)>100]))
        w2o = w2[np.abs(w2)<100] #ad hoc remove outliers

        fig, ax = plt.subplots()
        _,_,patches = ax.hist(w2o, density=True, bins=30, histtype='step', align='mid')
        ax.axvline(0, color='k')
        ax.set_xlabel('$W_2^{FLD}$')
        ax.set_ylabel('Probability')
    #    ax.set_title(fname[fname.find('/')+1:fname.find('[')])
        format_and_save(fig, 'inferotemporal/w2fld_{}.pdf'.format(netStr), w=58, h=50)


    # (c) Distribution of readout for familiar stimuli
        net.w2, net.b2 = nn.Parameter(torch.tensor(w2, dtype=torch.float).unsqueeze(0)), nn.Parameter(torch.tensor(b2, dtype=torch.float))
        Rmax = net.hist['increment_R'][-1][1]
        Rs = [1, Rmax, 2*Rmax, 4*Rmax]
        ax = plotting.plot_familiar_output_distr(fname[:-4]+'_W2FLD.pkl', gen_data, Rs=Rs, net=net, plotNov=True)
        ax.set_xlabel('$W_2^{FLD}h(t)+b_2$')
        format_and_save(ax.get_figure(), 'inferotemporal/out_distr_{}.pdf'.format(netStr), w=58, h=50)


    # (d) Reaction time curves
        Rs = np.unique(np.logspace(0,np.log10(500),20,dtype=int))
        ax = plotting.plot_RT_curves(fname, gen_data, Rs=Rs, net=net)
        format_and_save(ax.get_figure(), 'inferotemporal/rt_{}.pdf'.format(netStr), w=58, h=50)

    #%% (e) Weight matrix
    net = load_from_file('inferotemporal/HebbClassify[25,50,2]_train=cur1_incr=plus1.pkl')
    ax = plotting.plot_weight(net.w1.detach())
    ax.set_title('$W_1$')
    format_and_save(ax.get_figure(), 'inferotemporal/w1_HebbClassify.pdf', w=58, h=60)

    ax = plotting.plot_weight(net.w2.detach().t())
    ax.set_title('$W_2^T$')
    format_and_save(ax.get_figure(), 'inferotemporal/w2_HebbClassify.pdf', w=58, h=60)

    ax = plotting.plot_weight(net.b1.detach().unsqueeze(1))
    ax.set_title('$b_1$')
    format_and_save(ax.get_figure(), 'inferotemporal/b1_HebbClassify.pdf', w=58, h=60)

    ax = plotting.plot_weight(net.b2.detach().unsqueeze(1))
    ax.set_title('$b_2$')
    format_and_save(ax.get_figure(), 'inferotemporal/b2_HebbClassify.pdf', w=58, h=60)


if '8_ConvNet_input' in which_figures:
    #%% Fig 8, S8: Conv net input
    for netFile, imgsFile, label in [
            ('conv/HebbNet[50,16,1]_train=cur2_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl_b1init=scalar_w2init=scalar.pkl',
                               'conv/BradyOliva2008_UniqueObjects_ResNet18_d=50_binarize.pkl', 'binarize'),
            ('conv/HebbFeatureLayer[50,16,1]_train=cur1_incr=plus1_sampleSpace=BradyOliva2008_UniqueObjects_ResNet18.pkl_Nx=512_b1init=scalar-10_w2init=scalar-10_b2init=scalar10.pkl',
                               'conv/BradyOliva2008_UniqueObjects_ResNet18.pkl', 'featurize')
        ]:

        net = load_from_file(netFile)
        N,d = net.w1.shape
        print(net.hist['increment_R'][-1][1])
        continue

        images = torch.load(imgsFile)
        nImgs = images.shape[0]
        dummy = torch.zeros(nImgs,1)
        sampleSpace = TensorDataset(images, dummy)
        # This was designed to generate the Familiarity+Classification dataset, but we can hack it to generate familiarity
        # data from a fixed pool of images without repeats by adding a dummy variable for class which we then throw away
        generator = GenRecogClassifyData(sampleSpace=sampleSpace)
        def generate_recog_images(T, d, R, P=0.5, batchSize=None, multiRep=False):
            x,y = generator(T, R, P, batchSize=batchSize, multiRep=multiRep).tensors
            return TensorDataset(x, y[...,0:1])

        #% Input distribution
        if type(net) == networks.HebbFeatureLayer:
            downsampled = torch.empty(len(images),50)
            for i,x in enumerate(images):
                downsampled[i] = net.featurizer(x).detach()
        else:
            downsampled = images

        _,ax = plt.subplots(1,2)
        ax[0].hist(downsampled.flatten(), bins=100, histtype='step', align='mid', density=True)
        ax[0].set_xlabel('$x_i(t)$')
        ax[0].set_ylabel('Probability')

        # Input correlation
        _C = np.corrcoef(downsampled)
        C = _C[~np.eye(_C.shape[0],dtype=bool)]
        ax[1].hist(C, bins=100, histtype='step', align='mid', density=True)
        ax[1].axvline(C.mean(), color='k', ls=':')
        ax[1].set_xlim((-1,1))
        ax[1].set_xlabel('corr($x_i$, $x_j$)')
        #ax[1].set_ylabel('Probability')
        format_and_save(ax[0].get_figure(), 'conv/{}_info.pdf'.format(label), w=87, h=35)

        # W1 matrix
        ax = plotting.plot_weight(net.w1.detach())
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        format_and_save(ax.get_figure(), 'conv/{}_w1.pdf'.format(label), w=87, h=35)


        #% Wx+b histogram
        Rmax = net.hist['increment_R'][-1][1]
        data = generate_recog_images(T=nImgs, d=net.w1.shape[1], R=Rmax)
        res = plotting.get_evaluation_result(netFile, data, R=Rmax)

        _,ax = plt.subplots(2,2)
        ax = ax.flatten()
        ax[0].hist(res['Wxb'].flatten(), bins=50, density=True, histtype='step', align='mid', color='black')
        ax[0].set_xlabel('$W_1x(t)+b_1$')
        ax[0].set_ylabel('Probability')

        # Ax histogram
        isFam = res['data'].tensors[1].expand(-1, N).flatten()
        plotting.plot_distribution(res['Ax'].flatten(), isFam, ax=ax[1], xlabel='$A(t)x(t)$')
        ax[1].set_ylabel('')

        # (W+A)x+b histogram
        plotting.plot_distribution(res['a1'].flatten(), isFam, ax=ax[2], xlabel='$(W_1+A(t))x(t)+b_1$')
        ax[2].set_ylabel('Probability')


        #% Readout Wh+b
        isFam = res['data'].tensors[1].flatten()
        ax[3] = plotting.plot_distribution(res['a2'].flatten(), isFam, ax=ax[3], xlabel='$W_2h(t)+b_2$')
        ax[3].legend(['Novel', 'Familiar'])
        format_and_save(ax[0].get_figure(), 'conv/{}_histograms.pdf'.format(label), w=87, h=70)


        #% Hidden activation
        data = generate_recog_images(T=nImgs, d=net.w1.shape[1], R=1)
        res = plotting.get_evaluation_result(netFile, data, R=1)
        h = res['h'][:20]
        out = res['out'][:20]
        isFam = res['data'].tensors[1].bool()[:20]
        ax = plotting.plot_hidden_activity(h, isFam, out)
        format_and_save(ax.get_figure(), 'conv/{}_hidden.pdf'.format(label), w=87, h=35)


        #% Generalization performance
        gen_data = lambda R: generate_recog_images(T=nImgs, d=net.w1.shape[1], R=R)
        gen = plotting.get_generalization(netFile, d, net=net, gen_data=gen_data, upToR=300, stopAtR=500)
        ax = plotting.plot_generalization(*gen, label='Images')
        # control
        fname = 'conv/HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'
        net = load_from_file(fname)
        gen = plotting.get_generalization(fname, d, net=net)
        ax = plotting.plot_generalization(*gen, label='Uncorrelated', ax=ax)
        format_and_save(ax[0].get_figure(), 'conv/{}_gen.pdf'.format(label), w=58, h=70)

if 'S2_Compare_RNN_HebbFF_memory_units' in which_figures:
    #%% Fig S2: Compare RNN and HebbFF, matching number of dynamic vars instead of neurons
    #(a)
    d = 25
    N = 25*25
    ax = plotting.plot_loss_acc('RNN/nnLSTM[{},{},1]_train=dat3_T=5000.pkl'.format(d,N), chance=2./3)
    format_and_save(ax[0].get_figure(), 'RNN/dataset_supp.pdf', w=58, h=70)

    #(b)
    ax = None
    for R in [3,6]:
        gen = plotting.get_generalization('RNN/nnLSTM[{},{},1]_train=inf{}.pkl'.format(d,N,R), d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
        ax = plotting.plot_generalization(*gen, chance=2./3, label='$R_{{train}}={}$'.format(R), xscale='linear', ax=ax)
    format_and_save(ax[0].get_figure(), 'RNN/infinite_supp.pdf', w=58, h=70)

    #(c)
    N = 25
    ax = plotting.plot_loss_acc('antiHebb/HebbNet[{},{},1]_train=dat3_T=5000.pkl'.format(d,N), chance=2./3)
    format_and_save(ax[0].get_figure(), 'antiHebb/dataset_supp.pdf', w=58, h=70)

    #(d)
    ax = None
    for R in [3,6]:
        gen = plotting.get_generalization('antiHebb/HebbNet[{},{},1]_train=inf{}_forceAnti.pkl'.format(d,N,R), d=d, T=5000, xscale='linear', upToR=20, stopAtR=20)
        ax = plotting.plot_generalization(*gen, chance=2./3, label='$R_{{train}}={}$'.format(R), xscale='linear', ax=ax)
    format_and_save(ax[0].get_figure(), 'antiHebb/infinite_supp.pdf', w=58, h=70)

if 'S3_Bogacz' in which_figures:
    #%% Fig S3: Bogacz validation
    #(a) Bogacz network on non-continual task
    res = joblib.load('bogacz/bogacz_non_continual.pkl')
    fig, ax = plt.subplots(2,1, sharex=True)

    lines = ax[0].semilogx(res['Plist'], res['acc'].T)
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(lines, ['$\eta$={}'.format(eta) for eta in res['etaList']])

    linesTP = ax[1].semilogx(res['Plist'], res['tp'].T)
    linesFP = ax[1].semilogx(res['Plist'], res['fp'].T, '--')
    for i,line in enumerate(linesTP):
        linesFP[i].set_color(line.get_color())
    ax[1].set_xlabel('$P$ (# patterns presented)')
    ax[1].set_ylabel('Probability')
    ax[1].plot([], 'k-', label='True positive')
    ax[1].plot([], 'k--', label='False positive')
    ax[1].legend()
    format_and_save(ax[0].get_figure(), 'bogacz/non_continual_supp.pdf', w=58, h=70)


    #%% (b) Bogacz output histograms on continual
    res = joblib.load('bogacz/bogacz_out_hist.pkl')
    out, target, Tlist = res['out'], res['target'], res['Tlist']

    _,ax = plt.subplots(2,2, squeeze=False, sharex=True, sharey=False)
    ax = ax.flatten()
    for j,T in enumerate(Tlist):
        fam = out[j][target[j]]
        nov = out[j][~target[j]]
        thres = np.quantile(out[j], target[j].mean())

        _,_,patches = ax[j].hist(nov, bins=100, color='red', density=True, histtype='step', align='mid')
        ax[j].plot([], color=patches[0].get_edgecolor(), label='Novel')

        _,_,patches = ax[j].hist(fam, bins=100, color='green', density=True, histtype='step', align='mid')
        ax[j].plot([], color=patches[0].get_edgecolor(), label='Familiar')

        ax[j].axvline(thres, ls='-', c='k', lw=0.75)
        ax[j].set_title('T={}'.format(T))
    ax[0].set_ylabel('Probability')
    ax[2].set_ylabel('Probability')
    ax[2].set_xlabel('Output')
    ax[3].set_xlabel('Output')
    ax[-1].legend()
    plt.tight_layout()
    format_and_save(ax[0].get_figure(), 'bogacz/histograms_supp.pdf', w=87, h=70)

if 'S5_Curriculum_R' in which_figures:
    #%% Fig S5: Curriculum R over time for all networks
    iters, Rs = zip(*net.hist['increment_R'])
    iters = list(iters)
    Rs = list(Rs)
    iters.append( net.hist['iter'] )
    Rs.append( net.hist['increment_R'][-1][1] )
    ax = plotting.plot_R_curriculum(iters, Rs, label=label, ax=ax)

if 'S6_Idealized_model_vs_HebbFF' in which_figures:
    #%% Fig S6: Idealized model vs HebbFF differences
    # (a-b): Correlation between Wx and Ax
    d = 25
    N = 32
    n = int(np.log2(N))

    _hebbff = 'capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N)
    nets = [('idealized/idealized_{}x{}'.format(d,N), networks.DetHebb(D=d-n, n=n, f=2./3), 'Idealized'),
            (_hebbff,                                 load_from_file(_hebbff),              'HebbFF')]

    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
    for i, (fname, net, label) in enumerate(nets):
        res = plotting.get_corr_Wxb_Ax(fname, net)
        Rs, corr = res['Rs'], res['corr']
        for novOrFam in ['novel', 'familiar']:
            corrAvg = corr[novOrFam]['mean']
            corrStd = corr[novOrFam]['std']
            line = ax[i].semilogx(Rs, corrAvg, color='red' if novOrFam=='novel' else 'green', label=novOrFam.capitalize())[0]
    #        ax[i].fill_between(Rs, corrAvg+corrStd, corrAvg-corrStd, alpha=0.3, color=line.get_color())
    #    ax[i].set_title(label)
        ax[i].set_xlabel('$R_{test}$')
    ax[0].set_ylabel('Correlation')
    ax[1].legend()
    fig.tight_layout()
    format_and_save(ax[0].get_figure(), 'idealized/corr_supp.pdf', w=116, h=45)

    #%% (c-d): Performance with multiple repeats
    N = 32
    d = 25
    n = int(np.log2(N))
    D = d-n

    nets = [('capacity/HebbNet[{},{},1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl'.format(d,N), 'HebbFF'),
            ('idealized/idealized_{}x{}'.format(d,N), 'Idealized')]
    for fname, label in nets:
        ax = None
        for multiRep in [False, True]:
            if fname.startswith('idealized'):
                f = 2./3
                net = networks.DetHebb(D, n, f=f, Pfp=0.005, Ptp=0.995)
            else:
                net = load_from_file(fname)
            fname = fname + ('_multiRep' if multiRep else '')
            Rs, acc, tp, fp = plotting.get_generalization(fname, d, net=net, multiRep=multiRep)
            ax = plotting.plot_generalization(Rs, acc, tp, fp, ax=ax, label='Multiple repeats' if multiRep else 'One repeat')
    #    ax[0].set_title(label)
        format_and_save(ax[0].get_figure(), 'idealized/{}_multirep_supp.pdf'.format(label), w=58, h=70)

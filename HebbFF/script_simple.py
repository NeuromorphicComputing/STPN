import argparse
import os.path
import pdb

import torch
import networks as nets
from data import generate_recog_data, generate_recog_data_batch
from plotting import plot_generalization, get_recog_positive_rates

from STPN.Scripts.utils import DATA, RESULTS

def main(args):
    assert args.epochs == 'inf' or isinstance(int(args.epochs), int)
    args.epochs = int(args.epochs) if isinstance(args.epochs, int) else float(args.epochs)
    device = torch.device(f'cuda:{args.gpu}') if (args.gpu > -1 and torch.cuda.is_available()) else torch.device('cpu')

    # initialize net
    if args.netType == 'nnLSTM':
        net = nets.nnLSTM([args.d, args.N, 1])
    elif args.netType == 'HebbNet':
        net = nets.HebbNet([args.d, args.N, 1])
        # constrain Lambda
        if args.lam == 'reparamLam':
            net.reparamLam = True
        elif args.lam == 'reparamLamTau':
            net.reparamLamTau = True
        # constrain Gamma
        if args.force == 'Hebb':
            net.forceHebb = torch.tensor(True)
            net.init_hebb(eta=net.eta.item(), lam=net.lam.item())  # need to re-init for this to work
        elif args.force == 'Anti':
            net.forceAnti = torch.tensor(True)
            net.init_hebb(eta=net.eta.item(), lam=net.lam.item())
        elif args.force is not None:
            raise ValueError
    elif args.netType == 'uSTPNfNet':
        net = nets.uSTPNfNet([args.d, args.N, 1])
    elif args.netType == 'STPNfNet':
        net = nets.STPNfNet([args.d, args.N, 1])
    elif args.netType == 'uSTPNrNet':
        net = nets.uSTPNrNet([args.d, args.N, 1])
    elif args.netType == 'STPNrNet':
        net = nets.STPNrNet([args.d, args.N, 1])
    else:
        raise ValueError

    net.to(device)
    # train
    filename = f'{args.netType}[{args.d},{args.N},1]_train={args.trainMode}{args.R}'
    # TODO store T also for inf
    if args.trainMode == 'dat':
        filename += f'_T={args.T}'
    if args.force is not None:
        filename += f'_force{args.force}'

    filename += '.pkl'
    # train with same dataset for all epochs
    if args.trainMode == 'dat':
        trainData = generate_recog_data_batch(T=args.T, d=args.d, R=args.R, P=0.5, multiRep=False)  # noqa
        validBatch = generate_recog_data(T=args.T, d=args.d, R=args.R, P=0.5, multiRep=False).tensors  # noqa
        trainData.tensors = tuple(t.to(device) for t in trainData.tensors)
        validBatch = tuple(t.to(device) for t in validBatch)  # noqa
        net.fit('dataset', epochs=args.epochs, earlyStop=args.earlyStop, trainData=trainData, validBatch=validBatch,
                filename=os.path.expanduser(os.path.join(RESULTS, f'HebbFF/{filename}')))
    # generate a new dataset every epoch
    elif args.trainMode == 'inf':
        gen_data = lambda: generate_recog_data_batch(  # noqa
            T=args.T, d=args.d, R=args.R, P=0.5, multiRep=False, device=device)
        net.fit('infinite', gen_data, iters=args.epochs, earlyStop=args.earlyStop,
                filename=os.path.expanduser(os.path.join(RESULTS, f'HebbFF/{filename}')))
    else:
        raise ValueError

    # optional save
    if args.skipSave is False:
        fname = '{}[{},{},1]_{}train={}{}_{}.pkl'.format(
            args.netType, args.d, args.N, 'force{}_'.format(args.force) if args.force else '',
            args.trainMode, args.R, 'T={}'.format(args.T) if args.trainMode != 'cur' else ''
        )
        fname = os.path.expanduser(os.path.join(RESULTS, '/HebbFF/optional_save/', fname))
        net.save(fname)

    # plot generalization
    gen_data = lambda R: generate_recog_data_batch(T=args.T, d=args.d, R=R, P=0.5, multiRep=False)  # noqa
    testR, testAcc, truePosRate, falsePosRate = get_recog_positive_rates(net, gen_data)  # noqa
    testAcc = [t.cpu() for t in testAcc]  # noqa
    plot_generalization_ax = plot_generalization(testR, testAcc, truePosRate, falsePosRate)
    plot_generalization_ax.savefig(os.path.expanduser(os.path.join(RESULTS, f'/HebbFF/{filename}.png')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # network type and characteristics
    parser.add_argument('--netType', type=str,
                        choices=['uSTPNrNet', 'STPNfNet', 'STPNrNet', 'uSTPNfNet', 'HebbNet', 'nnLSTM'])
    parser.add_argument('--N', type=int, default=100,
                        help='Hidden size of net. Only used if ovverrideN is passed (as true)')
    parser.add_argument('--d', type=int, default=100, help='Input size of data to be generated.')
    # experiment characteristics
    parser.add_argument('--T', type=int, default=5000, help='Length of dataset (number of examples)')
    parser.add_argument('--R', type=int, choices=[3, 6], help='Distance between  associations')
    parser.add_argument('--trainMode', type=str, choices=['inf', 'dat'])
    parser.add_argument('--epochs', help='Number of epochs or <<inf>> if using earlyStop')
    parser.add_argument('--earlyStop', default=False, action='store_true', help='Stop training early')

    # technical choices about how to run experiment
    parser.add_argument('--skipSave', default=False, action='store_true', help='Save checkpoint at end of training')
    parser.add_argument('--gpu', type=int, help='GPU id', default=-1)

    # HebbNet specifics
    parser.add_argument('--force', type=str, default=None, help='Ensure Hebbian or antiHebbian plasticity behaviour',
                        choices=['Anti', 'Hebb'])
    parser.add_argument('--lam', type=str, default=None, help='Enforce a range on lambda',
                        choices=['reparamLam', 'reparamLamTau'])

    parsed_args = parser.parse_args()
    main(parsed_args)

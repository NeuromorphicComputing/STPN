"""
Grid maze modified from: Backpropamine: differentiable neuromdulated plasticity,
Miconi et al. ICML 2018 ( https://arxiv.org/abs/1804.02464 )
License from original source:
        Copyright (c) 2018-2019 Uber Technologies, Inc.

        Licensed under the Uber Non-Commercial License (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at the root directory of this project.

        See the License file in this repository for the specific language governing
        permissions and limitations under the License.

Changes include
  - additional evaluation metrics (eg. energy),
  - evaluation of trained agents,
  - new networks (random policy, omniscient agent)
  - parallelisation of transition function & input generation, which speeds up significantly
  - checkpointing and resuming of training
  - additional flags to use our STPN model
"""

import argparse
import copy
import json

import torch
from torch.autograd import Variable
import torch.nn.functional as F  # noqa
import random
import pickle
import time
import os
import platform
import numpy as np

from STPN.Maze.Scripts.maze_nets import STPNNetwork, RandomPolicyNet, MiconiNetwork
from STPN.Scripts.utils import DATA, RESULTS, PROJECT

# TODO: make these not hardcoded in the script
NBDA = 1  # Number of different DA output neurons. At present, code assumes NBDA=1 and will NOT WORK if you change this

np.set_printoptions(precision=4)

ADDINPUT = 4  # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs

NBACTIONS = 4  # Up, Down, Left, Right

RFSIZE = 3  # Receptive Field

TOTALNBINPUTS = RFSIZE * RFSIZE + ADDINPUT + NBACTIONS

USE_PARALLEL_LABS = False  # [False, True] # create batch_size number of labs for parallelisation. True is not supported
EPS_DISTANCE = 1e-5  # distance to reward


def train(paramdict):
    print("Starting training...")
    params = {}
    params.update(paramdict)
    print("Passed params: ", params)
    print(platform.uname())
    # TODO: make this flags to ignore a global variable
    # do not include unimportant flags in results filename
    flags_to_ignore_in_filename = [
        'nbsteps', 'rew', 'wp', 'bent', 'blossv', 'msize', 'da', 'gr', 'gc', 'rsp', 'addpw', 'l2', 'save_every', 'pe',
        'gpu', 'config_file_path', 'path_experiment_results', 'eval_energy', 'base_path_experiments',
        'config_file_train',
        'clamp', 'extra_config', 'wN', 'clamp', 'run_id',
        'stp', 'energy', 'device',  # these are added later by loading
    ]
    # Turning the parameters into a nice suffix for filenames
    suffix = "".join([str(x) + "_" if pair[0] not in flags_to_ignore_in_filename else '' for pair in
                      sorted(zip(params.keys(), params.values()), key=lambda x: x[0]) for x in pair])[:-1]
    if isinstance(params.get('config_file_path'), str):
        suffix += "_config_" + os.path.basename(params['config_file_path']).replace('.json', '')
    if params['gpu'] > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{params['gpu']}")
    else:
        device = torch.device(f"cpu")
    params['device'] = device

    # save id for this run, we add id if it's >0 for backwards compatibility
    sv_id = '' if params['run_id'] == 0 else params['run_id']
    # therefore we add id for loading only if this is run >1
    # for continuing training, we load the previous run, not so for evaluation
    if params['eval'] is True:
        ld_id = sv_id
    else:
        # load id for previous run, different if retraining
        ld_id = '' if params['run_id'] in [0, 1] else params['run_id'] - 1

    # Initialize random seeds
    print("Setting random seeds")
    np.random.seed(params['rngseed'])
    random.seed(params['rngseed'])
    torch.manual_seed(params['rngseed'])

    # load config file parameters
    if isinstance(params.get('config_file_path'), str):
        with open(params['config_file_path']) as config_file:
            config = json.load(config_file)
        params['energy'] = config.get('energy', {})
        params['wN'] = config.get('wN', None)
        if params['eval'] is True and params['net_type'] != 'random':
            with open(params.get('config_file_train')) as config_file:
                config_trained = json.load(config_file)
            params['stp'] = config_trained.get('stp', {})
            params['extra_config'] = config_trained.get('extra_config', {})
        else:
            params['stp'] = config.get('stp', {})
            params['extra_config'] = config.get('extra_config', {})
    else:
        params['stp'] = {}
        params['energy'] = {}
        params['extra_config'] = {}
        params['wN'] = None

    print("Initializing network")
    if params['net_type'] == 'miconi':
        net = MiconiNetwork(params)  # already in params['device']
    elif params['net_type'] == 'stpn':
        net = STPNNetwork(params).to(params['device'])
    elif params['net_type'] == 'random':
        net = RandomPolicyNet(params)
    else:
        raise Exception(f"not valid 'net_type' {params['net_type']}")

    # load previouly trained model: 1) for evaluation 2) to resume training
    if (params['eval'] is True or params['run_id'] > 0) and params['net_type'] != 'random':
        train_params = copy.deepcopy(params)
        if params['eval'] is True:
            for param, value in config.get('params_override_eval', {}).items():
                # pdb.set_trace()
                train_params[param] = value

        train_suffix = "".join([str(x) + "_" if pair[0] not in flags_to_ignore_in_filename else '' for pair in
                                sorted(zip(train_params.keys(), train_params.values()), key=lambda x: x[0]) for x in
                                pair])[
                       :-1]  # Turning the parameters into a nice suffix for filenames
        if isinstance(params.get('config_file_train', None), str):
            train_suffix += "_config_" + os.path.basename(params['config_file_train']).replace('.json', '')
        else:
            print("WARNING: you are loading the same configuration for previous training and current "
                  "evaluation/continued training. If this is not the intended behaviour, provide path to training "
                  "config file in config_file_train")
            train_suffix += "_config_" + os.path.basename(params['config_file_path']).replace('.json', '')

        net.load_state_dict(torch.load(
            os.path.join(params['path_experiment_results'], "logs",
                         f'torchmodel{ld_id}_' + train_suffix + '.dat').replace('_eval_True_', '_eval_False_'),
            map_location = params['device'],
        ))

    print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print("Size (numel) of all optimized elements:", allsizes)
    print("Total size (numel) of all optimized elements:", sum(allsizes))

    print("Initializing optimizer")
    if params['eval'] is False:
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'], eps=1e-4, weight_decay=params['l2'])
        if params['run_id'] > 0:
            optimizer.load_state_dict(torch.load(os.path.join(
                params['path_experiment_results'], "logs", f'torchopt{ld_id}_' + train_suffix + '.pt'),  # noqa: F823
                map_location = params['device']
            ))

    BATCHSIZE = params['bs']  # noqa
    LABSIZE = params['msize']  # default 9 # noqa
    lab = np.ones((LABSIZE, LABSIZE), dtype=int)
    CTR = LABSIZE // 2  # default (4,4) center # noqa

    validrposr, validrposc = [], []

    # Grid maze
    lab[1:LABSIZE - 1, 1:LABSIZE - 1].fill(0)
    for row in range(1, LABSIZE - 1):
        for col in range(1, LABSIZE - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
            else:
                validrposr.append(row)
                validrposc.append(col)
    # Not strictly necessary, but cleaner since we start the agent at the
    # center for each episode; may help loclization in some maze sizes
    # (including 13 and 9, but not 11) by introducing a detectable irregularity
    # in the center:
    lab[CTR, CTR] = 0
    validrposr.append(CTR)
    validrposc.append(CTR)
    numvalidrpos = len(validrposr)
    numvalidpos = numvalidrpos - 1  # one less available position for agent as we don't let it land on the reward
    validrposr = np.array(validrposr)
    validrposc = np.array(validrposc)

    if USE_PARALLEL_LABS is True:
        raise NotImplementedError

    all_grad_norms = []
    all_losses_objective = []
    all_total_rewards = []
    lossbetweensaves = 0
    nowtime = time.time()

    print("Total number of parameters:", sum([x.numel() for x in net.parameters()]))
    print("Starting episodes!")

    all_iters_energy = []
    if params['run_id'] == 0 or params['eval'] is True:
        start_iter = 0
    else:
        with open(
                os.path.join(params['path_experiment_results'], "logs", f'numiter{ld_id}_' + suffix + '.txt'), 'r'
        ) as thefile:
            f = thefile.read()
            start_iter = int(f)
    # =======================================================
    # =============== Start iterations ======================
    # =======================================================
    for numiter in range(start_iter, params['nbiter']):
        PRINTTRACE = 0  # noqa
        if (numiter + 1) % (params['pe']) == 0:
            PRINTTRACE = 1  # noqa

        # Select the reward location for this episode - not on a wall!
        # And not on the center either! (though not sure how useful that restriction is...)
        # We always start the episode from the center
        # (when hitting reward, we may teleport either to center or to a random location depending on params['rsp'])
        # posr and rposr are the y coordinaate, posc and rposc are the x coordinate
        # starting from the top left corner of the maze
        # Agent always starts an episode from the center
        posr = np.ones(BATCHSIZE, dtype=int) * CTR
        posc = np.ones(BATCHSIZE, dtype=int) * CTR

        # draw id of position for reward in this episode
        id_validrpos = np.random.randint(low=0, high=numvalidrpos, size=BATCHSIZE)
        # get actual position associated with position id from pre-calulated available positions
        rposr, rposc = validrposr[id_validrpos], validrposc[id_validrpos]

        # valid posr is diff for all parallel agents as the reward is located in a different place
        # recalulate the valid positions for each parallel agent in this episode,
        # by removing the reward location from available positions
        # (reason for doing this is fidelity to original implementation)
        validposr = np.repeat(validrposr[np.newaxis, :], BATCHSIZE, axis=0)[
            np.arange(numvalidrpos) != id_validrpos[:, None]
            ].reshape(BATCHSIZE, -1)
        validposc = np.repeat(validrposc[np.newaxis, :], BATCHSIZE, axis=0)[
            np.arange(numvalidrpos) != id_validrpos[:, None]
            ].reshape(BATCHSIZE, -1)

        if params['eval'] is False:
            optimizer.zero_grad()  # noqa: F823
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()
        et = net.initialZeroHebb()  # Eligibility Trace is identical to Hebbian Trace in shape
        pw = net.initialZeroPlasticWeights()

        reward = np.zeros(BATCHSIZE)
        sumreward = np.zeros(BATCHSIZE)
        rewards = []
        vs = []
        logprobs = []
        dist = 0
        numactionschosen = np.zeros(BATCHSIZE, dtype='int32')

        this_ep_energy, energy = [], None
        # =======================================================
        # =============== Start episode =========================
        # =======================================================
        for numstep in range(params['eplen']):
            inputs = np.zeros((BATCHSIZE, TOTALNBINPUTS), dtype='float32')
            labg = lab.copy()
            # field of view of agent (wall or not)
            inputs[:, 0:RFSIZE * RFSIZE] = np.concatenate([labg[
                                                           posr[nb] - RFSIZE // 2:posr[nb] + RFSIZE // 2 + 1,
                                                           posc[nb] - RFSIZE // 2:posc[nb] + RFSIZE // 2 + 1
                                                           ].flatten() for nb in range(BATCHSIZE)
                                                           ]).reshape(BATCHSIZE, -1) * 1.0

            inputs[:, RFSIZE * RFSIZE + 1] = 1.0  # Bias neuron
            inputs[:, RFSIZE * RFSIZE + 2] = numstep / params['eplen']  # percetange of episode at which it is
            inputs[:, RFSIZE * RFSIZE + 3] = 1.0 * reward  # previous reward
            # whether each action was chosen in previous step
            for nb in range(BATCHSIZE):  # no vectorisation for this one
                inputs[nb, RFSIZE * RFSIZE + ADDINPUT + numactionschosen[nb]] = 1

            inputsC = torch.from_numpy(inputs).to(params['device'])  # noqa

            # fwd pass
            if params['eval_energy'] is True:
                # y  should output raw scores, not probas
                y, v, hidden, hebb, et, pw, energy = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)
            else:
                y, v, hidden, hebb, et, pw = net(Variable(inputsC, requires_grad=False), hidden, hebb, et, pw)

            if energy is not None:
                # average along (batch, hidden), unsqueze episode id at 0
                this_ep_energy.append(energy.mean(dim=(0, 1)).cpu().unsqueeze(0))

            y = F.softmax(y, dim=1)  # Now y is conveted to "proba-like" quantities
            distrib = torch.distributions.Categorical(y)
            if params['net_type'] == 'random':
                numactionschosen = np.random.choice([0, 1, 2, 3], size=BATCHSIZE)
            else:
                actionschosen = distrib.sample()
                logprobs.append(distrib.log_prob(actionschosen))
                numactionschosen = actionschosen.data.cpu().numpy()  # Turn to scalar

            reward = np.zeros(BATCHSIZE, dtype='float32')

            transition_posc = np.array([0, 0, -1, 1])
            transition_posr = np.array([-1, 1, 0, 0])
            # target position = current postion + transiton(action taken)
            tgtposc = posc + transition_posc[numactionschosen]
            tgtposr = posr + transition_posr[numactionschosen]
            id_wall_agents = lab[tgtposr, tgtposc]  # batch id of agents that transitioned to wall
            # negative reward for agents that hit wall
            if params['wp'] != 0:
                reward -= id_wall_agents * params['wp']
            # new position is same position for agents that hit the wall (can't transition to wall)
            # or target position for those agents that performed valid transitions
            posr = id_wall_agents * posr + (1 - id_wall_agents) * tgtposr
            posc = id_wall_agents * posc + (1 - id_wall_agents) * tgtposc

            # distances from goal (reward) to position for each parallel agent
            rew_distr = rposr - posr
            rew_distc = rposc - posc
            id_rewarded_agents = np.where((rew_distr * rew_distr + rew_distc * rew_distc) < EPS_DISTANCE)[0]
            # add reward to agents that hit the goal
            reward[id_rewarded_agents] += params['rew']
            # draw a position id out of the number of available positions (calculated when generating maze)
            # = non-wall positions - 1 (due to reward location)
            id_validpos = np.random.randint(low=0, high=numvalidpos, size=len(id_rewarded_agents))
            # get the actual positions from the validpositions for each coordinate
            # (which are different  for each batch due to reward location being different)
            # Not parallel, but we about re-sampling in case there is a wall ...
            posr[id_rewarded_agents] = np.array(
                [validposr[nb, id_validpos[nb]] for nb in range(len(id_rewarded_agents))])
            posc[id_rewarded_agents] = np.array(
                [validposc[nb, id_validpos[nb]] for nb in range(len(id_rewarded_agents))])

            # episode step stats
            rewards.append(reward)
            vs.append(v)
            sumreward += reward

            if params['eval'] is False:
                # This is the "entropy bonus" of A2C, except that since our version
                # of PyTorch doesn't have an entropy() function, we implement it as
                # a penalty on the sum of squares instead. The effect is the same:
                # we want to penalize concentration of probabilities, i.e.
                # encourage diversity of actions.
                loss += (params['bent'] * y.pow(2).sum() / BATCHSIZE)

            if PRINTTRACE:
                print("Step ", numstep,
                      # " Inputs (to 1st in batch): ", inputs[0, :TOTALNBINPUTS],
                      # " - Outputs(1st in batch): ", y[0].data.cpu().numpy(), " - action chosen(1st in batch): ",
                      # numactionschosen[0],
                      # " - mean abs pw: ", np.mean(np.abs(pw.data.cpu().numpy())),
                      " -Reward (this step, 1st in batch): ", reward[0])

        # Episode stats
        if energy is not None:
            all_iters_energy.append(torch.cat(this_ep_energy, dim=0).unsqueeze(0))

        # Episode gradient update
        if params['eval'] is False:  # I think no need for return (or loss) in eval case
            R = Variable(torch.zeros(BATCHSIZE).to(params['device']), requires_grad=False)  # noqa
            gammaR = params['gr']  # noqa
            for numstepb in reversed(range(params['eplen'])):
                R = gammaR * R + Variable(  # noqa
                    torch.from_numpy(rewards[numstepb]).to(params['device']), requires_grad=False)
                ctrR = R - vs[numstepb][0]  # noqa
                lossv += ctrR.pow(2).sum() / BATCHSIZE
                loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BATCHSIZE  # Need to check if detach() is OK

            loss += params['blossv'] * lossv
            loss /= params['eplen']

            if PRINTTRACE:
                if True:
                    print("lossv: ", float(lossv))

            loss.backward()  # noqa
            all_grad_norms.append(torch.nn.utils.clip_grad_norm(net.parameters(), params['gc']))
            if numiter > 100:  # Burn-in period for meanrewards
                optimizer.step()

            lossnum = float(loss)
            lossbetweensaves += lossnum
            all_losses_objective.append(lossnum)
        all_total_rewards.append(sumreward.mean())  # noqa: F823

        if PRINTTRACE:  # noqa: F823
            print("Total reward for this episode (all in batch):", sumreward, "Dist:", dist)  # noqa: F823

        if (numiter + 1) % params['pe'] == 0:  # noqa: F823
            print("numeiter", numiter, "=" * 30)
            if params['eval'] is False:
                print("Mean loss: ", lossbetweensaves / params['pe'])
            lossbetweensaves = 0
            print("Mean reward (across batch and last", params['pe'], "eps.): ",
                  np.sum(all_total_rewards[-params['pe']:]) / params['pe'])
            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)
            if params['type'] == 'plastic' or params['type'] == 'lstmplastic':
                print("ETA: ", net.eta.data.cpu().numpy(), "alpha[0,1]: ", net.alpha.data.cpu().numpy()[0, 1],
                      "w[0,1]: ", net.w.data.cpu().numpy()[0, 1])
            elif params['type'] == 'modul':
                print("etaet: ", float(net.etaet), " mean-abs pw: ", torch.mean(torch.abs(pw.data)))  # noqa: F823
            elif params['type'] == 'rnn':
                print("w[0,1]: ", net.w.data.cpu().numpy()[0, 1])

        if (numiter + 1) % params['save_every'] == 0:
            if not os.path.isdir(os.path.join(params['path_experiment_results'], "logs")):
                os.makedirs(os.path.join(params['path_experiment_results'], "logs"))
            print("Saving files...")
            if params['eval'] is False:
                losslast100 = np.mean(all_losses_objective[-100:])
                print("Average loss over the last 100 episodes:", losslast100)
            print("Saving local files...")
            # save loss (reward) and params (training config) for for eval and train
            with open(
                    os.path.join(params['path_experiment_results'], "logs", f'loss{sv_id}_' + suffix + '.txt'), 'w'
            ) as thefile:
                # TODO: It doesn't average ! Just takes every 10.
                #  Lots of info thrown away, but keeping it this way for consistent comparison puposes
                for item in all_total_rewards[::10]:
                    thefile.write("%s\n" % item)
            with open(
                    os.path.join(params['path_experiment_results'], "logs", f'params{sv_id}_' + suffix + '.dat'), 'wb'
            ) as fo:
                pickle.dump(params, fo)

            # save grad norms only during training
            if params['eval'] is False:
                # grad norms for learning understanding purposes
                with open(
                        os.path.join(params['path_experiment_results'], "logs", f'grad{sv_id}_' + suffix + '.txt'), 'w'
                ) as thefile:
                    for item in all_grad_norms[::10]:
                        thefile.write("%s\n" % item)

                # checkpoint model
                torch.save(
                    net.state_dict(),
                    os.path.join(params['path_experiment_results'], "logs", f'torchmodel{sv_id}_' + suffix + '.dat')
                )

                # checkpoint optimizer
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(params['path_experiment_results'], "logs", f'torchopt{sv_id}_' + suffix + '.pt')
                )
                with open(
                        os.path.join(params['path_experiment_results'], "logs", f'numiter{sv_id}_' + suffix + '.txt'),
                        'w'
                ) as thefile:
                    thefile.write(f'{numiter}')

                print("Done!")

            # save energy
            if params['eval_energy'] is True:
                if not os.path.isdir(os.path.join(params['path_experiment_results'], "efficiency")):
                    os.makedirs(os.path.join(params['path_experiment_results'], "efficiency"))
                with open(
                        os.path.join(
                            params['path_experiment_results'], "efficiency",
                            f'energy{sv_id}_' + suffix + '.npy'
                        ),
                        'wb') as energy_results_file:
                    energy_to_save = torch.cat(all_iters_energy, dim=0).cpu()
                    print("mean energy consumption", energy_to_save.mean())
                    np.save(energy_results_file, energy_to_save)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value (reward increment for taking correct action after correct stimulus)",
                        default=1.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.05)
    parser.add_argument("--bent", type=float,
                        help="coefficient for the entropy reward (really Simpson index concentration measure)",
                        default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--type", help="network type ('lstm' or 'rnn' or 'plastic')", default='modul')
    parser.add_argument("--msize", type=int, help="size of the maze; must be odd", default=9)
    parser.add_argument("--da", help="transformation function of DA signal (tanh or sig or lin)", default='tanh')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--gc", type=float, help="gradient norm clipping", default=1000.0)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--rsp", type=int,
                        help="does the agent start each episode from random position (1) or center (0) ?", default=1)
    parser.add_argument("--addpw", type=int, help="are plastic weights purely additive (1) or forgetting (0) ?",
                        default=1)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--bs", type=int, help="batch size", default=1)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=3e-6)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=1000)
    parser.add_argument("--pe", type=int, help="number of cycles between successive printing of information",
                        default=100)
    parser.add_argument("--gpu", type=int, help="id of gpu used for gradient backpropagation", default=0)
    parser.add_argument("--net_type", help="Network type: 'miconi' or 'stpn'", default='miconi')
    # parser.add_argument("--stp_type", help="STP type: 'input', 'all', or 'recurrent", default='input')
    parser.add_argument("--eval-energy", action='store_true', help="Evaluate energy consumption during training",
                        default=False)
    parser.add_argument("--eval", default=False, action='store_true',
                        help="If passed, no training will be done, only evaluation of a trained model")
    parser.add_argument("--base-path-experiments", type=str, help="Absolute path to experiments",
                        default=os.path.join(PROJECT, "Maze"))
    parser.add_argument("--path-experiment-results", type=str,
                        help="Path to store results. Absolute path or relative to --base-path-experiments",
                        default=os.path.join(RESULTS, "Maze"))
    parser.add_argument("--config_file_path", type=str,
                        help="Relative path with config file for running train or eval (containing at least stp config "
                             "for training).",
                        default=None)
    parser.add_argument('--config_file_train', type=str,
                        help='If running evaluation (--eval flag), path of the config used to train model')
    # Deprecated. Support of weight normalisation for non-STPN networks.
    # parser.add_argument('--wN', default=False, action='store_true',
    #                     help='Use weight normalisation (RNN, Modpast)')
    parser.add_argument("--clamp",
                        help="clamping value of plastic weights for Miconi Networks (for STPN, do so in config)",
                        default=1.0)
    parser.add_argument("--run-id", type=int, default=False,
                        help="Index of run. If first type running this configuration, use default 0. "
                             "If resuming from a checkpoint, use id according to the number of previous runs with the "
                             "same config. Eg. if there are checkpoints from two"
                             " runs that where previously stopped, pass --run-id 2")

    args = parser.parse_args()
    argvars = vars(args)
    argdict = {k: argvars[k] for k in argvars if argvars[k] is not None}
    argdict["path_experiment_results"] = os.path.join(args.base_path_experiments, args.path_experiment_results)
    if args.config_file_path is not None:
        argdict["config_file_path"] = os.path.expanduser(
            os.path.join(args.base_path_experiments, args.config_file_path))

    if args.config_file_train is not None:
        argdict["config_file_train"] = os.path.expanduser(
            os.path.join(args.base_path_experiments, args.config_file_train))

    train(argdict)

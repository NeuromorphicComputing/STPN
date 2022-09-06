import argparse
import json
import os

import numpy as np
import torch

from STPN.Scripts.STPLayers import (
    STPNr, CustomLSTMCell, FastWeights, CustomRNNCell, STPNF, CustomMiconiNetwork, HebbFF
)
from STPN.Scripts.Nets import NoEmbed_AssociativeNet
from STPN.Scripts.eval_utils import simple_eval
from STPN.Scripts.train_utils import train, accuracy_evaluation
from STPN.Scripts import utils
from STPN.AssociativeRetrievalTask.data_art import load_data
from STPN.Scripts.utils import DATA, RESULTS


def main(config):
    utils.set_seed(config['seed'])
    # merge given cfg with default, overwriting default with given
    assert isinstance(config['use_embedding'], bool)
    train_dataloader, validation_dataloader, test_dataloader = load_data(
        batch_size=config['data']['batch_size'],
        data_path=config['data']['data_path'],
        onehot=not config['use_embedding'],
        train_size=config['data']['train_size'],  # default 100000,
        valid_size=config['data']['valid_size'],  # default 10000,
        test_size=config['data']['test_size'],  # default 20000
    )

    # rename config args
    config['train_args'] = config['training']
    config['train_args']['device'] = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1\
        else torch.device('cpu')

    # instantiate model
    if config['use_embedding'] is True:
        raise NotImplementedError
    else:
        if config['model']['recurrent_unit'] == 'STPNr':
            recurrent_unit = STPNr
        elif config['model']['recurrent_unit'] == 'LSTM':
            recurrent_unit = CustomLSTMCell
        elif config['model']['recurrent_unit'] == 'RNN':
            recurrent_unit = CustomRNNCell
        elif config['model']['recurrent_unit'] == 'STPNf':
            recurrent_unit = STPNF
        elif config['model']['recurrent_unit'] == 'HebbFF':
            recurrent_unit = HebbFF
        elif config['model']['recurrent_unit'] == 'FastWeights':
            recurrent_unit = FastWeights
        elif config['model']['recurrent_unit'] == 'MODPLAST':
            # The configuration is: net_type = 'modplast', da = 'tanh', addpw = 3, clamp_val = 1.0, NBDA = 1
            recurrent_unit = CustomMiconiNetwork
            config['model']['rnn_args']['device'] = config['train_args']['device']
        else:
            raise Exception(f"Reccurent unit of type {config['model']['recurrent_unit']} not supported")

        net = NoEmbed_AssociativeNet(
            dictionary_size=config['data']['dict_size'],
            hidden_dim=config['model']['hidden_size'],
            output_size=config['data']['dict_size'],
            recurrent_unit=recurrent_unit,
            rnn_args=config['model']['rnn_args'],
            rnn_activation=config['model']['activation'],
        )

    # reorganise more config args
    config['data_args'] = config['data']
    config['data_args']['train_dataloader'] = train_dataloader
    config['data_args']['output_size'] = config['data']['dict_size']

    convergence_args = config['training']['convergence_args']
    convergence_args["convergence_evaluation"] = accuracy_evaluation
    convergence_args["validation_dataloader"] = validation_dataloader

    stats_args = config['stats']
    stats_args['model_name'] = f"ART_from_{os.path.basename(config['config_file_path']).replace('.json', '')}" \
                               f"_seed_{config['seed']}"
    stats_args['keep_checkpoints'] = config['keep_checkpoints']

    if config['training']['stateful'] is True:
        raise NotImplementedError

    trained_model = None
    if config.get('train', False) is True:
        # train
        trained_model, states, *stats = train(
            net,
            data_args=config['data_args'],
            train_args=config['train_args'],
            convergence_args=convergence_args,
            stats_args=stats_args,
            states=None,
        )

        # store model after training
        if not os.path.isdir(config['train_args']['model_path']):
            os.makedirs(config['train_args']['model_path'])
        torch.save(
            trained_model.state_dict(),
            os.path.join(config['train_args']['model_path'], stats_args['model_name'] + '.pth')
        )

        # store validation results during training if they exists
        # accuracy
        if isinstance(stats, list) and isinstance(stats[0], dict) and 'validation_acc_list' in stats[0]:
            if not os.path.isdir(os.path.join(config['train_args']['results_path'], "proficiency")):
                os.makedirs(os.path.join(config['train_args']['results_path'], "proficiency"))
            with open(
                    os.path.join(config['train_args']['results_path'], 'proficiency',
                                 'val_' + stats_args['model_name'] + '.txt'), 'w'
            ) as results_file:
                results_file.write(f"{stats[0]['validation_acc_list']}")
        # energy consumption
        if isinstance(stats, list) and isinstance(stats[0], dict) and 'validation_energy_list' in stats[0]:
            energy = torch.cat([epoch_seqs.unsqueeze(0) for epoch_seqs in stats[0]['validation_energy_list']], dim=0)
            energy_results_path = os.path.join(config['train_args']['results_path'], "efficiency")
            if not os.path.isdir(energy_results_path):
                os.makedirs(energy_results_path)
            print("Storing energy efficiency results in", energy_results_path)
            with open(os.path.join(energy_results_path, 'val_' + stats_args['model_name'] + '.npy'), 'wb') as e_r_file:
                np.save(e_r_file, energy.cpu())  # type: ignore

    # evaluate model on test set
    if config.get('eval', False) is True:
        # if we haven't trained a model, load it
        if trained_model is None:
            trained_model_state_dict = torch.load(
                os.path.join(config['train_args']['model_path'], stats_args['model_name'] + '.pth'),
                map_location = torch.device(config['train_args']['device']),
            )
            net.load_state_dict(trained_model_state_dict)
            net.to(torch.device(config['train_args']['device']))
        else:
            net = trained_model

        # run evaluation
        results = simple_eval(
            net,
            test_dataloader,
            config['train_args']['device'],
            config['data_args'],
            energy=config['eval_energy']
        )
        print(results)
        # store results
        if config['eval_energy'] is True:
            # store test energy consumption results
            acc, energy = results
            if not os.path.isdir(os.path.join(config['train_args']['results_path'], "efficiency")):
                os.makedirs(os.path.join(config['train_args']['results_path'], "efficiency"))
            with open(
                    os.path.join(
                        config['train_args']['results_path'], "efficiency",
                        'test_' + stats_args['model_name'] + '.npy'
                    ),
                    'wb') as energy_results_file:
                np.save(energy_results_file, energy.cpu())  # type: ignore
        else:
            acc = results
        # store test accuracy results
        if not os.path.isdir(os.path.join(config['train_args']['results_path'], "proficiency")):
            os.makedirs(os.path.join(config['train_args']['results_path'], "proficiency"))
        with open(
                os.path.join(config['train_args']['results_path'], 'proficiency',
                             'test_' + stats_args['model_name'] + '.txt'), 'w'
        ) as results_file:
            results_file.write(f"{acc}")


def generate_config(args):
    argvars = vars(args)
    with open(args.config_file_path) as config_file:
        config = json.load(config_file)
    config = {**config, **argvars}

    # some default dataset parameters and paths
    default_cfg = {
        'use_embedding': False,
        'data': {
            'data_path': os.path.join(DATA, 'ART/datasets'),
            'train_size': 100000,
            'valid_size': 10000,
            'test_size': 20000,
        },
        'training': {
            'results_path': os.path.join(RESULTS, 'AssociativeRetrievalTask'),
            'model_path': os.path.join(RESULTS, 'AssociativeRetrievalTask/models'),
        }
    }

    config = utils.merge(default_cfg, config)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note: further configuration can be specified via .json config file specified by --config_file_path,
    # See files under config/ for some examples
    parser.add_argument("--gpu", type=int, help="id of gpu used for gradient backpropagation", default=0)
    parser.add_argument("--seed", type=int, help="random seed", default=0)
    parser.add_argument("--eval-energy", action='store_true', default=False,
                        help="Evaluate energy consumption during training")
    parser.add_argument("--eval", default=False, action='store_true', help="Evaluate on test set after training")
    parser.add_argument("--train", default=False, action='store_true', help="Train a model")
    parser.add_argument(
        "--config_file_path", type=str, default=None,
        help="Relative path with config file for running train or eval (containing at least stp config for training).",
    )
    parser.add_argument(
        "--keep-checkpoints", default=False, action='store_true',
        help="Keep checkpoints of last and best model. Useful if you expect training to stop, or performance to decay "
             "with further training (as it can keep model with highest validation accuracy"
    )

    parsed_args = parser.parse_args()
    config_from_args = generate_config(parsed_args)
    main(config_from_args)

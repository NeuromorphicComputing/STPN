import matplotlib.pyplot as plt
import numpy as np

from STPN.HebbFF.plotting import load_from_file
from STPN.Scripts.nb_utils import get_single_color_cmap
from typing import Tuple


def plot_loss_acc_comparison(
        nets,  # nested list, each level is one model, each sublevel one net
        chance=None, n_runs_per_net=1,
        plot_metrics='loss,acc', plot_datasets='train,valid', colors=None, cmaps=None, alpha=0.2,
        include_legend=True, title=None, linewidth=None, return_all_results=False,
        subplot_coordinates: Tuple[int, int] = None,
        # if None, use ax. we want to plot with plt because we want to use this plot as a subplot
        external_axs=None,
        model2label=None,
):

    all_results = {}

    plot_metrics = plot_metrics.split(',')
    assert 3 > len(plot_metrics) > 0
    valid_plot_metrics = ['loss', 'acc']
    assert all([plot_metric in valid_plot_metrics for plot_metric in plot_metrics])

    plot_datasets = plot_datasets.split(',')
    assert 3 > len(plot_datasets) > 0
    valid_plot_datasets = ['train', 'valid']
    assert all([plot_dataset in valid_plot_datasets for plot_dataset in plot_datasets])

    all_possible_metric_names = [d + "_" + m for m in valid_plot_metrics for d in valid_plot_datasets]

    n_plots = len(plot_metrics)
    n_datasets = len(plot_datasets)

    if subplot_coordinates is None:
        assert n_plots == 1, "Cannot put multiple plots as subplots yet"  # we could just pass multiple ones in subplot_coordinates as list of tuples

    dataset2label = {'train': 'Train', 'valid': 'Validation'}
    if model2label is None:
        get_model_label = lambda m: m
    else:
        get_model_label = lambda m: model2label[m]
    if n_datasets > 1:
        titles = {'loss': 'Loss', 'acc': 'Accuracy'}
        get_label = lambda d, m: f'{dataset2label[d]} {get_model_label(m)}'
    else:
        titles = {'loss': f'{dataset2label[plot_datasets[0]]} loss',
                  'acc': f'{dataset2label[plot_datasets[0]]} accuracy'}
        get_label = lambda d, m: get_model_label(m)

    if n_plots == 1:
        if subplot_coordinates is None:
            fig, ax = plt.subplots(n_plots, sharex=True, figsize=(16, 8))
    else:
        fig, ax = plt.subplots(n_plots, 1, sharex=True, figsize=(24, 16))
    n_models = int(len(nets) / n_runs_per_net)
    cmap = plt.cm.get_cmap('hsv', n_models + 1)

    for i_net_type, net_type in enumerate(nets):
        print("Processing n", i_net_type, "net")
        metrics_net_type = {m: [] for m in all_possible_metric_names}
        for i_net, net in enumerate(net_type):
            # net name
            model_label = net.split('/')[-1].split('[')[0]
            # load net checkpoint (stores results too)
            if type(net) == str:
                net = load_from_file(net)
            assert net.hist is not None, f"Error: net without history {net_type} {net}"
            assert net.hist is not None, f"net without history {net_type} {net}"
            if 'loss' in plot_metrics:
                if 'train' in plot_datasets:
                    metrics_net_type['train_loss'].append(net.hist['train_loss'])
                if 'valid' in plot_datasets:
                    metrics_net_type['valid_loss'].append(net.hist['valid_loss'])

            if 'acc' in plot_metrics:
                if 'train' in plot_datasets:
                    metrics_net_type['train_acc'].append(net.hist['train_acc'])
                if 'valid' in plot_datasets:
                    metrics_net_type['valid_acc'].append(net.hist['valid_acc'])
        #######################
        # instantiate axis for metrics
        if subplot_coordinates is None:
            current_ax = ax if n_plots == 1 else ax[0]
            first_ax = current_ax
            current_ax_id = 0
        else:
            current_ax = plt
            subplot_coordinates
            this_ax = external_axs[subplot_coordinates[0]][subplot_coordinates[1]]
            plt.sca(this_ax)

        # color for this net type
        if colors is not None:
            model_color = colors[model_label]
        elif cmaps is not None:
            model_color = get_single_color_cmap(cmaps[model_label], net.hist['valid_acc'][-1])
        else:
            model_color = cmap(int(i_net // n_runs_per_net))

        monitor_interval = 10
        iters = np.arange(len(net.hist['train_loss'])) * monitor_interval

        metrics_line, metrics_ci_top, metrics_ci_bot = {}, {}, {}
        #         metrics_net_type = {}
        for metric_name, metric_val in metrics_net_type.items():
            if len(metric_val) == 0:
                continue
            metrics_line[metric_name] = np.mean(np.array(metric_val), axis=0)
            this_std = np.std(np.array(metric_val), axis=0)
            metrics_ci_bot[metric_name] = metrics_line[metric_name] - this_std
            metrics_ci_top[metric_name] = metrics_line[metric_name] + this_std
        if 'loss' in plot_metrics:
            if 'train' in plot_datasets:
                current_ax.plot(iters, metrics_line['train_loss'], label=get_label('train', model_label),
                                color=model_color, linewidth=linewidth)
                plt.fill_between(iters, metrics_ci_bot['train_loss'], metrics_ci_top['train_loss'],
                                 color=model_color, alpha=alpha,
                                 )
            if 'valid' in plot_datasets:
                current_ax.plot(iters, metrics_line['valid_loss'], label=get_label('valid', model_label),
                                color=model_color, linewidth=linewidth)
                plt.fill_between(iters, metrics_ci_bot['valid_loss'], metrics_ci_top['valid_loss'],
                                 color=model_color, alpha=alpha,
                                 )
            if external_axs is not None:
                current_ax.ylabel(titles['loss'])
            else:
                current_ax.set_ylabel(titles['loss'])
                if current_ax_id + 1 < n_plots:
                    current_ax = ax[current_ax_id + 1]
                    current_ax_id += 1

        if 'acc' in plot_metrics:
            if 'train' in plot_datasets:
                current_ax.plot(iters, metrics_line['train_acc'], label=get_label('train', model_label),
                                color=model_color, linewidth=linewidth)
                plt.fill_between(iters, metrics_ci_bot['train_acc'], metrics_ci_top['train_acc'],
                                 color=model_color, alpha=alpha,
                                 )
            if 'valid' in plot_datasets:
                current_ax.plot(iters, metrics_line['valid_acc'], label=get_label('valid', model_label),
                                color=model_color, linewidth=linewidth)
                plt.fill_between(iters, metrics_ci_bot['valid_acc'], metrics_ci_top['valid_acc'],
                                 color=model_color, alpha=alpha,
                                 )
            if chance is not None:
                if np.isscalar(chance):
                    chance = chance * np.ones_like(iters)
                current_ax.plot(iters, chance, 'k:', linewidth=0.75)
            if external_axs is not None:
                current_ax.ylabel(titles['acc'])
                current_ax.ylim(bottom=0.55, top=1.001)
            else:
                current_ax.set_ylabel(titles['acc'])
                current_ax.set_ylim(bottom=0.55, top=1.001)
        if external_axs is not None:
            current_ax.xlabel('Iteration')
        else:
            if current_ax_id + 1 < n_plots:
                current_ax = ax[current_ax_id + 1]
                current_ax_id += 1

            current_ax.set_xlabel('Iteration')
        # Put a legend below current axis
        if include_legend:
            current_ax.legend(
                loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True,
                ncol=1  # 5
            )

        if title is not None:
            if external_axs is not None:
                current_ax.title(title)
            else:
                first_ax.title.set_text(title)
        if return_all_results:
            all_results[model_label] = metrics_net_type

    if external_axs is not None:
        # TODO: change this hotfix for enforcing a number of ticks into an optional feature via argument
        n_ticks = 3
        plt.locator_params(nbins=n_ticks)
        xticks = np.linspace(0, iters[-1], n_ticks, dtype=int)
        xlabels = xticks
        plt.xticks(xticks, xlabels)
        if return_all_results:
            return external_axs, all_results
        return external_axs
    else:
        if return_all_results:
            return ax, all_results
        return ax
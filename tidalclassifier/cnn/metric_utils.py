import os
import json
import warnings
from collections import namedtuple

import matplotlib
# matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from tidalclassifier.cnn import input_utils

def recordMetrics(acc, loss, val_acc, val_loss, instruct):
    warnings.warn('recordMetrics will be deprecated in favour of record_aggregate_metrics or similar!')
    name = instruct['name'] + '_' + str(instruct['run_n'])
    np.savetxt(name + '_acc_results.txt', acc)
    np.savetxt(name + '_loss_results.txt', loss)
    np.savetxt(name + '_val_acc_results.txt', val_acc)
    np.savetxt(name + '_val_loss_results.txt', val_loss)


def record_aggregate_metrics(acc, loss, val_acc, val_loss, instruct):
    """Take metrics lists (over many runs) and save as JSON.
    JSON form is { {run: 0, epoch: [0, 1, 2, ...], loss: [0.5, 1.5, 2.5, ...], ... }, ... }
    Used as final stage of training a CNN, flexibly writing metrics to disk for later analysis
    
    Args:
        acc (np.array): of form (run, train accuracy after n epochs)
        loss (np.array): of form (run, train loss after n epochs)
        val_acc (np.array): of form (run, validation accuracy after n epochs)
        val_loss (np.array): of form (run, validation loss after n epochs)
        instruct (dict): configuration instructions
    """
    loc = get_metrics_loc(instruct)

    try:
        assert len(acc.shape) == 2
    except:
        raise ValueError('Failure: dimensionality of metrics not understood')

    results_list = []
    for run in range(len(acc)):
        run_data = {
            'acc': list(acc[run].astype(float)),
            'loss': list(loss[run].astype(float)),
            'val_acc': list(val_acc[run].astype(float)),
            'val_loss': list(val_loss[run].astype(float)),
            'epoch': list(range(len(acc[run]))),  # danger, implicit assumption of epochs
            'run': int(run)
        }
        results_list.append(run_data)

    with open(loc, 'w') as f:
        json.dump(results_list, f)


def get_metrics_loc(instruct):
    return os.path.join(instruct['model_dir'], 'metrics.json')


def load_metrics_as_table(instruct):
    """Load previously saved CNN metrics into convenient table for analysis
    
    Args:
        instruct (dict): configuration instructions, used for identifying filename to load
    
    Returns:
        pd.DataFrame: flat df with metrics by epoch, distinguished by run
    """

    with open(get_metrics_loc(instruct), 'r') as f:
        metrics = json.load(f)

    results_list = []
    for experiment in metrics:
        results_list.append(pd.DataFrame(data=experiment))  # sets run to same value throughout df

    metric_df = pd.concat(results_list, axis=0)  # combine run blocks into single df

    return metric_df


def plot_aggregate_metrics(metric_df, output_dir):
    """Save figures comparing train/test accuracy and loss by epoch.
    
    Args:
        metric_df (pd.DataFrame): metrics, including 'loss', 'epoch', 'run' etc. columns
        output_dir (str): directory into which to save figures. Will overwrite!
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    figsize = (8, 4)

    plt.clf()
    sns.set_style("whitegrid")
    # sns.set(font_scale=1.5)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize, sharex=True, sharey=True)
    sns.lineplot(x='epoch', y='val_acc', data=metric_df, ax=ax1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1.0])
    ax1.set_title('Validation')

    sns.lineplot(x='epoch', y='acc', data=metric_df, ax=ax2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1.0])
    ax2.set_title('Training')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'acc.png'))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize, sharex=True, sharey=True)
    sns.lineplot(x='epoch', y='val_loss', data=metric_df, ax=ax1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, 1.0])
    ax1.set_title('Validation')

    sns.lineplot(x='epoch', y='loss', data=metric_df, ax=ax2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim([0, 1.0])
    ax2.set_title('Training')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss.png'))
    plt.clf()


def record_test_predictions(model, val_table, instruct, run=None):
    """Save predictions on all validation subjects to disk, for later analysis.
    
    Args:
        model (keras.models.Sequential): trained CNN model
        val_table (pd.DataFrame): catalog of test subjects on which to make predictions
        instruct (dict): configuration instructions
        run (int): Optional. If not None, encode run number in filename.
    """
    prediction_table = val_table.copy()
    images = np.stack(  # stack into batch dimension
        [input_utils.construct_image(row, instruct) for _, row in prediction_table.iterrows()]
    )
    prediction_table['prediction'] = model.predict(images)
    if run is None:
        filename = 'validation_predictions.csv'
    else:
        filename = 'validation_predictions_run_{}.csv'.format(run)
    prediction_table.to_csv(os.path.join(instruct['model_dir'], filename))


def plot_metrics(acc, loss, val_acc, val_loss, instruct, plt):
    warnings.warn('plot_metrics will be deprecated in favour of plot_aggregatre metrics or similar!')
    name = instruct['name'] + '_' + str(instruct['run_n'])

    # print(val_acc)
    # av_acc = np.average(acc, axis=0)
    # av_val_acc = np.average(val_acc, axis=0)
    # av_loss = np.average(loss, axis=0)
    # av_val_loss = np.average(val_loss, axis=0)

    # epochs = np.arange(len(acc[0]), dtype='int')
    # print(epochs)
    #
    # plt.figure(1)
    #
    # plt.subplot(121)
    # plt.plot(epochs, av_val_acc, 'k')
    # plt = add_conf_interval(plt, epochs, val_acc)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0, 1.0])
    # # plt.legend(['Train set', 'Validation set'], loc=0)
    # plt.title('Validation')
    #
    #
    # plt.subplot(122)
    # plt.plot(epochs, av_acc, 'r')
    # plt = add_conf_interval(plt, epochs, acc)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0, 1.0])
    # # plt.legend(['Train set', 'Validation set'], loc=0)
    # plt.title('Training')
    #
    # plt.figure(1).subplots_adjust(left=0.1, right=0.9, wspace=0.25)
    # plt.savefig(name + '_acc.png')
    #
    # plt.figure(2)
    #
    # plt.subplot(121)
    # plt.semilogy(epochs, av_val_loss, 'k')
    # plt = add_conf_interval(plt, epochs, val_loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # # plt.ylim([0, 0.7])
    # # plt.legend(['Train set', 'Validation set'], loc=0)
    # plt.title('Validation')
    # # plt.show()
    #
    #
    # plt.subplot(122)
    # plt.semilogy(epochs, av_loss, 'r')
    # plt = add_conf_interval(plt, epochs, loss)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # # plt.ylim([0, 0.7])
    # # plt.legend(['Train set', 'Validation set'], loc=0)
    # plt.title('Training')
    #
    # plt.figure(2).subplots_adjust(left=0.1, right=0.9, wspace=0.25)
    # plt.savefig(name + '_loss.png')


    plt.clf()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)
    plt.figure(1)
    plt.subplot(121)
    sns.tsplot(val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])
    plt.title('Validation')

    plt.subplot(122)
    sns.tsplot(acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])
    # plt.legend(['Train set', 'Validation set'], loc=0)
    plt.title('Training')

    plt.figure(1).subplots_adjust(left=0.1, right=0.9, wspace=0.25)
    plt.savefig(name + '_acc.png')


    plt.clf()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.5)
    plt.figure(2)
    plt.subplot(121)
    sns.tsplot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    plt.title('Validation')

    plt.subplot(122)
    sns.tsplot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    # plt.legend(['Train set', 'Validation set'], loc=0)
    plt.title('Training')

    plt.figure(1).subplots_adjust(left=0.1, right=0.9, wspace=0.25)
    plt.savefig(name + '_loss.png')

    # plt.show()



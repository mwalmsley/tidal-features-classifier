"""
Use previous output catalogs to investigate correct and incorrect images
"""
import logging
import os
import json

import pandas as pd 
import numpy as np
from astropy.io import fits
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from tidalclassifier.utils import custom_image_utils
from tidalclassifier.cnn.individual_cnn.meta_CNN import define_parameters


def log_loss(true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def absolute_error(true_label, predicted):
    return np.abs(true_label - predicted)


def visualise_error_by_class(df, save_loc):
    """Show performance on each pure tidal class based on predictions
    
    Args:
        df (pd.DataFrame): rows of galaxies, cols of picture_id, FEAT, true_label, and prediction
        save_loc (str): path to save performance figure
    """
    for n in range(len(df)):
        df.at[n, 'log_loss'] = log_loss(df.iloc[n]['true_label'], df.iloc[n]['prediction'])
        df.at[n, 'abs_error'] = absolute_error(df.iloc[n]['true_label'], df.iloc[n]['prediction'])

    single_tidal_feature = {'L', 'M', 'A', 'S', 'F'}
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 6))

    ax1 = sns.boxplot(x="FEAT", y="log_loss", data=df[df['FEAT'].isin(single_tidal_feature)], ax=ax1)
    ax1.set_xlabel('Tidal Class')
    ax1.set_ylabel('Log Binary Cross-Entropy')

    ax2 = sns.violinplot(x="FEAT", y="log_loss", data=df[df['FEAT'].isin(single_tidal_feature)], inner="stick", ax=ax2)
    ax2.set_xlabel('Tidal Class')
    ax2.set_ylabel('Log Binary Cross-Entropy')

    ax3 = sns.boxplot(x="FEAT", y="abs_error", data=df[df['FEAT'].isin(single_tidal_feature)], ax=ax3)
    ax3.set_xlabel('Tidal Class')
    ax3.set_ylabel('Absolute Error')
    ax3.set_ylim([0, 1])

    ax4 = sns.violinplot(x="FEAT", y="abs_error", data=df[df['FEAT'].isin(single_tidal_feature)], inner="stick", ax=ax4)
    ax4.set_xlabel('Tidal Class')
    ax4.set_ylabel('Absolute Error')
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_loc)


def get_performance(model_dir):

    with open(os.path.join(model_dir, 'instruct.json'), 'r') as f:
        instruct = json.load(f)

    fits_dir = instruct['directory']

    single_predictions_loc = os.path.join(model_dir, 'validation_predictions.csv')
    if os.path.isfile(single_predictions_loc):
        df = pd.read_csv(single_predictions_loc)
    else:  # assume multiple predictions, by run
        run_dfs = []
        for run in range(instruct['runs']):
            loc = os.path.join(model_dir, 'validation_predictions_run_{}.csv'.format(run))
            run_dfs.append(pd.read_csv(loc))
        df = pd.concat(run_dfs, axis=0).reset_index()
    # if galaxies appear several times, keep only the first measurement
    df = df.drop_duplicates(subset='picture_id', keep='first')

    logging.debug(df.sample(5))
    logging.info('Predictions: {}'.format(len(df)))

    # TODO review this
    df['true_label'] = (df['CONF'] > 2).astype(int)

    logging.info('Mean label: {}'.format(df['true_label'].mean()))

    df['predicted_label'] = (df['prediction'] > 0.5).astype(int)
    correct = df[df['predicted_label'] == df['true_label']]
    incorrect = df[df['predicted_label'] != df['true_label']]
    logging.info('Correct: {}. Incorrect: {}'.format(len(correct), len(incorrect)))
    assert len(correct) > 0
    assert len(incorrect) > 0

    # TODO this code is a bit messy
    correct_fig_dir = os.path.join(model_dir, 'correct')
    incorrect_fig_dir = os.path.join(model_dir, 'incorrect')
    target_dirs = {correct_fig_dir: correct, incorrect_fig_dir: incorrect}

    # instruct, _ = define_parameters(True)  # TODO add as command-line
    max_galaxies = 20
    # gal_file_col = 'threshold_3sig_filename'
    gal_file_col = 'stacked_filename'
    # gal_file_col = 'threshold_bkg_3sig_filename'

    for directory, galaxy_set in target_dirs.items():
        # TODO auto wipe directory if it exists
        for _, galaxy in galaxy_set[:max_galaxies].iterrows():
            fits_loc = os.path.join(fits_dir, galaxy[gal_file_col])
            custom_image_utils.save_png(fits_loc, target_dir=directory, instruct=instruct)

    visualise_error_by_class(df, os.path.join(model_dir, 'error_by_class.png'))


if __name__ == '__main__':
    get_performance('/Data/repos/tidalclassifier/results/cnn_runs/oxford_first_run_80_epochs')

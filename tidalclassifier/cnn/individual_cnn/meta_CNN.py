
"""
Contains core CNN functionality:
Define CNN models (optional defaults)
Train one or many CNN
Provide subjects and labels from saved catalog
Report metrics
etc, etc
"""
import logging
import warnings
import time
import collections
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')
# color channel first

from tidalclassifier.cnn import input_utils, metric_utils
from tidalclassifier.utils.helper_funcs import ThreadsafeIter, shuffle_df, to_json


def create_model(instruct):
    """
    Create a Keras model to train, following the configuration in instruct
        
        Args:
            instruct (dict): configuration for model
        
        Returns:
            (keras.models.Sequential) Trainable sequential CNN model
    """
    channels = instruct['channels']
    img_width = instruct['img_width']
    img_height = instruct['img_height']

    model = Sequential()
    model.add(Conv2D(instruct['layer0_size'], (3, 3), input_shape=(channels, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(instruct['layer1_size'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(instruct['layer2_size'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(instruct['layerFC_size']))
    # is this a good idea? Knocks learning rate WAY down, uncleaar final improvements
    # model.add(ActivityRegularization(l2=0.01))
    if instruct['dropout']:
        model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',  # changed from rmsprop
                  metrics=['accuracy'])
    return model


def cross_validation(model_func, catalog, instruct):
    """Train a CNN on each cross-validation fold, sequentially.

    Args:
        model_func (function): given instruct, return Sequential Keras model to train
        catalog (pd.DataFrame): catalog on which to...
        instruct (dict): configuration instructions
    
    Returns:
        See trainCNNOnTables
    """
    train_tables, val_tables = input_utils.fold_tables(catalog, instruct)
    return trainCNNOnTables(model_func, train_tables, val_tables, instruct)



def crossValidation(model_func, meta, instruct):
    """Manual cross-validation with generators
    for each fold, create train/test generators, fit, store metrics, repeat
    if instruct['which_fold'] is not -1, run only on the specified fold

    Args:
        model_func (function): given instruct, return Sequential Keras model to train
        catalog (pd.DataFrame): catalog on which to...
        instruct (dict): configuration instructions
    
    Returns:
        See trainCNNOnTables
    """
    warnings.warn('crossValidation will be deprecated in favor of cross_validation')
    train_tables, val_tables = input_utils.fold_tables(meta, instruct)

    if instruct['which_fold'] != -1:
        # only run using which_fold
        instruct['folds'] = 1 # to only do a single run, with fold = 0
        train_tables = [train_tables[instruct['which_fold']]] # so that the fold=0 value of train_tables is which_fold
        val_tables = [val_tables[instruct['which_fold']]] # similarly
    instruct['n_of_runs'] = instruct['folds']
    return trainCNNOnTables(model_func, train_tables, val_tables, instruct)


def get_random_folds(catalog, instruct):
    """Get random cross-validation folds of catalog
    TODO should be replaced by simply shuffling and randomly splitting an appropriate fraction
    
    Args:
        catalog (pd.DataFrame): catalog on which to create random train/test splits
        instruct (dict): configuration instructions
    
    Returns:
        list: where nth item is randomly-selected train catalog for nth run
        list: where nth item is randomly-selected test catalog for nth run
    """
    # pick a random fold to crossValidate with, for n runs
    train_tables = []
    val_tables = []
    for run in range(instruct['runs']): 
         # construct the desired train and val table pairs to which to apply train_on_tables
        # shuffle catalog to ensure a unique fold (train/val pair) to train on
        catalog = shuffle_df(catalog)
        # get new unique cross-validation folds
        folded_train_tables, folded_val_tables = input_utils.fold_tables(catalog, instruct)  
        # pick the first (arbitrary) on which to train and test 
        train_table = folded_train_tables[0]
        val_table = folded_val_tables[0]
        train_tables.append(train_table)
        val_tables.append(val_table)
    return train_tables, val_tables


def crossValidationRandom(model_func, meta, instruct):
    """
    Repeatedly train a CNN on random cross-validation folds of meta (catalog)
    Args:
        model_func (function): given instruct, return Sequential Keras model to train
        meta (pd.DataFrame): catalog on which to create random train/test splits
        instruct (dict): configuration instructions

    Returns:
        See trainCNNOnTables
    """
    warnings.warn('crossValidationRandom will be deprecated in favor of direct access')
    train_tables, val_tables = get_random_folds(meta, instruct)
    # having constructed a random set of train and validation tables, 
    # train a CNN (potentially many times) and return metrics
    return trainCNNOnTables(model_func, train_tables, val_tables, instruct)


def trainCNNOnTables(model_func, train_tables, val_tables, instruct):
    """Serially, train a CNN on each train_table and val_table pair.
    For each CNN, record final validation predictions and save as csv
    for each CNN, record the metrics (over time). Save to disk as single JSON
    
    Args:
        model_func ([type]): [description]
        train_tables (list): where nth item is the catalog to train on for run n
        val_tables (list): where nth item is the catalog to test on for run n
        instruct (dict): configuration information
    
    Returns:
        [type]: [description]
    """
    to_json(instruct, os.path.join(instruct['model_dir'], 'instruct.json') )

    # expects exactly one run per train/val catalog pair
    assert len(train_tables) == instruct['runs']
    assert len(val_tables) == instruct['runs']

    # note: not used by metaclassifier, calls trainCNNonTable directly
    runs = instruct['runs']
    nb_epoch = instruct['nb_epoch']

    acc = np.zeros((runs, nb_epoch))
    val_acc = np.zeros((runs, nb_epoch))
    loss = np.zeros((runs, nb_epoch))
    val_loss = np.zeros((runs, nb_epoch))

    model = None # to ensure assignment
    for run in range(runs):
        time.sleep(0.5) # wait half a second to allow any generators to complete
        logging.info('Initialise CNN training run ' + str(run))
        train_table = train_tables[run]
        val_table = val_tables[run]
        r_acc, r_val_acc, r_loss, r_val_loss, model = trainCNNOnTable(model_func, train_table, val_table, run, instruct)

        acc[run, :] = r_acc
        val_acc[run, :] = r_val_acc
        loss[run, :] = r_loss
        val_loss[run, :] = r_val_loss

        metric_utils.record_test_predictions(model, val_table, instruct, run=run)

    metric_utils.record_aggregate_metrics(acc, loss, val_acc, val_loss, instruct)
    return acc, loss, val_acc, val_loss, model


def trainCNNOnTable(model_func, train_table, val_table, run, instruct):
    """Train a CNN on train_table and evaluate on val_table, following `instruct` config.
    Report train and test accuracy/loss metrics over time.

    Provide subjects in even split
    
    Args:
        model_func (function): given 'instruct' dict, returns trainable model
        train_table ([type]): [description]
        val_table ([type]): [description]
        run ([type]): [description]
        instruct ([type]): [description]
    
    Returns:
        [type]: [description]
    """

    # 'run' denotes which set of tables, simply to keep track of order/names

    # set up generators to be used for all models
    custom_gen_train = input_utils.custom_flow_from_directory(train_table, instruct, even_split=True, gen_name='train')
    custom_gen_val = input_utils.custom_flow_from_directory(val_table, instruct, even_split=True, gen_name='val')

    # WARNING
    # custom_gen_train = ThreadsafeIter(custom_gen_train)
    # custom_gen_val = ThreadsafeIter(custom_gen_val)
    assert isinstance(custom_gen_train, collections.Iterable)
    assert isinstance(custom_gen_val, collections.Iterable)

    nb_steps_per_epoch = int(instruct['nb_train_samples'] / instruct['batch_size'])
    nb_epoch = instruct['nb_epoch']
    nb_validation_steps = int(instruct['nb_validation_samples'] / instruct['batch_size'])

    # train CNN
    logging.info('Train subjects: {}'.format(len(train_table)))
    logging.info('Val subjects: {}'.format(len(val_table)))
    logging.info('Mean train label: {}'.format(np.mean(train_table['FEAT']=='N')))
    logging.info('Mean val label: {}'.format(np.mean(val_table['FEAT']=='N')))
    logging.info('steps per epoch: {}'.format(nb_steps_per_epoch))
    logging.info('epochs: {}'.format(nb_epoch))
    logging.info('validation steps: {}'.format(nb_validation_steps))

    # make a new model
    # model = model_func(include_top=False)
    model = model_func(instruct)

    hist = model.fit_generator(
        custom_gen_train,
        steps_per_epoch=nb_steps_per_epoch,
        epochs=nb_epoch,
        validation_data=custom_gen_val,
        validation_steps=nb_validation_steps,
        verbose=2,
    )

    hist_dict = hist.history
    r_acc = np.array([v for v in hist_dict['acc']])
    r_val_acc = np.array([v for v in hist_dict['val_acc']])
    r_loss = np.array([v for v in hist_dict['loss']])
    r_val_loss = np.array([v for v in hist_dict['val_loss']])

    # confusion_matrix_total = np.zeros((2, 2))
    # labels = [0, 1]  # in case some classes have no examples, provide labels to ensure 2d matrix
    # for batch in range(int(instruct['confusion_images'] / instruct['batch_size'])):
        # print('batch', batch)
        # confusion_batch = custom_gen_train.next()  # output is (data, labels)
        # print(confusion_batch.shape)
        # print(confusion_batch)
        # y_true = confusion_batch[1]
        # y_pred = model.predict_classes(confusion_batch[0], batch_size=len(confusion_batch[0]))
        # print('y_true', y_true, 'y_pred', y_pred)
        # confusion_matrix_batch = confusion_matrix(y_true, y_pred, labels=labels)
        # print(confusion_matrix_batch)
        # print(confusion_matrix_batch.shape)
        # print(confusion_matrix_batch).ravel()
        # confusion_matrix_total += confusion_matrix_batch
    # print('tn', 'fp', 'fn', 'tp')
    # print(confusion_matrix_total).ravel()
    # np.savetxt(instruct['name'] + '_conf_matrix_' + str(run) + '.txt', confusion_matrix_total.ravel())

    return r_acc, r_val_acc, r_loss, r_val_loss, model

#
# def metricsToDataframe(acc, loss, val_acc, val_loss, name):
#     # metrics in shape (run, epoch)
#

def benchmarkModel(instruct, meta):
    """Simple benchmark of model"""
    warnings.warn('benchmarkModel will be deprecated!')
    instruct['folds'] = 4
    instruct['which_fold'] = -1
    acc, loss, val_acc, val_loss, model = crossValidation(create_model, meta, instruct)
    metric_utils.plot_metrics(acc,loss,val_acc,val_loss, instruct['name'], plt)


def define_parameters(aws):
    # based on "Building powerful image classification models using very little data" blog, FC, Keras author
    instruct = {}

    instruct['aws'] = aws
    # only include s(h)ells and N
    # meta = meta[(meta.FEAT == 'N') | (meta.FEAT == 'H')]
    # print(meta[0:20])

    instruct['input_mode'] = 'stacked'
    # instruct['input_mode'] = 'color'
    # instruct['input_mode'] = 'masked'
    # instruct['input_mode'] = 'threshold'
    # instruct['input_mode'] = 'threshold_mask'
    # instruct['input_mode'] = 'threshold_bkg'
    # instruct['input_mode'] = 'threshold_5sig'

    instruct['tidal_conf'] = 4
    # instruct['tidal_conf'] = 34
    # instruct['tidal_conf'] = 134

    # training settings
    # instruct['nb_train_samples'] = 750
    # instruct['nb_validation_samples'] = 75
    # instruct['nb_epoch'] = 2
    # instruct['batch_size'] = 75
    # instruct['confusion_images'] = 75
    # boltz mode
    instruct['nb_train_samples'] = 1050
    instruct['nb_validation_samples'] = 300
    instruct['nb_epoch'] = 150
    instruct['batch_size'] = 75
    instruct['confusion_images'] = 300

    # instruct['runs'] = 5

    instruct['layer0_size'] = 32
    instruct['layer1_size'] = 32
    instruct['layer2_size'] = 64
    instruct['layerFC_size'] = 64
    instruct['dropout'] = True

    # image corrections
    instruct['crop'] = True
    instruct['w'] = 128
    instruct['clip'] = None # or 'ceiling' or 'threshold'
    instruct['sig_n'] = 5
    instruct['rel'] = False
    instruct['hist'] = False
    instruct['gamma'] = 1.
    instruct['clip_lim'] = 0.01
    instruct['convolve'] = False
    instruct['scale'] = 'log' # False or 'pow' or 'log'
    instruct['pow_val'] = 1 # power to raise too (should be < 1 for LSB enhancement)
    instruct['multiply'] = 1

    # dimensions
    instruct['channels'] = 1
    instruct['img_width'] = 512
    instruct['img_height'] = 512
    if instruct['crop'] == True: instruct['img_width'] = instruct['w'] * 2
    if instruct['crop'] == True: instruct['img_height'] = instruct['w'] * 2
    instruct['channel_index'] = 0
    instruct['row_index'] = 1
    instruct['col_index'] = 2

    augmentations = True
    if augmentations:
        # augmentations ON
        instruct['rotation_range'] = 90
        instruct['height_shift_range'] = 0.05
        instruct['width_shift_range'] = 0.05
        instruct['shear_range'] = 0
        instruct['zoom_range'] = [0.9, 1.1]
        instruct['horizontal_flip'] = True
        instruct['vertical_flip'] = True
        instruct['fill_mode'] = 'wrap'
        instruct['cval'] = 0. # or 'nearest'
        instruct['vgg_zoom'] = None
    else:
        # augmentations OFF
        instruct['rotation_range'] = 0
        instruct['height_shift_range'] = 0
        instruct['width_shift_range'] = 0
        instruct['shear_range'] = 0
        instruct['zoom_range'] = [1, 1]
        instruct['horizontal_flip'] = False
        instruct['vertical_flip'] = False
        instruct['fill_mode'] = 'wrap'
        instruct['cval'] = 0. # or 'nearest'
        instruct['vgg_zoom'] = None

    instruct, meta = eddieSwitch(instruct, aws)

    instruct['save_pics'] = False
    instruct['save_gen_output'] = False

    return instruct, meta


def eddieSwitch(instruct, aws=True):
    """
    Depending on if we're running on aws or not:
        - Add correct default directories to `instruct` for reading subjects from
        - Load catalog (metatable) 
    
    Args:
        instruct (dict): overall configuration for this experiment
        aws (bool, optional): Defaults to True. if True, use aws directory and meta defaults
    
    Returns:
        dict: configuration, now including directory to load subjects from
        meta: catalog of subjects (including directory-less filenames) and labels
    """
    if aws=='SDSS':  # still not sure what this is doing
        instruct['directory'] = r'/exports/eddie/scratch/s1220970/regenerated/SDSS/'
        meta = pd.read_csv(r'/exports/eddie/scratch/s1220970/regenerated/SDSS/key_sel2000.csv')
    elif aws:  # use AWS directories
        instruct['directory'] = r'/home/ubuntu/subjects/static_processed/512/'
        meta = pd.read_csv(r'/home/ubuntu/tables/training/meta_table.csv')
    else:  # use normal computer directories
        instruct['directory'] = r'/data/tidalclassifier/subjects/static_processed/512/'
        meta = pd.read_csv(r'/data/tidalclassifier/tables/training/meta_table.csv')
    return instruct, meta


def defaultSetup(aws):
    # TODO rename define_parameters and remove this func
    return define_parameters(aws)

#######

# Seeing if gs_aON runs and saves properly. Will then do the required pair of plots

# TODO: See which images are well-classified by which networks, and add those properties later before metaclassifying.

# TODO: VGG should show visible (plottable) training at both stages. Runs - but how well? 150->244, linear->log

# TODO: Redo nearly all plots in better font size and with seaborn? Use saved results data, don't rerun.

#######

# TODO: should redo renamer at some point. Need to manually update A.

# TODO: WND-CHARM longer train (and ROC, already set up)

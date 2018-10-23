"""
based on "Building powerful image classification models using very little data" blog, FC, Keras author
"""
import logging

# from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import numpy as np
import pandas as pd
from helper_funcs import ThreadsafeIter, shuffle_df, load_lines
import time

from keras import backend as K
K.set_image_dim_ordering('th')
from meta_CNN import custom_flow_from_directory, fold_tables, trainCNNOnTable
# color channel first


def trainAllOnTables(model_func, train_tables, val_tables, instruct):
    """Train a defined CNN many times.
    High-level wrapper for `trainCNNOnTable`
    
    Args:
        model_func (function): given 'instruct' dict, returns trainable model
        train_tables (list): of dataframes, training on table[n] for run n
        val_tables (list): of dataframes, testing on table[n] for run n. No train overlap allowed.
        instruct (dict): configuration parameters for cnn to train, pre-processing, etc.
    
    Returns:
        list: of np.arrays of training accuracy by 
        [type]: [description]
        [type]: [description]
        [type]: [description]
        [type]: [description]
    """

    runs = instruct['runs']

    # final benchmark measurements
    acc = np.zeros(runs)
    val_acc = np.zeros(runs)
    loss = np.zeros(runs)
    val_loss = np.zeros(runs)

    model = None  # to ensure assignment
    for run in range(runs):
        time.sleep(0.5)  # wait half a second to allow any generators to complete
        logging.info('Initialise run ' + str(run))
        train_table = train_tables[run]
        val_table_full = val_tables[run]
        #split val_table into train_val and meta_val. Note that folds should be halved. Min 4.
        val_table_train = val_table_full[:int(len(train_table)/2)] # to train the meta-classifier
        val_table_test = val_table_full[int(len(train_table)/2):] # to test the meta-classifier
        # here is where things need to get more complicated

        # train the CNN, and make sure save_gen_output = True
        instruct['save_gen_output'] = False
        r_acc, r_val_acc, r_loss, r_val_loss, trained_model = trainCNNOnTable(model_func, train_table, val_table_train, run, instruct)

        # generate a prediction on each row in train_table and val_table
        # (potentially extend to averaging different augmentations)
        # challenge is custom_flow is random and doesn't record which pictures are which elements - think!
        instruct['save_gen_output'] = True

        # custom_gen_train = custom_flow_from_directory(train_table, instruct, even_split=True, gen_name='train')
        custom_gen_val = custom_flow_from_directory(val_table_train, instruct, even_split=True, gen_name='val')

        # custom_gen_train = threadsafe_iter(custom_gen_train)
        custom_gen_val = ThreadsafeIter(custom_gen_val)

        # make predictions on test data
        Y = trained_model.predict_generator(custom_gen_val, val_samples=instruct['nb_val_samples']*10)
        # this will also cause generator output to be saved
        # load generator output
        Y_true = load_lines(instruct['directory'] + 'train' + '_label.txt')[:instruct['nb_train_samples']]
        Y_pic_ids = load_lines(instruct['directory'] + 'train' + '_pic.txt')[:instruct['nb_train_samples']]
        # place in dataframe train_feats
        index = np.arange(len(Y_true))
        train_meta = pd.DataFrame(index=index, data={'picture_id': Y_pic_ids, 'CNN_label': Y, 'true_label': Y_true})

        # look up the mask values of those images in pawlik meta_table
        meta_table = pd.read_csv(instruct['directory'] + 'meta_with_A.csv')
        meta_table = meta_table[['picture_id', 'standard_A', 'mask_A']]  # only include pawlik data (for now)
        # inner join on picture_id
        train_meta = pd.merge(train_meta, meta_table,how='inner')

        # repeat for other CNN's, if appropriate. May be same but retrained.

        # make predictions from meta_val
        # custom_gen_train = custom_flow_from_directory(train_table, instruct, even_split=True, gen_name='train')
        logging.debug(val_table_test.head())
        custom_gen_val = custom_flow_from_directory(val_table_test, instruct, even_split=True, gen_name='val')

        # custom_gen_train = threadsafe_iter(custom_gen_train)
        custom_gen_val = ThreadsafeIter(custom_gen_val)

        # make predictions on test data
        Y = trained_model.predict_generator(custom_gen_val, val_samples=instruct['nb_val_samples']*10)
        # this will also cause generator output to be saved
        # load generator output
        Y_true = load_lines(instruct['directory'] + 'train' + '_label.txt')[:instruct['nb_train_samples']]
        Y_pic_ids = load_lines(instruct['directory'] + 'train' + '_pic.txt')[:instruct['nb_train_samples']]
        # place in dataframe train_feats
        index = np.arange(len(Y_true))
        test_meta = pd.DataFrame(index=index, data={'picture_id': Y_pic_ids, 'CNN_label': Y, 'true_label': Y_true})

        # look up the mask values of those images in pawlik meta_table
        meta_table = pd.read_csv(instruct['directory'] + 'meta_with_A.csv')
        meta_table = meta_table[['picture_id', 'standard_A', 'mask_A']]  # only include pawlik data (for now)
        # inner join on picture_id
        test_meta = pd.merge(test_meta, meta_table, how='inner')

        train_meta.to_csv(instruct['directory'] + 'train_meta.csv')
        test_meta.to_csv(instruct['directory'] + 'test_meta.csv')

        # using only feature_table as input, train a meta-classifier (NN? SVM? Check kaggle)
        # TODO

        # make final predictions on val_meta

        # measure accuracy on val_feats. This is r_acc etc

        acc[run] = r_acc
        val_acc[run] = r_val_acc
        loss[run] = r_loss
        val_loss[run] = r_val_loss
    return acc, loss, val_acc, val_loss, model


def crossValidationRandomAll(model_func, meta, instruct):
    """Randomised multi-trial cross-validation with generators"""
    # pick a random fold to crossValidate with, for n runs
    train_tables = []
    val_tables = []
    for run in range(instruct['runs']):  # construct the desired train and val table pairs to apply train_on_tables to
        meta = shuffle_df(meta)
        folded_train_tables, folded_val_tables = fold_tables(meta, instruct['folds'])
        train_table = folded_train_tables[0]
        val_table = folded_val_tables[0]
        train_tables.append(train_table)
        val_tables.append(val_table)
    return trainAllOnTables(model_func, train_tables, val_tables, instruct)
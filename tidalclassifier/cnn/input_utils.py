import logging

from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tidalclassifier.utils.custom_image_utils import augment, apply_corrections
from tidalclassifier.utils.helper_funcs import ThreadsafeIter, shuffle_df, to_json, remove_file


def read_image(row, instruct, debug=False):
    im = 'error'
    input_mode = instruct['input_mode']
    directory = instruct['directory']
    if debug: print(input_mode)
    if input_mode == 'masked':
        file_loc = directory + 'sextractor/' + row['masked_filename']
        im = fits.getdata(file_loc)  # TODO: masked should be saved (1,256,256), consistent with stacked
        im = np.expand_dims(im, axis=0)
    elif input_mode == 'stacked' or 'color' or 'threshold' or 'threshold_5sig' or 'threshold_mask_5sig' or 'threshold_bkg_5sig' or 'threshold_color_5sig' or 'threshold_3sig' or 'threshold_mask_3sig' or 'threshold_bkg_3sig' or 'threshold_color_3sig':
        file_loc = directory + str(row[input_mode + '_filename'])
        if debug: print(file_loc)
        im = fits.getdata(file_loc)
        if instruct['aws'] == 'SDSS' and instruct['input_mode'] == 'stacked':
            im = np.array([im])  # expand into third dimension
        if debug: print(im.shape)
        if input_mode == 'threshold_color_5sig' or input_mode == 'threshold_color_3sig':  # TODO: temporary until renamer re-run
            im = im[0, :, :, :]
    else: exit(1)
    if np.size(im) < 10:
        exit(1)
    if debug:
        print('returning', im.shape)
    return im


def construct_image(row, instruct, augmentations=True, debug=False):
    """given a meta-table row, read in the premade image and apply desired transforms"""
    im = read_image(row, instruct) # simple read of fits file

    if im.shape != (1, 512, 512) and im.shape != (1, 256, 256) and im.shape != (3,256,256) and im.shape != (3,512,512):  # TODO: make automatic. But beware: uncropped!
        print('shape error with', row)
        print(im.shape)
        exit(0)

    if augmentations: im = augment(im, instruct)
    if debug:
        print('augmented')
    im = apply_corrections(im, instruct)
    if debug:
        print('corrected')
    if instruct['save_pics']:
        hdulist = fits.open(instruct['directory'] + row['threshold_filename'])  # read original image for base file
        hdu = hdulist[0]  # open main compartment
        hdu.data = im  # set main compartment data component to be the final image
        hdu.writeto(instruct['directory']+'debug_'+str(row['ID'])+'_'+str(row['FEAT']) + '_' + str(row['CONF'])+'_'+str(np.random.randint(0,1000))+'.fits', overwrite=True)
        # random number to avoid augmented overwrites
        if debug:
            print('saved')
    # scaled_plot(np.squeeze(im), plt)
    # plt.show()
    return im

"""Generators"""
def custom_selected_images(meta, instruct, picture_id_list):
    channels = instruct['channels']
    img_width = instruct['img_width']
    img_height = instruct['img_height']
    while True:
        batch_size = len(picture_id_list)
        iter=0
        data = np.zeros((batch_size, channels, img_width, img_height))
        labels = np.ones(batch_size)*-1
        while iter < batch_size:
            selected_id = picture_id_list[iter]
            print(selected_id)
            rows = meta[meta.picture_id == selected_id]
            if len(rows) > 1: exit(1)
            row = rows.squeeze()
            im = construct_image(row, instruct)
            data[iter, :, :, :] = im
            feature = row['FEAT']
            labels[iter] = 1
            if feature == 'N': labels[iter] = 0
            iter += 1
        yield (data, labels)


def custom_flow_from_directory(
    table, 
    instruct, 
    gen_name='default_gen',
    write=False, 
    even_split=True,
    p=False,
    class_mode='both',
    debug=False):
    """Yield (subjects, labels) batch tuples from subjects saved in directory and catalog
    
    Args:
        table ([type]): [description]
        instruct ([type]): [description]
        gen_name (str, optional): Defaults to 'no_write'. [description]
        even_split (bool, optional): Defaults to True. [description]
        p (bool, optional): Defaults to False. [description]
        class_mode (str, optional): Defaults to 'both'. [description]
    """
    batch_size = instruct['batch_size']
    channels = instruct['channels']
    img_width = instruct['img_width']
    img_height = instruct['img_height']
    name = 'no_write'
    if instruct['save_gen_output']:
        name = gen_name

    # table is a pandas table with only desired files included
    # index = 0 # current table index
    table_full = table.copy()  # check not by reference

    if write:
        label_fname = instruct['directory'] + name + '_' + str(instruct['run']) + '_label.txt'
        remove_file(label_fname)  # necessary to keep cleaning files each time, or will append forever!
        pic_fname = instruct['directory'] + name + '_' + str(instruct['run']) + '_pic.txt'
        remove_file(pic_fname)  # necessary to keep cleaning files each time, or will append forever!

    while True:
        data = np.zeros((batch_size, channels, img_width, img_height))
        labels = np.ones(batch_size) * -1
        iteration = 0
        while iteration < batch_size:  # iterate until have completed batch
            logging.debug(iteration)
            if len(table[table.FEAT != 'N']) == 0:
                table = table_full  # reset the table if all entries have been used
            if len(table[table.FEAT == 'N']) == 0:
                table = table_full  # reset the table if all entries have been used
            if even_split:
                feat_switch = np.random.randint(0,2)  # high limit is exclusive
                if feat_switch == 1:
                    table_in = table[table.FEAT != 'N']
                else:
                    table_in = table[table.FEAT == 'N']
            else:
                table_in = table

            # table_in contains all images that may possibly be selected in this single image loop (by class, usually)
            rand_index = np.random.randint(0,len(table_in))
            # print('rand index: ', rand_index)
            picture_id = table_in.iloc[rand_index]['picture_id']  # pick a random picture id
            table = table[table.picture_id != picture_id]  # remove that pic from the OUTER table, don't redraw (yet)
            rows = table_in[table_in.picture_id == picture_id]  # pick metatable rows with that pic_id
            if len(rows) > 1:
                exit(1) # if pic id duplicates, exit!
            row = rows.squeeze()         
            im = construct_image(row, instruct)  # read image contained in that metatable row
            data[iteration, :, :, :] = im  # save for X output
            feature = rows.iloc[0]['FEAT']  # find feature of current image

            labels[iteration] = 1  # assume Y = tidal
            if feature == 'N':
                labels[iteration] = 0 # if feature is N, change to Y = not tidal

            if write:  # append record of labels to 'name'
                with open(label_fname, "a") as label_file:
                    label_file.write(str(int(labels[iteration]))+'\n')
                with open(pic_fname, "a") as label_file:
                    label_file.write(str(int(picture_id))+'\n')

            iteration += 1

        if debug:
            logging.info(data.shape)
            final_batch_im = data[-1, 0, :, :]  # batch, channel, height, width
            final_batch_label = labels[-1]
            name = gen_name + '_' + row['ID'] + '_' + str(final_batch_label) + '_' + str(np.random.rand())
            logging.info(name)
            logging.info(final_batch_im.shape)
            plt.clf()
            plt.imshow(final_batch_im, cmap='gray')
            plt.savefig(name + '.png')

        logging.info('Mean batch label: {}'.format(labels.mean()))
        logging.debug('batch shape: {}'.format(data.shape))
        if class_mode is None:
            yield data
        else: yield (data, labels)


def fold_tables(meta, instruct):
    """
    Separate catalog into instruct['folds'] cross-validation folds.
    Check that no galaxy appears in both the train table and val table for each single permutation
    
    Shuffles catalog, hence resulting folds are unique for each call
    instruct['folds'] controls how many folds to create (e.g. 5 for 5-fold cross-validation)
    instruct['tidal_conf'] controls how expert labels are binned into binary classes.
    
    Args:
        meta (pd.DataFrame): catalog
        instruct (dict): configuration instructions
    
    Raises:
        ValueError: if instruct['tidal_conf'] is not a defined option (below)
    
    Returns:
        list: where nth item is train table for nth permutation
        list: where nth item is validation table for nth permutation
    """

    folds = instruct['folds']

    # all combinations need these two
    conf_4 = np.array(meta.CONF == 4, dtype=bool)
    conf_0 = np.array(meta.CONF == 0, dtype=bool)

    if instruct['tidal_conf'] == 34:
        conf_3 = np.array(meta.CONF == 3, dtype=bool)
        tidal_table = meta[conf_3 + conf_4]
        nontidal_table = meta[meta.CONF == 0]
    elif instruct['tidal_conf'] == 4:
        tidal_table = meta[conf_4]
        nontidal_table = meta[meta.CONF == 0]
    elif instruct['tidal_conf'] == 134:
        conf_3 = np.array(meta.CONF == 3, dtype=bool)
        conf_1 = np.array(meta.CONF == 1, dtype=bool)
        tidal_table = meta[conf_3 + conf_4]
        nontidal_table = meta[conf_1+conf_0]
    else: 
        failure_str = 'fatal fold error: instruct tidal_conf not recognised'
        raise ValueError(failure_str)

    
    tidal_val_size = int(len(tidal_table)/folds)
    nontidal_val_size = int(len(nontidal_table)/folds)

    train_tables = ['error' for v in range(folds)]
    val_tables = ['error' for v in range(folds)]

    for fold in range(folds):
        # choose the boundaries of moving window to select as val data
        tidal_window_low_edge = fold * tidal_val_size
        tidal_window_high_edge = (fold+1) * tidal_val_size
        nontidal_window_low_edge = fold * nontidal_val_size
        nontidal_window_high_edge = (fold+1) * nontidal_val_size
        # validation set is the rows within fold's selected window
        # val window should include low edge and exclude high edge
        val_tidal_table = tidal_table[tidal_window_low_edge:tidal_window_high_edge]
        val_nontidal_table = nontidal_table[nontidal_window_low_edge:nontidal_window_high_edge]
        val_table = pd.concat((val_tidal_table, val_nontidal_table))
        # train set is all the other rows
        # train_below should exclude low_edge
        train_tidal_table_below = tidal_table[:tidal_window_low_edge]
        train_nontidal_table_below = nontidal_table[:nontidal_window_low_edge]
        if fold == (folds - 1): # final row, don't try access above limit!
            train_tidal_table_above = pd.DataFrame()
            train_nontidal_table_above = pd.DataFrame()
        else:
            # train_above should include high edge as val window excludes it
            train_tidal_table_above = tidal_table[tidal_window_high_edge:]
            train_nontidal_table_above = nontidal_table[nontidal_window_high_edge:]
        train_table = pd.concat((train_tidal_table_below, train_nontidal_table_below,
                                 train_tidal_table_above, train_nontidal_table_above))

        val_table = shuffle_df(val_table)
        train_table = shuffle_df(train_table)

        val_tables[fold] = val_table
        train_tables[fold] = train_table

    for fold in range(folds): # verify that no pictures appear twice in any train/test pair
        val_pics = val_tables[fold]['picture_id'].unique()
        train_pics = train_tables[fold]['picture_id'].unique()
        for val_v in val_pics:
            for train_v in train_pics:
                if val_v == train_v:
                    print('fold error: duplicate pic detected!')
                    print(val_v, train_v)
                    exit(1)

    return train_tables, val_tables

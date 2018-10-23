import os
from shutil import copyfile
import subprocess
import logging

import numpy as np
import pandas as pd
from astropy.io import fits

from tidalclassifier.telescope_download.construct_url_text import create_url_text
from tidalclassifier.telescope_download.load_clean_cutout import load_cutout
from tidalclassifier.telescope_download.load_clean_table import load_table
from tidalclassifier.telescope_download.load_clean_url import load_url
from tidalclassifier.utils.custom_image_utils import hasGaps, get_stacked_image, get_color_image
from tidalclassifier.utils.thresholder import thresholdImage


def constructMasterTable(atk, url, cutout, dimension):
    """
    Merge URL and Cutout tables to identify shared pictures
    Verify that there are three bands.
    Join with Atkinson Table 4 for labels
    
    Args:
        atk (pd.DataFrame): Atkinson Table 4, including expert labels
        url (pd.DataFrame): [description]
        cutout (pd.DataFrame): [description]
        dimension (int): expected x, y size of images
    
    Returns:
        pd.DataFrame: catalog of available single band images
    """

    # u and c is url and cutout table combined i.e. urls matched with object identifier (picture_id)
    u_and_c_table = pd.merge(url, cutout, how='outer', left_index=True, right_index=True)

    # remove all pictures with dimensions that don't match target
    u_and_c_table = u_and_c_table[u_and_c_table.x_width == dimension]
    u_and_c_table = u_and_c_table[u_and_c_table.y_width == dimension]
    # check at least 3 pics still exist for all bands
    for id in range(1780):
        pic_rows = u_and_c_table[u_and_c_table.picture_id == id]
        if len(pic_rows) < 3:
            print('error: not enough bands for id ' + str(id))

    u_and_c_table['picture_id'] = u_and_c_table['picture_id'].map(lambda x: int(x))
    u_and_c_table['picture_id_index'] = u_and_c_table['picture_id']
    u_and_c_table = u_and_c_table.set_index('picture_id_index')

    master = atk.join(u_and_c_table)

    master['filename'] = master['ID'] + '__' + master['raw_url_id'].astype('int').astype('str') + \
                         '__' + master['picture_id'].astype('int').astype('str') + '__' + master[
                             'band'] + \
                         '__' + master['FEAT'] + '__' + master['CONF'] + '.fits'

    master['index'] = np.arange(len(master))
    master = master.set_index('index')

    return master



# read in images, write headers, copy under new name
# def update_fits(row):
#     # load fits
#     print(row['ID'])
#     filename = row['filename']
#     hdulist = fits.open(filename,mode='update')
#     # update fits header
#     # header is not caps sensitive
#     update_list = ['ID', 'RA', 'DE', 'FEAT','CONF']
#     dic = dict()
#     for item in update_list:
#         # print(dic)
#         dic[item] = str(row[item])
#     # dic = {'val':'12'}
#     # print(type(dic))
#     # [print( (item, row[item]) ) for item in update_list]
#     hdulist[0].header.update(dic)
#     # overwrite, now with header updated
#     # print(filename)
#     # fits.writeto(hdulist)
#     # fits.append(filename, 12, header='val')
#     # [fits.append(filename, str(row[item]), header=item) for item in update_list]
#     hdulist.close()

def constructMetaTable(master_table, new):
    # meta_table contains single details the final images for use by CNN: stacked, 3D, and masked
    if new:
        # begin constructing meta_table
        # include only imaged bands with no errors
        master_table = master_table[master_table['errors'] != 1]
        meta_table = master_table[master_table.band == 'g']
        meta_table['meta_index'] = np.arange(len(meta_table))  # need to correct index after dropping rows
        meta_table = meta_table.set_index('meta_index')
        del meta_table['band']  # and any other non-meta details
        # if any duplicates (after removing errors), pick the first
        selected_raw_ids = []
        for picture_id in range(1780):
            if len(meta_table[meta_table.picture_id == picture_id]) > 0:
                selected_raw_ids.append(meta_table[meta_table.picture_id == picture_id].iloc[0]['raw_url_id']) # outside if
        meta_table = meta_table[meta_table['raw_url_id'].isin(selected_raw_ids)]
        # this will return the empty outline of meta_table, to be filled in by image constructing methods
        meta_table['stacked_filename'] = ['error' for i in range(len(meta_table))]
        meta_table['color_filename'] = ['error' for i in range(len(meta_table))]
        meta_table['masked_filename'] = ['error' for i in range(len(meta_table))]
        meta_table['threshold_5sig_filename'] = ['error' for i in range(len(meta_table))]  # the final image
        meta_table['threshold_mask_5sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        meta_table['threshold_bkg_5sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        meta_table['threshold_color_5sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        meta_table['threshold_3sig_filename'] = ['error' for i in range(len(meta_table))]  # the final image
        meta_table['threshold_mask_3sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        meta_table['threshold_bkg_3sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        meta_table['threshold_color_3sig_filename'] = ['error' for i in range(len(meta_table))]  # the binary mask used
        # this provides a check that those methods work properly
        meta_table['meta_index'] = np.arange(len(meta_table))  # need to correct index after dropping rows
        meta_table = meta_table.set_index('meta_index')
        meta_table.to_csv(download_table_dir + '/meta_table_new.csv')
    else: meta_table = pd.read_csv(download_table_dir + '/tables/meta_table_saved.csv')
    return meta_table


def createStackedImages(master, meta, meta_i, read_dir, write_dir):
    raw_url_id = meta.iloc[meta_i]['raw_url_id']  # raw url_id of row
    rows = master[master.raw_url_id == raw_url_id]  # select rows with that raw_url_id

    filename_list = rows['filename'].values
    logging.info('Combining {}'.format(filename_list))
    # filename_list = directory + filename_list     # will download to code folder
    im_list = [fits.getdata(read_dir + filename) for filename in filename_list]
    # print(np.sum(im_list[0]),np.sum(im_list[1]),np.sum(im_list[2]))
    if len(filename_list) != 3:
        logging.warning('Found ', len(filename_list)), ' files'
    im = get_stacked_image(im_list)
    # save cleaned version as image
    hdulist = fits.open(read_dir + rows.iloc[0]['filename'])  # read (the first) original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = im  # set main compartment data component to be the subtracted image
    hdu.header['STACKED'] = True  # add line in header to indicate that this is a stacked image

    save_filename = rows.iloc[0]['ID'] + '_stacked.fits'

    hdu.writeto(write_dir + save_filename, clobber=True)  # write to file, may overwrite
    meta_table.set_value(meta_i, 'stacked_filename', save_filename)  # check that this is in place


def createColorImages(master, meta, meta_i, read_dir, write_dir):


    raw_url_id = meta.iloc[meta_i]['raw_url_id']  # raw_url_id of row
    # # print(raw_url_id)
    rows = master[master.raw_url_id == raw_url_id]  # select rows with that picture_id
    # # print(rows)
    filename_list = rows['filename'].values
    # print(filename_list)
    # print(meta_i)
    im_list = [fits.getdata(read_dir + filename) for filename in filename_list]
    if len(filename_list) != 3:
        print('warning: found ', len(filename_list)), ' files'
    im = get_color_image(im_list)

    # write image to fits
    # save cleaned version as image
    save_filename = rows.iloc[0]['ID'] + '_color.fits'

    hdulist = fits.open(read_dir + rows.iloc[0]['filename'])  # read (the first) original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = im  # set main compartment data component to be the subtracted image
    hdu.header['COLOR'] = True  # add line in header to indicate that this is a stacked image
    hdu.writeto(write_dir + save_filename, clobber=True)  # write to file, may overwrite

    meta_table.set_value(meta_i, 'color_filename', save_filename)  # check that this is in place
    # print(picture_id)
    # print(meta_table['color_filename'][:4])
#
# def createMaskedImages(master, meta, meta_i, read_dir=default_read_dir,
#                        write_dir=default_write_dir):
#     # calls the mask script
#     # picture_id = meta.iloc[meta_i]['picture_id']  # picture_id of row
#     # presumes that you have already run createStackedImages
#     print(meta.iloc[meta_i]['ID'])
#     im = fits.getdata(write_dir + meta.iloc[meta_i]['ID'] + '_stacked.fits') # stacked image is (1,width,height)
#     # print(np.sum(im))
#     im = im[0,:,:] #plt.imshow requires [width, height]
#     # print(np.sum(im))
#     # print(np.shape(im))
#     table_id = meta.iloc[meta_i]['ID']
#     # print(table_id)
#     # if table_id == 'W2-385': print(im)
#     maskImage(im, table_id, read_dir=read_dir, write_dir=write_dir) # mask image will masked image on its own
#     save_filename = meta.iloc[meta_i]['ID'] + '_masked.fits'
#     meta_table.set_value(meta_i, 'masked_filename', save_filename)  # check that this is in place


def createThresholdImages(master, meta, meta_i, pre_instruct, read_dir, write_dir):
    # calls the mask script
    picture_id = meta.iloc[meta_i]['picture_id']  # picture_id of row
    # presumes that you have already run createStackedImages

    logging.info('Thresholding {}'.format(meta.iloc[meta_i]['ID']))
    stacked_im = fits.getdata(write_dir + meta.iloc[meta_i]['ID'] + '_stacked.fits') # stacked image is (1,width,height)
    color_im = fits.getdata(write_dir + meta.iloc[meta_i]['ID'] + '_color.fits') # color image is (3,width,height)
    table_id = meta.iloc[meta_i]['ID']
    thresholdImage(stacked_im, color_im, table_id, pre_instruct, read_dir=read_dir, write_dir=write_dir) # mask image will masked image on its own

    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_5sig.fits'
    meta.set_value(meta_i, 'threshold_5sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_mask_5sig.fits'
    meta.set_value(meta_i, 'threshold_mask_5sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_bkg_5sig.fits'
    meta.set_value(meta_i, 'threshold_bkg_5sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_color_5sig.fits'
    meta_table.set_value(meta_i, 'threshold_color_5sig_filename', save_filename)  # check that this is in place

    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_3sig.fits'
    meta.set_value(meta_i, 'threshold_3sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_mask_3sig.fits'
    meta.set_value(meta_i, 'threshold_mask_3sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_bkg_3sig.fits'
    meta.set_value(meta_i, 'threshold_bkg_3sig_filename', save_filename)  # check that this is in place
    save_filename = meta.iloc[meta_i]['ID'] + '_threshold_color_3sig.fits'
    meta.set_value(meta_i, 'threshold_color_3sig_filename', save_filename)  # check that this is in place


# W1-72 is the first with <95% filled (one band only)
# W1-151 is the first blank example
# W1-160 is the first edge case
# W1-255 has only two bands missing and a strange effect on the third
# W1-552 is an excellent masking success (probably)
def checkDownloads(master, image_dir):
    logging.info('Checking raw images are not corrupt')
    for index in range(len(master)): # load fits for every raw band image
        current_file = master.iloc[index]['filename']
        current_raw_url_id = master.iloc[index]['raw_url_id']
        error = False
        try:
            im = fits.getdata(os.path.join(image_dir, current_file))
            if hasGaps(im):
                logging.warning('image has gaps: {}'.format(current_file))
                error = True
        except (IOError, TypeError):  # corrupt file: re-download
            logging.warning('file error: {}'.format(current_file))
            error = True

        if error:  # set all bands of this image as errors
            master.loc[master.raw_url_id == current_raw_url_id, 'errors'] = 1
            # TODO check at least 3 remain and keep those
            # TODO copy errors somewhere for later analysis

    logging.info('Downloads successfully checked but NOT corrected')
    return master

def get_external_service_tables(download_table_dir):

    # raw urls will be concatenated and placed here
    url_loc = os.path.join(download_table_dir, 'url_list_full_512.txt')  
    # after parsing raw urls into image metadata, save the (intermediate) parsed table here
    url_parsed_loc = os.path.join(download_table_dir, 'url_list_full_512_parsed.csv')

    # concatenate raw urls and place them at url_loc
    url_subdir = os.path.join(download_table_dir, 'url_text_512')
    if not os.path.isdir(url_subdir):
        os.mkdir(url_name)
    create_url_text(url_subdir, url_loc) # final raw list will be placed in url_loc for later use


    table_loc = os.path.join(download_table_dir, 'atkinson_table4.csv')

    cutout_loc = os.path.join(download_table_dir, 'cutout_512.txt')
    cutout_parsed_loc = os.path.join(download_table_dir, 'cutout_512_parsed.csv')

    atk_table = load_table(table_loc)
    url_table = load_url(url_loc, parsed_loc=url_parsed_loc) # will use the raw list created by create_url_text
    cutout_table = load_cutout(cutout_loc, parsed_loc=cutout_parsed_loc)

    return atk_table, url_table, cutout_table


def download_raw_images(master_table, target_dir):

    urls_to_download = list(master_table['url'])
    urls_to_download_wget_file = 'temp_url.txt'
    with open(urls_to_download_wget_file, "w") as f:
        for url in urls_to_download:
            f.write(url + '\n')
    
    raw_subdir = os.path.join(target_dir, 'before_renaming')
    if not os.path.isdir(raw_subdir):
        os.mkdir(raw_subdir)
    # rapid download of all files, with default names of url retrieved
    # command_string = r'wget --content-disposition -i ' + urls_to_download_wget_file
    command_string = 'cat {}'.format(urls_to_download_wget_file) + r' | parallel --gnu wget {}' + ' --directory-prefix={} --no-clobber'.format(raw_subdir)
    #  + ' -O ' + write_dir + row['filename']
    subprocess.call(command_string, shell=True)
    # rename files from url retrieved to meaningful name
    for _, row in master_table.iterrows():
        expected_name = row['url'][77:].replace(r'%5B', '[').replace(r'%3A', ':').replace(r'%2C',',').replace(r'%5D',']')
        desired_name = row['filename']
        copyfile(os.path.join(raw_subdir, expected_name), os.path.join(target_dir, desired_name))

    # [update_fits(master_table.iloc[index]) for index in range(len(master_table))]


def process_raw_images(master_table, meta_table, read_dir, write_dir):
    """Turn raw images into processed images, and add to meta table
    WARNING must always re-run all 3, with a fresh meta_table. It's pretty quick!
    WARNING modifies meta_table!
    
    Args:
        master_table (pd.DataFrame): catalog of single band images
        meta_table (pd.DataFrame): catalog of galaxies
        read_dir (str): directory from which to read raw images
        write_dir (str): directory into which to write processed images
    """
    [createStackedImages(master_table, meta_table, meta_index, read_dir, write_dir) for meta_index in range(len(meta_table))]
    [createColorImages(master_table, meta_table, meta_index, read_dir, write_dir) for meta_index in range(len(meta_table))]
    # [createMaskedImages(master_table, meta_table, meta_index) for meta_index in range(len(meta_table))]
    # TODO refactor, threshold uses derived (write dir) stacked image as input
    [createThresholdImages(master_table, meta_table, meta_index, pre_instruct, write_dir, write_dir) for meta_index in range(len(meta_table))]


if __name__ == '__main__':

    logging.basicConfig(
        filename='latest_download.log',
        filemode='w',
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    # local mode
    # note that the ~ path is a bit buggy: better to use /home/...
    download_table_dir = '/data/tidalclassifier/tables/download'
    read_dir = '/data/tidalclassifier/subjects/raw/512_parallel/'  # TODO rename
    write_dir ='/data/tidalclassifier/subjects/static_processed/512/'  # TODO rename

    # atk_table, url_table, cutout_table = get_external_service_tables(download_table_dir)

    # dimension = 511
    # master_table = constructMasterTable(atk_table, url_table, cutout_table, dimension)
    # master_table.to_csv(os.path.join(download_table_dir, 'master_table_before_download.csv'))

    # # download_raw_images(master_table, target_dir=read_dir) 

    # # intermittent errors on both single-file and batch-file scales
    # master_table['errors'] = np.zeros(len(master_table))
    # master_table = checkDownloads(master_table, image_dir=read_dir)
    # # TODO perhaps need a more sophisticated way to remove images with artifacts/diffraction spikes - flag images?

    # master_table.to_csv(os.path.join(download_table_dir, 'master_table.csv'))


    master_table = pd.read_csv(os.path.join(download_table_dir, 'master_table.csv'))
    meta_table = constructMetaTable(master_table, new=True)
    # threshold process requires parameter choices
    pre_instruct = {}
    # pre_instruct['sig_n'] = 5
    pre_instruct['mode'] = 'only_central'
    pre_instruct['dilation_radius'] = 10 # sx, sy

    process_raw_images(master_table, meta_table, read_dir, write_dir)

    meta_table.to_csv(os.path.join(download_table_dir, 'meta_table.csv'))

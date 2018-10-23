import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imshow
import scipy.ndimage as ndi
from scipy.ndimage.filters import convolve
from scipy.ndimage import zoom
# from scipy.stats import threshold DEPRECATED
from skimage import exposure
from skimage.io import imsave
from skimage import measure
from photutils import make_source_mask
from astropy.stats import sigma_clipped_stats
from astropy.io import fits


def rgb_scale(X):
    return X * 255 / X.max()


def clip(im, instruct):
    im = im.copy()  # np.place operates inplace, so let's avoid closure errors
    sig_n = instruct['sig_n']

    if instruct['rel']:
        flat_bkg, fake_bkg, std = estimate_background(im)  # sig above bkg
    else:
        std = np.std(im)  # sig above image mean

    if instruct['clip'] == 'threshold':
        av = np.mean(im)
        threshmax = av + sig_n * std
        newval = np.min(im)
        above_max = im > threshmax
        np.place(im, above_max, newval)
        # im = threshold(im, threshmax= av + sig_n * std, newval=np.min(im)) # minima if above value DEPRECATED
    elif instruct['clip'] == 'ceiling':
        im = np.clip(im, 0, sig_n * std)  # ceiling if above value
    return im


def crop(im, instruct):
    # crop from center by width of w IN EACH DIRECTION
    w = instruct['w']
    channels, x_len, y_len = im.shape
    xmin = int(x_len/2 - w)
    xmax = int(x_len/2 + w)
    ymin = int(y_len/2 - w)
    ymax = int(y_len/2 + w)
    return im[:, xmin:xmax, ymin:ymax]


def estimate_background(im):
    # im must be BW
    if len(im.shape) == 3: im=im.squeeze()

    mask = make_source_mask(im, snr=2, npixels=5, dilate_size=11)
    mean, median, std = sigma_clipped_stats(im, sigma=3.0, mask=mask)
    flat_background = np.ones_like(im) * mean
    fake_background = np.random.normal(mean, std, size=im.shape)
    return flat_background, fake_background, std


def apply_corrections(im, instruct):
    # if im.min() < 0: print('warning: negative values in image')
    if len(im.shape) == 2: im = np.expand_dims(im, axis=0)
    if instruct['vgg_zoom'] is not None: 
        im = zoom(im, [1.0,instruct['vgg_zoom'], instruct['vgg_zoom']]) # shrink before crop
    if instruct['crop']: 
        im = crop(im, instruct)
    if instruct['clip'] is not None: 
        im = clip(im, instruct)
    if instruct['convolve']: 
        im = convolve(im, weights=np.full((1, 3, 3), 1.0 / 9.))
    if instruct['scale'] == 'pow':
        im = np.abs(im)
        im_original_max = im.max()
        im = np.power(im, instruct['pow_val'])
        im = im * (im_original_max / im.max())
    if instruct['scale'] == 'log':
        # may have unexpected behaviour if im background is not about 0
        im[im < 0] = 0
        im = np.log(1+im)
    if instruct['multiply'] != 1:
        im = im * instruct['multiply']
    # # TODO: photographical adjustments require rescale, this is bad!
    # if instruct['hist']:
    #     im = exposure.rescale_intensity(im)  # linear rescale according to typical values, -1.0 -> 1.0
    #     im = exposure.equalize_adapthist(im, clip_limit=instruct['clip_lim'])  # excellent local contrast enhancement
    # if instruct['gamma'] != 1: im = exposure.adjust_gamma(im, instruct['gamma'])  # raise to power, brighten if a < 1, gets noisy

    if len(im.shape) == 2: im = np.expand_dims(im, axis=0)
    return im # temporarily making apply_corrections just clip, so meta == integrated


def get_stacked_image(im_list):
    # merge 3 images into stacked image
    stacked_im = np.sum(im_list, axis=0)
    stacked_im = np.array([stacked_im]) # (1, x, y) shape
    return stacked_im


def get_color_image(im_list):
    # merge 3 images into color image
    color_im = np.array(im_list)
    return color_im


def save_stacked_png(im_list, instruct, save_path, save_filename):
    # merge 3 images into stacked image
    im = get_stacked_image(im_list)
    im = apply_corrections(im, instruct)
    imsave(save_path + 'bw_images/stack__' + save_filename, im)


def save_color_png(im_list, instruct, save_path, save_filename):
    im = get_color_image(im_list)
    im = apply_corrections(im, instruct)
    im = np.swapaxes(im, 0, 2)  # imsave requires (x, y, channel) ordering
    imsave(save_path + 'color_images/col__' + save_filename, im)


def flip_axis(x, axis):  # TODO: remove?
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for
                      x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def augment(x, instruct):
    # x is a single image, so it doesn't have image number at index 0
    # slightly tailored from Keras source code (Francois, Keras author)
    img_row_index = instruct['row_index']
    img_col_index = instruct['col_index']
    img_channel_index = instruct['channel_index']

    # use composition of homographies to generate final transform that needs to be applied
    if instruct['rotation_range']:
        theta = np.pi / 180 * np.random.uniform(-instruct['rotation_range'], instruct['rotation_range'])
    else:
        theta = 0
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    if instruct['height_shift_range']:
        tx = np.random.uniform(-instruct['height_shift_range'], instruct['height_shift_range']) * x.shape[img_row_index]
    else:
        tx = 0

    if instruct['width_shift_range']:
        ty = np.random.uniform(-instruct['width_shift_range'], instruct['width_shift_range'] * x.shape[img_col_index])
    else:
        ty = 0

    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    if instruct['shear_range']:
        shear = np.random.uniform(-instruct['shear_range'], instruct['shear_range'])
    else:
        shear = 0
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    if instruct['zoom_range'][0] == 1 and instruct['zoom_range'][1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(instruct['zoom_range'][0], instruct['zoom_range'][1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

    h, w = x.shape[img_row_index], x.shape[img_col_index]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    x = apply_transform(x, transform_matrix, img_channel_index,
                        fill_mode=instruct['fill_mode'], cval=instruct['cval'])

    if instruct['horizontal_flip']:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)

    if instruct['vertical_flip']:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)

    return x


def scaled_plot(im, plt, clip_q=False, rel_clip=True):
    plt.figure()
    if len(im.shape)==3: im = np.squeeze(im)
    # if clip_q: im = clip(im,instruct)
    # im = exposure.rescale_intensity(im, out_range=(0, 1))
    print(im.min(), im.max())
    # plt.colorbar()
    # plt.imshow(np.sqrt(im))
    plt.imshow(im)
    return plt


def manual_rescale(im):
    im_original = im
    im = im - im.min()  # lowest number is 0
    im = 2* im / im.max()  # highest number is 2
    im = im - 1  # now in range [-1 , 1]
    # calculate shifts to allow duplication
    original_range = im_original.max() - im_original.min()
    new_range = im.max() - im.min()
    scale_factor = original_range / new_range # original * sf = new
    scaled_original_range = original_range * scale_factor
    shift = -1  # if image min is 0, shift is -1: sf makes 0 to 2 TODO generalise
    print(scale_factor, shift)
    print(im.max(), im.min())
    print(im_original.max(), im_original.min())
    return im, scale_factor, shift


def trimMask(mask, mode='only_central'):
    # given a mask, label the regions and return according to mode
    z_width, x_width, y_width = mask.shape
    # http://scikit-image.org/docs/dev/api/skimage.measure.html
    labels = measure.label(mask, background=0)  # binary mask replaced by integer labelled mask of connected regions
    central_label = labels[0, int(x_width/2.), int(y_width/2.)]
    if mode == 'only_central':
        return labels == central_label
    if mode == 'not_central':
        return (labels != central_label) and (labels != 0)
    print('incorrect trim mode')
    return mask


def hasGaps(im):
    if np.count_nonzero(im) < 0.8 * np.size(im):  # if any gaps or saturated pixels
        return True
    return False


def save_png(fits_loc, target_dir, instruct):
    try:
        data = fits.getdata(fits_loc)
    except FileNotFoundError:
        logging.warning('Galaxy not found at {}'.format(fits_loc))
        return None
    logging.debug(data)
    assert isinstance(data, np.ndarray)
    logging.debug(data.shape)

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    filename = os.path.split(fits_loc)[-1][:-5]  # name of fits file, without the .fits
    target_loc = os.path.join(target_dir, filename + '.png')

    im = apply_corrections(data, instruct)  # should be the same as cnn sees

    im = im.squeeze()
    plt.imshow(im, cmap='gray')
    plt.savefig(target_loc)
    plt.clf()
    logging.info('Saved to {}'.format(target_loc))

import numpy as np
from numpy import ma  # useful for thresholding
# import matplotlib.pyplot as plt
from astropy.io import fits
# from skimage import exposure
# from skimage.filters import rank
# from skimage.morphology import rectangle
from scipy.ndimage.filters import convolve
from scipy import stats
from skimage.transform import AffineTransform
from skimage.morphology import dilation, opening
from skimage.morphology import disk

from tidalclassifier.utils.custom_image_utils import clip, estimate_background, scaled_plot, trimMask


# https://stackoverflow.com/questions/46046928/how-to-find-replacement-of-deprecated-function-in-scipy
def threshold(a, threshmin=None, threshmax=None, newval=0):
    a = ma.array(a, copy=True)
    mask = np.zeros(a.shape, dtype=bool)
    if threshmin is not None:
        mask |= (a < threshmin).filled(False)

    if threshmax is not None:
        mask |= (a > threshmax).filled(False)

    a[mask] = newval
    return a


def thresholder(stacked_im, color_im, pre_instruct):
    # custom version of https://arxiv.org/pdf/1512.02000.pdf

    # some choices need to be made about exactly how this works. Currently:
    # mask is created by identifying pixels in convolved subtracted image n std's above 0 (i.e. above raw img bkg - bkg)
    # mask is filled by that convolved subtracted image

    flat_bkg, fake_bkg, std = estimate_background(stacked_im)
    threshold_level = pre_instruct['sig_n'] * std
    subtracted_im = stacked_im - flat_bkg

    filter_im = convolve(subtracted_im, weights=np.full((1, 3, 3), 1.0 / 9.)) # convolve the background-sub'd image

    # create mask by thresholding the convolved image
    nothing_above_threshold = threshold(filter_im, threshmax=threshold_level, newval=0)  # 0 if above value
    nothing_below_threshold = filter_im - nothing_above_threshold
    mask = np.ceil(nothing_below_threshold / nothing_below_threshold.max())

    # https://en.wikipedia.org/wiki/Connected-component_labeling
    trimmed_mask = trimMask(mask, mode=pre_instruct['mode']).astype(float)

    # fill the mask with convolved background-subtracted image, return as 'threshold' (as before)
    filled_mask_convd = filter_im * trimmed_mask

    trimmed_mask = np.squeeze(trimmed_mask)
    selem = disk(pre_instruct['dilation_radius'])
    dilated_mask = dilation(trimmed_mask, selem)
    # dilated_mask = dilated_mask - trimmed_mask

    smooth_mask = np.zeros_like(stacked_im)
    smooth_mask[0,:,:] = dilated_mask

    # zoomed_mask = AffineTransform(matrix=trimmed_mask, scale=pre_instruct['mask_zoom'])
    # convolve the mask with 5x5 average to smooth it out
    smooth_mask = convolve(smooth_mask, weights=np.full((1, 15, 15), 1.0 / 225.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))
    # smooth_mask = convolve(smooth_mask, weights=np.full((1, 5, 5), 1.0 / 25.))

    # fill the mask with original image and convolve dynamically if desired
    filled_mask = stacked_im * smooth_mask
    # invert the mask
    smooth_inverted_mask = 1 - smooth_mask


    # fill the inverted mask with background
    bkg_for_mask = fake_bkg * smooth_inverted_mask
    # combine
    threshold_bkg = filled_mask + bkg_for_mask
    # ensure positive everywhere: linear rescale (may need to adapt this)
    threshold_bkg = threshold_bkg + pre_instruct['sig_n'] * std
    threshold_bkg = np.abs(threshold_bkg) # neg. pixels are very rare (5 sigma deviation required) but will exist. Avoid.

    # fill the inverted mask with color image
    smooth_mask = np.squeeze(smooth_mask) # now (256x256)
    smooth_mask_3dim = np.stack([smooth_mask,smooth_mask,smooth_mask],axis=0) # now (3,256,256)
    threshold_col = smooth_mask_3dim * color_im # color im is also (3,256,256)
    # TODO: add fake bkg by band

    # threshold_bkg = smooth_inverted_mask

    # take care if lowering n_sig: this could become more significant, if below 3 sig or so, bkg will increase
    # scaled_plot(im, plt, clip_q=True)
    # scaled_plot(subtracted_im, plt, clip_q=True)
    # scaled_plot(filtered_im,plt,clip_q=True)
    # scaled_plot(cut_below_threshold, plt)
    # scaled_plot(mask, plt)
    # scaled_plot(trimmed_mask, plt)
    # scaled_plot(filled_mask, plt)
    # plt.show()

    # mask = np.expand_dims(mask, axis=0)
    # print(mask.shape)
    # print('new')
    return filled_mask_convd, trimmed_mask, threshold_bkg, threshold_col


def thresholdImage(stacked_im, color_im, table_id, pre_instruct, read_dir, write_dir, alt_filename=None):
    # This method is primary. Called by renamer script to generate masked images to be passed to metaCNN

    table_id = str(table_id)

    # 5 sig mode
    pre_instruct['sig_n'] = 5
    filled_mask, trimmed_mask, bkg_mask, col_mask = thresholder(stacked_im, color_im, pre_instruct)

    base_filename = read_dir + table_id + '_stacked.fits'
    if alt_filename != None:
        base_filename = alt_filename

    # save cleaned version as image
    hdulist = fits.open(base_filename) # read original image for base file
    hdu = hdulist[0] # open main compartment
    hdu.data = filled_mask # set main compartment data component to be the final image
    hdu.header['T_FILLED'] = True # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_5sig.fits', clobber=True) # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename) # read original image for base file
    hdu = hdulist[0] # open main compartment
    hdu.data = trimmed_mask.astype(int) # set main compartment data component to be the final image
    hdu.header['T_MASK'] = True # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_mask_5sig.fits', clobber=True) # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename) # read original image for base file
    hdu = hdulist[0] # open main compartment
    hdu.data = bkg_mask # set main compartment data component to be the final image
    hdu.header['T_BKG'] = True # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_bkg_5sig.fits', clobber=True) # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename) # read original image for base file
    hdu = hdulist[0] # open main compartment
    hdu.data = col_mask # set main compartment data component to be the final image
    hdu.header['T_COL'] = True # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_color_5sig.fits', clobber=True) # write to file, may overwrite


    ###

    # 3 sig mode
    pre_instruct['sig_n'] = 3
    filled_mask, trimmed_mask, bkg_mask, col_mask = thresholder(stacked_im, color_im, pre_instruct)

    # save cleaned version as image
    hdulist = fits.open(base_filename)  # read original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = filled_mask  # set main compartment data component to be the final image
    hdu.header['T_FILLED'] = True  # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_3sig.fits', clobber=True)  # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename)  # read original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = trimmed_mask.astype(int)  # set main compartment data component to be the final image
    hdu.header['T_MASK'] = True  # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_mask_3sig.fits', clobber=True)  # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename)  # read original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = bkg_mask  # set main compartment data component to be the final image
    hdu.header['T_BKG'] = True  # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_bkg_3sig.fits', clobber=True)  # write to file, may overwrite

    # save cleaned version as image
    hdulist = fits.open(base_filename)  # read original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = col_mask  # set main compartment data component to be the final image
    hdu.header['T_COL'] = True  # add line in header
    hdu.writeto(write_dir + table_id + '_threshold_color_3sig.fits', clobber=True)  # write to file, may overwrite

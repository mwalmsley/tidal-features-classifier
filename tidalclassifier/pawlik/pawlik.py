import matplotlib
import pandas as pd

matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import numpy as np
from astropy.io import fits
from custom_image_utils import scaled_plot
from matplotlib import pyplot as plt
import scipy.spatial.distance
import skimage.transform

# recreate the methods of Pawlik et al 2016
# https://arxiv.org/pdf/1512.02000v2.pdf


# binary mask created from 8-connected pixels in background-subtracted image at > 1 sig above bkg

# Rmax is distance between 'center' (brightest pixel in mask) and most distant pixel



def findMax(im):
    # print(im.shape)
    center_flat = np.argmax(im)
    x_c, y_c = np.unravel_index(center_flat, im.shape)
    return x_c, y_c

def findMin(im):
    # print(im.shape)
    center_flat = np.argmin(im)
    x_c, y_c = np.unravel_index(center_flat, im.shape)
    return x_c, y_c

def normMatrix(im):
    norm_matrix = np.zeros((im_height,im_width))
    x_c, y_c = findMax(im)
    # print(x_c, y_c)
    for x in range(im_height):
        for y in range(im_width):
            x_dist = x - x_c
            y_dist = y - y_c
            norm_matrix[x,y] = np.linalg.norm((x_dist, y_dist))
    # print(norm_matrix[126:132,126:132]) # verify center
    return norm_matrix

def findRmax(im, mask):
    norm_matrix = normMatrix(im) * mask
    Rmax = norm_matrix.max()
    # print(Rmax) # max distance in pixels between maxima and furthest mask pixel
    return Rmax

def cmask(index,radius,array):
  a,b = index
  nx,ny = array.shape
  y,x = np.ogrid[-a:nx-a,-b:ny-b]
  mask = x*x + y*y <= radius*radius
  return(mask)

def estimateCenter(im):
    x_len, y_len = im.shape
    x_c, y_c = int(x_len / 2), int(y_len/2)
    # print(x_len)
    # x_rot_c, y_rot_c = findMax(im) # this causes errors in images with bright interlopers
    # maximum in central 50 pixel box for initial guess
    box_width = 25 # x2
    box_mask = np.zeros_like(im)
    box_mask[x_c - box_width:x_c + box_width, y_c - box_width:y_c + box_width] += 1
    box_masked_im = im * box_mask
    # plt.imshow(np.sqrt(box_masked_im))
    # plt.show()
    x_rot_c, y_rot_c = findMax(box_masked_im)
    return x_rot_c, y_rot_c

def findResidual(im, mask, rot_center_shift, mode='standard'):
    x_rot_c, y_rot_c = estimateCenter(im)
    # print(x_rot_c, y_rot_c)
    mask_center = (x_rot_c, y_rot_c)
    Rmax = findRmax(im, mask)
    if mode == 'mask': im = mask
    im = im * cmask(mask_center, Rmax, im) # circular mask of radius Rmax
    # plt.imshow(im)
    # plt.show()
    # if rot_center_shift == None: rot_center_shift = (x_len/2-x_c, y_len/2-y_c) # calculate rot center if not provided
    tform_c = skimage.transform.SimilarityTransform(translation=rot_center_shift) # radians
    im = skimage.transform.warp(im, inverse_map=tform_c, preserve_range=True) # re-center
    im_r = skimage.transform.rotate(im, 180, preserve_range=True) # rotated image, degrees
    im_res = np.abs(im - im_r)
    # plt.imshow(im_res)
    # plt.show()
    return im_res

def findAsymmetry(im, mask):
    x_len, y_len = im.shape
    x_c, y_c = findMax(im)
    # print(x_c, y_c)
    # find minimum A asymmetry, and corresponding rotation center

    # rot_center_shift = (x_len/2-x_c, y_len/2-y_c) # default shift was center
    rot_center_shift = (0,0) # no shift, already start from brightest central pixel!

    A = np.ones((max_path*2+1,max_path*2+1))
    x_tweak = np.arange(-max_path, max_path+step, step)
    y_tweak = np.arange(-max_path, max_path+step, step)

    hunt = True
    curr_x = max_path # start at center of 0 to 4 grid, current x
    curr_y = max_path # similarly
    while hunt:
        # measure A at 8-connected pixels
        for alt_x in [curr_x - 1, curr_x, curr_x + 1]:
            for alt_y in [curr_y - 1, curr_y, curr_y + 1]:
                if  A[alt_x, alt_y] == 1:
                    final_rot_center_shift = (
                    rot_center_shift[0] + x_tweak[alt_x],
                    rot_center_shift[1] + y_tweak[alt_y]
                    )
                    im_res = findResidual(im, mask, final_rot_center_shift, mode='standard')
                    A[alt_x, alt_y] = np.sum(im_res) / (2 * np.sum(im))
        # print(A)
        # print('\n')
        min_x, min_y = findMin(A)
        # print(min_x, min_y)
        if min_x == curr_x and min_y == curr_y: hunt=False # stop if no change
        if min_x == 0: hunt = False # stop if at left edge
        if min_x == max_path*2: hunt = False # right edge
        if min_y == 0: hunt = False # top edge
        if min_y == max_path*2: hunt = False # bottom edge
        curr_x, curr_y = min_x, min_y

    # measure mask asymmetry about that rotation center
    best_x, best_y = findMin(A)
    best_rot_center_shift = (rot_center_shift[0] + x_tweak[best_x] , rot_center_shift[1] + y_tweak[best_y])
    im_res_mask = findResidual(im, mask, rot_center_shift=best_rot_center_shift, mode='mask')
    A_mask = np.sum(im_res_mask) / (2 * np.sum(mask))
    # plot best residual
    # best_im_res= findResidual(im, mask, rot_center_shift=best_rot_center_shift, mode='standard')
    # plt.imshow(best_im_res)
    # plt.show()
    # print(best_rot_center_shift, A.min())
    return A.min(), A_mask

def replicatePawlik():

    if aws == False:
        directory = r'/media/mike/SandiskLinux/threshold/threshold/'
        meta = pd.read_csv('tables/meta_table_saved.csv')[39:54]
        meta_full = pd.read_csv('tables/meta_table_saved.csv')[39:54]
    if aws == True:
        directory = r'/exports/aws/scratch/s1220970/regenerated/512/'
        meta = pd.read_csv('SemesterOne/meta_table.csv')
        meta_full = pd.read_csv('SemesterOne/meta_table.csv')

    meta_full['standard_A'] = np.zeros(len(meta_full), dtype=float)
    meta_full['mask_A'] = np.zeros(len(meta_full), dtype=float)

    print(meta.head())
    # _threshold is the filled mask, _threshold_mask is the binary mask only

    standard_A_0 = []
    mask_A_0 = []

    standard_A_4 = []
    mask_A_4 = []


    print(len(meta))
    for meta_index in range(len(meta)):
        # print(meta_index)
        im = fits.getdata(
            directory + meta.iloc[meta_index]['threshold_5sig_filename'])  # read (the first) original image for base file
        im = np.squeeze(im)

        mask = fits.getdata(
            directory + meta.iloc[meta_index][
                'threshold_mask_5sig_filename'])  # read (the first) original image for base file
        mask = np.squeeze(mask)

        # scaled_plot(im, plt, clip_q=True, rel_clip=False)
        # plt.show()

        min_A, mask_A = findAsymmetry(im, mask)
        print(min_A, mask_A)
        # print(min_A + ' , ' + mask_A)

        meta_full = meta_full.set_value(meta_index, 'standard_A', float(min_A))
        meta_full = meta_full.set_value(meta_index, 'mask_A', float(mask_A))


        # if meta.iloc[meta_index]['FEAT'] != 'N':
        #     standard_A_4.append(min_A)
        #     mask_A_4.append(mask_A)
        # else:
        #     standard_A_0.append(min_A)
        #     mask_A_0.append((mask_A))

        if aws:meta_full.to_csv('meta_with_A_full_eddie_5sig.csv')
        else: meta_full.to_csv('meta_with_A_full_local.csv')


# aws=True
# im_height = 512
# im_width = 512

aws = True
im_height = 512
im_width = 512

# max_path = 20
# step = 0.5

max_path = 8
step = 0.5

replicatePawlik()




# plt.scatter(standard_A_4, mask_A_4, color='r')
# plt.scatter(standard_A_0, mask_A_0, color='k')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.legend(['tidal', 'non'],loc=0)
# plt.savefig('pawlik_scatter_1000_3sig.png')
# plt.show()

# Implements path-hopping from Conselice
# Runs on background-subtracted images so does not include -bkg term
# Selects center by maxima: is this smart?
# it could be that the brightest point within the central mask is NOT the galactic center

# the threshold significance level of the mask is quite important in the size of the mask -> including interlopers
# at one sigma, almost all images include a bright interloper.
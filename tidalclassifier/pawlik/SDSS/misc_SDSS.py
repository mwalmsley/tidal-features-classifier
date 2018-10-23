"""
Unknown - I have no idea what key_sel2000 is
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from thresholder import thresholdImage
from astropy.io import fits
from sklearn.neighbors import KernelDensity

# sns.set_context("poster")
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=0.12, right=0.9, bottom=0.12, wspace=0.4)
# sns.set_style("whitegrid")

# meta = pd.read_csv('/home/mike/key_sel2000.csv')

# Run KDE estimate of distribution of galaxy parameter 'P_MG'

# X = table['P_MG'].values.reshape(-1,1)
# print X.shape
# kde = KernelDensity(kernel='gaussian',bandwidth=0.001).fit(X)
# X_plot = np.linspace(0,1,1000).reshape(-1,1)
# log_dens = kde.score_samples(X_plot)
# dens = np.exp(log_dens)
# plt.plot(X_plot,dens)
# plt.xlabel('P_MG')
# plt.ylabel('Num. Galaxies, Kernel Density Estimate')
# plt.ylim([0,50])
# plt.show()

# distribution gap at p=0.008
# otherwise, long tail, 0.05 or 0.1 seem natural cuts

#####

"""
Read SDSS fits files from read_dir using key_sel2000.csv metadata (perhaps the 2k I ran Pawlik asymmetry on?
Apply thresholding to them
Write back out to write_dir
"""


read_dir = r'/exports/eddie/scratch/s1220970/regenerated/SDSS/'
write_dir = read_dir
meta = pd.read_csv(r'/exports/eddie/scratch/s1220970/regenerated/SDSS/key_sel2000.csv')
for meta_i in range(len(meta)):
    filename = write_dir + meta.iloc[meta_i]['fits_name']
    im = fits.getdata(filename) # im is (width, height)
    stacked_im = np.array([im])  # stacked image is (1,width,height)
    print(stacked_im.shape)
    color_im = np.array([im,im,im])
    print(color_im.shape)
    table_id = 'ID_' + str(meta_i) # forces string interpretation later in pandas

    # threshold process requires parameter choices
    pre_instruct = {}
    # pre_instruct['sig_n'] = 5
    pre_instruct['mode'] = 'only_central'
    pre_instruct['dilation_radius'] = 10  # sx, sy

    thresholdImage(stacked_im, color_im, table_id, pre_instruct, read_dir, write_dir, alt_filename=filename)


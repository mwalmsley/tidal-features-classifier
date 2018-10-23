import numpy as np
import threading
import json
import os
import sys
from astropy.io import fits
from sklearn.metrics import accuracy_score, log_loss


def contains_any(seq, aset):
    """ Check whether sequence seq contains ANY of the items in aset. """
    for c in seq:
        if c in aset:
            return True
    return False


def str_to_N(str):
    if contains_any(str, ['A', 'F', 'H', 'L', 'M', 'S']):
        return str.replace(' ', '')
    else:
        return 'N'


def check_rounding(a_el, MinClip):
    if (a_el / MinClip == np.floor(a_el / MinClip)):
        # slightly increase (proportional to MinClip) to avoid rounding error
        a_el += MinClip * 0.01
    return a_el
checkRounding = np.vectorize(check_rounding)


def step_round(a, MinClip):
    a = check_rounding(a, MinClip)
    return np.around(a / MinClip) * MinClip


class ThreadsafeIter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def shuffle_df(df):
    return df.sample(frac=1).reset_index(drop=True)


def load_lines(fname):
    with open(fname) as f:
        content = f.readlines()
    return np.array(content)


def to_json(x, filename):
    with open(filename, 'w') as f:
        json.dump(x, f)


def from_json(filename):
    with open(filename) as f:
        x = json.load(f)
    return x


def remove_file(fname):
    try:
        os.remove(fname)
    except OSError:
        pass


def write_list_to_file(fname, list):
    remove_file(fname)  # delete if it already exists
    with open(fname, 'a') as f:  # Open for writing.  The file is created if it does not exist.
        for item in list:
            f.write("%s\n" % item)


def read_task_id():
    try:
        task_id = int(sys.argv[1])
        return task_id
    except KeyError:
        print("Error: could not read input file name from task id")
        exit(1)


def write_fits(img, read_directory, write_directory, savename):
    hdulist = fits.open(read_directory + 'W1-2_threshold.fits')  # read original image for base file
    hdu = hdulist[0]  # open main compartment
    hdu.data = img  # set main compartment data component to be the final image
    hdu.writeto(write_directory + savename, clobber=True)  # write to file, may overwrite
    print('wrote to ', write_directory + savename)


def calculate_metrics(y, y_score):
    y_int = np.around(y).astype(int)
    y_score_int = np.around(y_score).astype(int)
    acc = accuracy_score(y_int, y_score_int)
    loss = log_loss(y_int, y_score_int)
    return acc, loss

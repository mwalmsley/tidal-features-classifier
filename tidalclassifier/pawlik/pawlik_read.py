import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_lines(fname):
    with open(fname) as f:
        content = f.readlines() # list of lines
    return content


# def interpret_lines(lines):
#     n_of_calcs = len(lines)/3
#     first_line = 0
#     meta_full = pd.read_csv('tables/meta_table_with_A_full_eddie.csv')
#     while first_line < n_of_calcs:
#         # print(first_line)
#         meta_index = np.array(lines[first_line], dtype=int)
#         # print(meta_index)
#         # x_c, y_c = np.array(lines[first_line+1])
#         # print(lines[first_line+2])
#         A = np.fromstring(lines[first_line+2][1:-1], sep=',',count=2)
#         # print(A)
#         standard_A, mask_A = A
#         standard_A = float(standard_A)
#         mask_A = float(mask_A)
#         print(mask_A)
#
#         meta_full.set_value(int(meta_index), 'standard_A', standard_A)
#         meta_full.set_value(int(meta_index), 'mask_A', mask_A)
#
#         first_line = first_line + 3
#
#
#     meta_full.to_csv('/home/mike/meta_with_A.csv')

def interpret_lines(lines):
    translations = []
    picture_ids = []
    for line in lines:
        # print(line)
        if line[0] == '(':
            # translation = np.fromstring(line)
            translation = np.fromstring(line[1:-1], sep=',',count=2)
            # print(translation)
            translations.append(translation)
        else:
            # print(line)
            picture_id = np.array(line,dtype=int)
            picture_ids.append(picture_id)
    # print(len(picture_ids), len(translations))
    picture_ids=picture_ids[:len(translations)] # there's a trailing id
    if len(picture_ids) != len(translations): print('reading error on length')
    translations = np.array(translations)
    print(translations)
    standard_A = translations[:,0]
    mask_A = translations[:, 1]
    picture_ids = np.array(picture_ids)
    print(standard_A.shape)
    return {'standard_A': standard_A, 'mask_A': mask_A, 'picture_ids': picture_ids}

def load_pawlik(fname):
    lines = load_lines(fname)
    interpret_lines(lines)
# load_pawlik('/home/mike/pawlik.txt')

# first line is index
# second line is tuple of centroid coordinates (x, y)
# third line is tuple of (standard asym, mask asym)


#
# content = load_lines(r'/home/mike/sym.txt')
# data = interpret_lines(content)
# meta_full = pd.read_csv(r'/home/mike/meta_with_A_full_eddie.csv')
# for index in range(len(data['picture_ids'])):
#     # if data['picture_ids'][index] != meta_full['picture_id'][index]: print('pic id error',data['picture_ids'][index],meta_full['picture_id'][index])
#     meta_full.set_value(index, 'standard_A', data['standard_A'][index])
#     meta_full.set_value(index, 'mask_A', data['mask_A'][index])
# meta_full.to_csv('meta_with_A_full_eddie_complete.csv')

# read_pawlik(fname='meta_with_A_full_eddie_complete.csv')
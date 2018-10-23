import pandas as pd
import numpy as np


def clean_url_list(df, file_str):
    """
    Remove urls that aren't images
    No effect if url_list already cleaned
    
    Args:
        df (pd.DataFrame): [description]
        file_str ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    # TODO: url must have been added as header
    df = df[df.url.str.contains('mosaic') != True]
    df = df[df.url.str.contains('mask') != True]
    df = df[df.url.str.contains('weight') != True]
    df = df[df.url.str.contains('flag') != True]
    df = df[df.url.str.contains('sum') != True]
    # keep only g, r, i bands (y is i, renamed after filter broke)
    df = df[(df.url.str.contains('_g') | df.url.str.contains('_r') | df.url.str.contains('_y') | df.url.str.contains('_i'))]
    # save for review
    df.to_csv(file_str, index=False)
    return df


def translate_url(url):
    t = url[77:]
    # cut out %??
    t = t.replace('W', '__W')
    t = t.replace('_g', '__g')
    t= t.replace('_r', '__r')
    t = t.replace('_y', '__y')
    t = t.replace('_i', '__i')
    t = t.replace('fits%5B', '__')
    t = t.replace('%3A', '__')
    t = t.replace('%2C', '__')
    t = t.replace('%5D', '__')
    t = t.replace('.V2.2A.swarp.cut.', '')
    return t


def restrict_sizes(df):
    df = df.query('x_width > 100')
    df = df.query('y_width > 100')
    df = df['url']
    df.to_csv('/home/mike/MPhys_Code/Semester One/tables/url_list_full.txt', index=False)
    # index = 0
    # drop_list = []
    # while index < len(df):
    #     # print(index, len(df))
    #     if df.iloc[index]['x_width'] < 100:
    #         drop_list.append(index)
    #     if df.iloc[index]['y_width'] < 100:
    #         drop_list.append(index)
    #     index += 1
    # df.drop(drop_list, inplace=True)

def identical_pix(row1, row2):
    # comment
    to_match = ['xloc_min', 'yloc_min', 'xloc_max', 'yloc_max']
    matches = [row1[col] == row2[col] for col in to_match]
    if all(matches):
        return True
    return False

def similar_pix(row1, row2, threshold=30):
    # require different pointing (so multi-band is not duplicate)
    # if row1['field'] != row2['field']:
        # if either x pix match or y pix match
    x_match = ['xloc_min', 'xloc_max']
    y_match = ['yloc_min', 'yloc_max']
    for dim_match in [x_match, y_match]:
        matches = [np.abs(row1[col] - row2[col]) < threshold for col in dim_match]
        if all(matches):
            return True
    return False


def counter(start=int(0)):
    while True:
        yield start
        start += 1

# skipping the final row (special case, no next row)

def matching_pairs(pairs, left_val, right_val):
    for pair in pairs:
        if (left_val, right_val) == pair:
            return True
    return False


def label_consecutive_rows_by_conditional(df,condition_func,output_col, row_batch=12, unique_pointing=False):
    df[output_col] = np.ones(len(df)) * -1
    id_gen = counter()
    for indx in range(len(df)):
        if indx + row_batch > len(df) - 1:
            # if near end of df, reduce row_batch to avoid checking off the edge of the df
            # final index value is len(df) - 1 as length is not 0 indexed
            # row batch should be 0 at this final value (therefore no while loop triggered)
            row_batch = len(df) - indx - 1
        # if no id yet (i.e. if not yet matched)
        if df.iloc[indx][output_col] < 0:
            # will assign new unique ID
            # gen_val = next(id_gen)
            # print('before',df.iloc[indx])
            df.set_value(indx, output_col, next(id_gen))
            # print('after', df.iloc[indx])
            # new record of bands and fields for new id
            fb_pairs = []
        # for next row batch, until we've looked that far ahead/behind:
        rows_ahead = 0
        curr_row_cp = df.iloc[indx]
        fb_pairs.append( (curr_row_cp['field'], curr_row_cp['band']) )
        field_band_pairs=[]
        dummy_row_batch = row_batch
        while np.abs(rows_ahead) < dummy_row_batch:
            broken = False
            rows_ahead += 1
            next_row_cp = df.iloc[indx + rows_ahead]
            # if this row and row in next (batch) row(s) have same pixels
            if condition_func(curr_row_cp, next_row_cp):
                if unique_pointing:
                    # if any([existing_field==next_row_cp['field']for existing_field in fields[next_row_cp['band']]]):
                    if matching_pairs(fb_pairs, next_row_cp['field'], next_row_cp['band']):
                        break # pointing is already used, must have reached new index
                    # fields[next_row_cp['band']].append(next_row_cp['field'])
                    fb_pairs.append( (next_row_cp['field'], next_row_cp['band']) )
                # give next row the id of this row
                df.set_value(indx + rows_ahead, output_col, curr_row_cp[output_col])
            else:
                break
    return df


if __name__ == '__main__':

    directory = '/home/mike/MPhys_Code/Semester One/tables/'
    url_str = directory + 'url_list_full.txt'

    # load, clean
    url_list = pd.read_csv(url_str)
    url_list = clean_url_list(url_list, url_str)

    # add column to url_list by applying url_to_filename to every element in url_list['url']
    url_list['url_translated'] = url_list['url'].map(translate_url)
    url_list['field'] = url_list['url_translated'].map(lambda x: x.split('__')[1])
    url_list['band'] = url_list['url_translated'].map(lambda x: x.split('__')[2])
    url_list['xloc_min'] = url_list['url_translated'].map(lambda x: float(x.split('__')[3]))
    url_list['xloc_max'] = url_list['url_translated'].map(lambda x: float(x.split('__')[4]))
    url_list['yloc_min'] = url_list['url_translated'].map(lambda x: float(x.split('__')[5]))
    url_list['yloc_max'] = url_list['url_translated'].map(lambda x: float(x.split('__')[6]))
    del url_list['url_translated']
    url_list['band'] = url_list['band'].map(lambda x: x.replace('y', 'i'))
    url_list['x_width'] = url_list['xloc_max'] - url_list['xloc_min']
    url_list['y_width'] = url_list['yloc_max'] - url_list['yloc_min']

    # restrict_sizes(url_list)

    # print(url_list[:50])

    # url_list = url_list_new2
    # print(url_list[:50])

    url_list = label_consecutive_rows_by_conditional(url_list, identical_pix,'image_id')
    url_list = label_consecutive_rows_by_conditional(url_list, similar_pix,'duplicate_id', unique_pointing=True)

    # validation checks

    # def any(iterable):
    #     el_num = counter()
    #     for element in iterable:
    #         print(next(el_num))
    #         if element:
    #             print(element)
    #             return True
    #     return False

    #
    # if any([value!=3 for value in url_list['image_id'].value_counts(sort=False).values]) :
    #     print([value!=3 for value in url_list['image_id'].value_counts(sort=False).values])
    #     print('error')

    # print(url_list['image_id'].value_counts())
    # print(url_list['duplicate_id'].value_counts())

    # url_list[url_list.x_width==4.0]['url'].to_csv('temp.csv')

    # print(url_list['xloc_min'].value_counts())
    # print(url_list['band'].value_counts())

    # remove images with wrong dimensions

    # identify unique images by pixel and pointing

    # check exactly 3 of each
    # if 4, remove 'i' band in favor of 'y' band

    # for each band
    #   if the next url has an x or y min/max within n pix from the current, and dif pointing: duplicates detected


    # # -1 indicates not a mulitihit, hit index starts from 0
    # url_list['multihit'] = -1 * np.ones(len(url_list))
    # # for band in ['g', 'r', 'i']:
    # multihit_indx = 0
    # for indx in range(len(url_list)):
    #     curr_row = url_list.iloc[indx]
    #     for check_indx in range(len(url_list)):
    #         if check_indx != indx:
    #             check_row = url_list.iloc[check_indx]
    #             if curr_row['field'] != check_row['field'] and curr_row['band'] == check_row['band']:
    #                 if np.abs(curr_row['xloc_min'] - check_row['xloc_min']) < 10 or np.abs(
    #                                 curr_row['yloc_min'] - check_row['yloc_min']) < 10:
    #                     # current and next row are both duplicates
    #                     url_list.set_value(indx, 'multihit', multihit_indx)
    #                     url_list.set_value(check_indx, 'multihit', multihit_indx)
    #                     print(multihit_indx)
    #     multihit_indx += 1


    # print(url_list.head())

    url_list.to_csv('temp.csv')


    # if duplicates
    #   remove entries without the expected dimensions
    #   check at least one image left
    #   pick the first remaining url and remove the rest

    # check table length

    # musings:
    # add into pro-processing a symmetry-measuring proesss
    # e.g. find ideal axis of symmmetry for inner circle and then either subtract off or enhance to hihglight asymmetry
    # see tidal papers for other inspiration
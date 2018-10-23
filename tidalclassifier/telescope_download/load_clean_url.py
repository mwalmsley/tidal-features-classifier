import pandas as pd
import numpy as np



def clean_url_list(df, file_str):
    # # delete urls that are not gr(y)i band and re-save to same file
    # no effect if url_list already cleaned
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
    # replace alleged Y-band with i
    t = t.replace('y', 'i')
    t = t.replace('_i', '__i')
    t = t.replace('fits%5B', '__')
    t = t.replace('%3A', '__')
    t = t.replace('%2C', '__')
    t = t.replace('%5D', '__')
    t = t.replace('.V2.2A.swarp.cut.', '')
    return t



def load_url(url_str, parsed_loc=None):
    url_list = pd.read_csv(url_str) # read in complete raw list of urls (may already be cleaned)
    url_list = clean_url_list(url_list, url_str) # delete urls of non-gri(y) band images

    # add 'url_translated' column to url_list by applying url_to_filename to every element in url_list['url']
    # 'url_translated' can then be split on '__' to fill out specific columns
    url_list['url_translated'] = url_list['url'].map(translate_url)
    url_list['field'] = url_list['url_translated'].map(lambda x: x.split('__')[1])
    url_list['band'] = url_list['url_translated'].map(lambda x: x.split('__')[2])
    url_list['xloc_min'] = url_list['url_translated'].map(lambda x: float(x.split('__')[3]))
    url_list['xloc_max'] = url_list['url_translated'].map(lambda x: float(x.split('__')[4]))
    url_list['yloc_min'] = url_list['url_translated'].map(lambda x: float(x.split('__')[5]))
    url_list['yloc_max'] = url_list['url_translated'].map(lambda x: float(x.split('__')[6]))
    del url_list['url_translated']

    url_list['xloc_c'] = url_list['xloc_min'] + (url_list['xloc_max'] - url_list['xloc_min']) / 2.
    url_list['yloc_c'] = url_list['yloc_min'] + (url_list['yloc_max'] - url_list['yloc_min']) / 2.

    url_list['x_width'] = url_list['xloc_max'] - url_list['xloc_min']
    url_list['y_width'] = url_list['yloc_max'] - url_list['yloc_min']

    # line index is the line of the url in url_text.txt, after deleting other bands
    url_list['line_index'] = np.arange(len(url_list))
    url_list.set_index('line_index', inplace=True)

    if parsed_loc is not None:
        url_list.to_csv(parsed_loc)

    return url_list
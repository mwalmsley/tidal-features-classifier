"""Neatly load and clean Atkinson Table 4 (expert labels and ra/dec)
"""
import pandas as pd
import numpy as np

from tidalclassifier.utils.helper_funcs import str_to_N


# misc. prep work
def clean_table(df, file_str):
    # ensure FEAT is a string, replace '" with N, remove ','
    df['FEAT'] = df['FEAT'].map(str)
    # in making the table, treat CONF as a string. Useful for output.
    df['CONF'] = df['CONF'].map(str)
    df['FEAT'] = df['FEAT'].map(lambda feat_str: feat_str.replace(',', ''))
    df['FEAT'] = df['FEAT'].map(lambda feat_str: str_to_N(feat_str))
    df['ID'] = df['ID'].map(lambda x: x.replace(' ', ''))
    df.to_csv(file_str, index=False, sep='\t')
    # df['table4_index'] = np.arange(len(df))
    # df.set_index('table4_index', inplace=True)

    # atk table index is a simple index that corresponds to picture id in u_and_c list
    df.sort_index(inplace=True)
    df['line_index'] = np.arange(len(df))
    df.set_index('line_index', inplace=True)
    return df

def load_table(table_str): # includes clean
    # table = pd.read_csv(table_str, sep='\s+')
    table = pd.read_csv(table_str, sep='\t')
    table = clean_table(table, table_str)
    return table
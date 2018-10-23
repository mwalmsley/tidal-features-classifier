"""
Previously, query cutout service with a batch of 200ish ra/dec pairs.
Save the resulting page
This script translates the resulting page into a meaningful catalog,
where 'picture_id' and 'raw_url_id' can be used to relate the single band image urls back to the ra/dec pairs queried
"""
import numpy as np
import pandas as pd

def load_cutout(cutout_html_loc, parsed_loc=None):
    """Parse html source of cutout service reported matches into urls with picture and url ids
    
    Args:
        cutout_html_loc (str): html source of cutout service reported matches 
    
    Returns:
        [type]: [description]
    """

    # cutout_str is the text file of the cutout service page source
    with open(cutout_html_loc) as f:
        content = f.readlines() # read in as a list of strings

    translated_list = [match_num(line) for line in content] # convert to either a number of matches or 'not helpful'
    final_list = remove_values_from_list(translated_list, 'not_helpful') # remove the 'not helpful' lines
    # final_list is now an ordered list of numbers of matches for each url e.g. 11211141112
    # note that the index of final list is NOT the id of current url since final_list(l) has no double matches
    # hence we need picture_id: a list of object identifiers that matches on index with url_list

    # print(len(final_list))
    # print('\n')
    # for num in range(13):
    #     print(num, len(select_values_from_list(final_list, num)))

    # lists are passed by reference to functions!
    raw_url_id = np.array(generate_raw_url_id_series(np.array(final_list)))
    picture_id = np.array(generate_picture_id_series(final_list))

    # for index in range(len(raw_url_id)):
    # for index in range(50):
    #     print(raw_url_id[index], picture_id[index])

    # combine into dataframe and return
    raw_url_id = pd.Series(raw_url_id)
    picture_id = pd.Series(picture_id)
    data = {'raw_url_id': raw_url_id, 'picture_id': picture_id}
    cutout_table = pd.DataFrame(data)

    if parsed_loc:
        cutout_table.to_csv(parsed_loc, index=None)

    return cutout_table


# given a line in the cutout table source, translate it into either 'not helpful' or a numeric number of matches
def match_num(line):
    num_list = ['no', 'one match', '2 matches', '3 matches', '4 matches', '5 matches', '6 matches',
                '7 matches', '8 matches', '9 matches', '10 matches', '11 matches', '12 matches']
    output_list = range(13)
    output_dict = dict(zip(num_list, output_list))
    for num in num_list:
        if num in line:
            return output_dict[num]
    return 'not_helpful'

# convenience list functions
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def select_values_from_list(the_list, val):
   return [value for value in the_list if value == val]


def generate_raw_url_id_series(l):
    # l is the ordered list of num. of matches
    raw_url_id = []
    # raw_url_id is simply e.g. 000111222333444555666 regardless of multiple matches

    current_id = 0 # which picture is this (without accounting for multiple matches)
    index = 0 # index of input list
    while index < len(l):
        if l[index] != 0:
            raw_url_id.append(current_id)
            raw_url_id.append(current_id)
            raw_url_id.append(current_id)
            l[index] -= 1
            current_id += 1
        else:
            index += 1
    # print(len(raw_url_id))
    return raw_url_id

def generate_picture_id_series(l):
    # l is the ordered list of num. of matches
    picture_id = [] # this will be the 'picture_id' column in final df
    current_id = 0 # current_id is the index of the list element being manipulated AND the current picture id

    # picture_id is which object the url at that index in url_list corresponds too
    # e.g. 000111222333444444555666 for a double-match on 4th object i.e. lines 12-18 on url_list

    while current_id < len(l):
        if l[current_id] != 0: # remove a match from the list value each time: if not 0, matches remaining
            picture_id.append(current_id) # add the index of which object the current match refers too e.g. 555
            picture_id.append(current_id)
            picture_id.append(current_id)
            l[current_id] -= 1

        else:
            current_id += 1 # move on to next list index
    # print(len(picture_id))
    return picture_id


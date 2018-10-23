"""Beforehand, manually trim Atkinson Table 4 into sets of 200 ra/dec pairs
Query the cutout service with these to retrieve download urls for matching galaxies
(Cannot be more than 200ish due to cutout service maxing out)
Save each batch of matching urls to text files
This utility joins the matching urls back into one huge file

Each rXXX.text file is ta list of the download urls that matched on RA/DEC
These urls can later be parsed to understand which image they match
"""
import os


def combine_files(read_files, write_file):
    with open(write_file, 'w') as writefile:
        writefile.write('url\n')  # overwrite existing file with 'url' header
    with open(write_file, 'a') as writefile:
        for read_file in read_files:
            with open(read_file, 'r') as readfile:
                writefile.write(readfile.read())


def create_url_text(directory, url_loc):
    file_strs = ['r2_200.txt', 'r201_400.txt', 'r401_600.txt', 'r601_800.txt', 'r801_1000.txt',
                 'r1001_1200.txt', 'r1201_1400.txt', 'r1401_1600.txt', 'r1601_1781.txt']
    read_files = [os.path.join(directory, file) for file in file_strs]
    combine_files(read_files, url_loc)

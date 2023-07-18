#!/usr/bin/env python

# A collection of general and simple python toolbox 

# A basic rules for these utilities is that they should be simple and stand
# alone without any other requirements except standard library

import os
import sys
import tempfile
import numpy as np


def cut_lines(infile, start=0, end=-1, savefile=False, outfile=None, debug=False,):
    '''
    run a section of a program in python interative shell

    The line number starts from 0

    Parameters
    ----------
    infile : str
        the filename of the text file or program
    start : int
        the order of start line, start from 1; negative number start from the last
    end : int
        the end order of the file, negative number start from the last
    outfile : str
        the filename of the output
        default: None, write into a temperary file

    Examples
    --------
    
        cut_lines('hello.py', start=1, end=-2, savefile=True, filename='')

    '''
    with open(infile, 'r') as f:
        # lines = f.read().splitlines()
        lines = f.readlines()
    max_line_num = len(lines)
        
    # fixed a zero issue
    if start < 0:
        start = start + max_line_num + 1
    if end < 0:
        # this is to make -1 point to the last line
        end = end + max_line_num + 1
    elif end > max_line_num:
        end = max_line_num
    if start > end:
        raise ValueError('The start must smaller than the end.')
        
    snippets = lines[start: end]

    code_string = ''
    for line in snippets:
        code_string += line

    if debug:
        print(f"length of the lines: {len(snippets)}")
        print('-----start-----')
        print(code_string)
        print('------end-----')
    if savefile:
        if outfile is None:
            # explicit save the file in the /tmp folder
            # outfile = '/tmp/'+'_' + os.path.basename(infile) + '_{}to{}.snippets'.format(start, end)
            # write into a termperary file
            with tempfile.NamedTemporaryFile(delete=False) as f_tmp:
                for line in snippets:
                    f_tmp.write(line.encode('utf-8'))
            outfile = f_tmp.name

        # write into a local file, for reusable
        else:
            with open(outfile, 'w') as f_out:
                for line in snippets:
                    f_out.write(line)
        return outfile
    return code_string

def run_lines(filename, start=0, end=-1, debug=False):
    # execute the selected lines from the file
    exec(cut_lines(filename, start, end, debug=debug))

def processbar(progress, total=1, barlength=80):
    '''A simple Progress Bar
    
    Parameters
    ----------
    progress : float or int
        the progress of the program, it can be float number between [0, 1] with total=1,
        Or, it can be any numbers with manully defined total
    total : float or int
        the total number
        default: 1
    barlength : int
        the length of the bar
        default: 80

    '''
    status = ""
    progress = float(progress) / float(total)
    if progress >= 1:
        progress, status = 1, "\r\n"
    block = int(round(barlength * progress))
    text = "\r[{}] {:.0f}% {}".format("#" * block + "-" * (barlength - block),
                                      round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


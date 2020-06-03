#!/usr/bin/env python

# A simple tools to run a section of a give file
#
# Author: Jianhang Chen
# Email: cjhastro@gmail
#
# Usage: run lines from 9 to 21 of infile.py 
# In ipython, 
#       % run -i run_lines.py infile.py 9 21
# As a package
#       from run_lines import run_lines
#       run_lines(infile.py, 9, 21)
#
# History:
#   2020.6.2: first release

import os
import argparse
import tempfile

version = '0.0.1'

def run_lines(infile, start, end, outfile=None):
    '''
    run a section of a program in python interative shell

    Parameters
    ----------
    infile : str
        the filename of the text file or program
    start : int
        the order of start line, start from 1
    end : int
        the end order of the file 
    outfile : str
        the filename of the output
        default: None, write into a temperary file
    '''

    with open(infile, 'r') as f:
        # lines = f.read().splitlines()
        lines = f.readlines()
        
    # fixed a zero issue
    if start < 1:
        start = 1
    max_line_num = len(lines)
    if end > max_line_num:
        end = max_line_num
        
    snippets = lines[start-1: end]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run snippet from a large file")
    parser.add_argument('infile', type=str, help='input filename')
    parser.add_argument('start', type=int, help='start line index, start from 1')
    parser.add_argument('end', type=int, help='end line index')
    parser.add_argument('--outfile', type=str, help='output filename')

    parser.add_argument('--version', action='version', version=version)

    args = parser.parse_args()

    outfile = run_lines(args.infile, args.start, args.end, args.outfile)
    exec(open(outfile).read())

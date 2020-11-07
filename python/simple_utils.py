#!/usr/bin/env python

# A collection of general and simple python toolbox 

# A basic rules for these utilities is that they should be simple and stand
# alone without any other requirements except standard library

import sys

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
    

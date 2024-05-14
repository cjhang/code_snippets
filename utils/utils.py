#!/usr/bin/env python

# A collection of general and simple python toolbox 

# A basic rules for these utilities is that they should be simple and stand
# alone without any other requirements except standard library

import os
import sys
import tempfile
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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

def demo_wavefunc(t, amp=None, freq=None):
    """Used for plot_slider demonstration 
    """
    return amp * np.sin(2*np.pi*freq*t)

def demo_wavefunc2d(x, y, amp_x=None, amp_y=None, freq_x=None, freq_y=None):
    """Used for image_slider demonstration

    Parameters
    ----------
    x : `numpy.ndarray`
        The x coordinates of the image
    y : `numpy.ndarray`
        The y coordinates of the image
    """
    return amp_x * np.sin(2*np.pi*freq_x*x) + amp_y * np.sin(2*np.pi*freq_y*y)

def plot_slider(func, args, slider_kwargs=None, slider_height=0.1, xlabel=None, ylabel=None, 
                **plot_kwargs,):
    """plot with annimation
    
    Parameters:
    -----------
    func : callable
        The input function, which take args as the input and slider_args
    args : list
        The independent variable or the list of independent variable of the input func
        for func(x,) ==> args = x or
        for func(x, y, z) ==> args=[x, y, z]
    kwargs : dict
        The keyword arguments of the input function
        args = {amp:{'default':5, 'range':[0.1, 2]},
                freq:{'default':3, 'range':[0, 10]}}

    Example:
        
        plot_slider(demo_wavefunc, np.linspace(0,10,100), 
                     {'amp':{'default':1, 'range':[0.1,2]}, 
                     'freq':{'default':0.2,'range':[0.01,2]}},
                     xlabel='Time', ylabel='Amplitude')

    """
    # ge the parameters of the input function
    inargs = inspect.getfullargspec(func)[0]

    # add the canvas
    fig = plt.figure(figsize=(12, 7))
    
    # set the height of the sliders
    if slider_height == 'auto':
        nparam = len(slider_args)
        slider_height = np.min([1.0/nparam, 0.1])
    bottom_pad = 0.87
    
    # generate the slider the for the keywords args of input function
    args_default = {}
    args_slider = {}
    for i,item in enumerate(slider_kwargs.items()):
        key, value = item
        ax_param = fig.add_axes([0.6, bottom_pad-i*slider_height, 0.35, slider_height])
        arg_range = value['range']
        arg_default = value['default']
        ax_slider = Slider(ax=ax_param, label=key, orientation='horizontal', 
                           valinit = arg_default, valmin=arg_range[0], valmax=arg_range[1])
        args_default[key] = arg_default
        args_slider[key] = ax_slider
   
    # plot the main figure
    ax_main = fig.add_axes([0.08, 0.1, 0.45, 0.85])
    line, = ax_main.plot(args, func(args, **args_default), **plot_kwargs)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    # instantly update the canvas
    def update(val):
        kwargs = {}
        for a in slider_kwargs.keys():
            kwargs[a] = args_slider[a].val
        line.set_ydata(func(args, **kwargs))
        fig.canvas.draw_idle()

    for slid in args_slider.values():
        slid.on_changed(update)

    # add the reset bottom
    resetax = fig.add_axes([0.9, 0.025, 0.05, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        for key,sli in args_slider.items():
            sli.reset()
    button.on_clicked(reset)

    # set block=True to hold the plot
    plt.show(block=True)
    # print the final slider keyword values
    final_kwargs = {}
    for a in slider_kwargs.keys():
        final_kwargs[a] = args_slider[a].val
    return final_kwargs

def image_slider(func, args, slider_kwargs=None, slider_height=0.1, xlabel=None, ylabel=None, 
                **image_kwargs,):
    """plot with annimation
    
    Parameters:
    -----------
    func : callable
        The input function, which take args as the input and slider_args
    args : list
        The independent variable or the list of independent variable of the input func
        example: [xmap, ymap]
    kwargs : dict
        The keyword arguments of the input function
        args = {amp:{'default':5, 'range':[0.1, 2]},
                freq:{'default':3, 'range':[0, 10]}}

    Example:
        xmap = np.linspace(0, 5, 100)
        ymap = np.linspace(0, 5, 100)
        xmap, ymap = np.meshgrid(xmap, ymap)
        image_slider(demo_wavefunc2d, [xmap, ymap], 
                     {'amp_x':(1, [0.1,2]), 
                      'freq_x':(0.2, [0.01,2]),
                      'amp_y':(1, [0.1,2]), 
                      'freq_y':(0.2, [0.01,2])},
                     xlabel='x', ylabel='y', origin='lower')

    """
    # add the canvas
    fig = plt.figure(figsize=(12, 7))
    
    # set the height of the sliders
    if slider_height == 'auto':
        nparam = len(slider_args)
        slider_height = np.min([0.5/nparam, 0.1])
    bottom_pad = 0.87
    
    # generate the slider the for the keywords args of input function
    args_default = {}
    args_slider = {}
    for i,item in enumerate(slider_kwargs.items()):
        key, value = item
        ax_param = fig.add_axes([0.65, bottom_pad-i*0.5*slider_height, 0.25, 0.5*slider_height])
        arg_default = value[0]
        arg_range = value[1]
        ax_slider = Slider(ax=ax_param, label=key, orientation='horizontal', 
                           valinit = arg_default, valmin=arg_range[0], valmax=arg_range[1])
        args_default[key] = arg_default
        args_slider[key] = ax_slider
   
    # plot the main figure
    ax_main = fig.add_axes([0.08, 0.1, 0.45, 0.85])
    image = ax_main.imshow(func(*args, **args_default), **image_kwargs)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    # instantly update the canvas
    def update(val):
        kwargs = {}
        for a in slider_kwargs.keys():
            kwargs[a] = args_slider[a].val
        image.set(data=func(*args, **kwargs))
        fig.canvas.draw_idle()

    for slid in args_slider.values():
        slid.on_changed(update)

    # add the reset bottom
    resetax = fig.add_axes([0.9, 0.025, 0.05, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        for key,sli in args_slider.items():
            sli.reset()
    button.on_clicked(reset)

    # set block=True to hold the plot
    plt.show(block=True)
    # print the final slider keyword values
    final_kwargs = {}
    for a in slider_kwargs.keys():
        final_kwargs[a] = args_slider[a].val
    return final_kwargs



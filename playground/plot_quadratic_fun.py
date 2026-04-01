import sys
import numpy as np
sys.path.append('../utils')

from utils import plot_slider 

def quadratic_fun(x, x0=0, a0=1, a1=1, a2=1):
    return a0 + a1*(x-x0) + a2*(x-x0)**2

if __name__ == '__main__':
    plot_slider(quadratic_fun, np.linspace(-10, 10, 200),
                {'x0':{'default':0, 'range':[-10,10]}, 
                 'a0':{'default':1, 'range':[-2,2]},
                 'a1':{'default':1, 'range':[-2,2]},
                 'a2':{'default':-1, 'range':[-3,2]}},
                 xlabel='x', ylabel='y')

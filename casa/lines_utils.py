# a collection of function to help search and identify molecular and ion lines

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
#   2021.07.18, release


from collections import OrderedDict
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const


def print_lines(format='freq', z=0.0):
    '''

    Parameters
    ----------
    format : str
        output format, either in frequency (freq) or wavelength (wave)
    '''
    # CO family from 
    CO_family = OrderedDict([('12C16O_1-0', 115.27120256*u.GHz), ('12C16O_2-1', 230.53800100*u.GHz),
                 ('12C16O_3-2', 345.79599131*u.GHz), ('12C16O_4-3', 461.04076975*u.GHz),
                 ('12C16O_5-4', 576.26793295*u.GHz), ('12C16O_6-5', 691.4730780*u.GHz),
                 ('12C16O_7-6', 806.6518028*u.GHz),  ('12C16O_8-7', 921.7997056*u.GHz),
                 ('12C16O_9-8', 1036.9123861*u.GHz), ('12C16O_10-9', 1151.9854444*u.GHz)])
    CO_family2 = OrderedDict([('13C16O_1-0', 110.20135487*u.GHz), ('13C16O_2-1', 220.39868527*u.GHz),
                               ('13C16O_3-2', 330.58796682*u.GHz), ('13C16O_4-3', 440.76517539*u.GHz)])
    C_ion = OrderedDict([('CI_1-0', 492.16065100*u.GHz),])
    H2O = OrderedDict([('H2O_110-101', 556.93598770*u.GHz), ('H2O_211-202', 752.03314300*u.GHz),
                       ('H2O_422-331', 916.17158000*u.GHz), ('H2O_423-330', 448.00107750*u.GHz),
                       ('H2O_414-321', 380.19735980*u.GHz), ('H2O_532-441', 620.70095490*u.GHz)])

    for line, freq in CO_family.items():
        if 'freq' in format:
            print("{}: {:.4f}GHz".format(line, (freq/(1+z)).to(u.GHz).value))
        elif 'wave' in format:
            print("{}: {:.4f}um".format(line, (const.c/freq).to(u.um).value))

    for line, freq in CO_family2.items():
        if 'freq' in format:
            print("{}: {:.4f}GHz".format(line, (freq/(1+z)).to(u.GHz).value))
        elif 'wave' in format:
            print("{}: {:.4f}um".format(line, (const.c/freq).to(u.um).value))
    for line, freq in C_ion.items():
        if 'freq' in format:
            print("{}: {:.4f}GHz".format(line, (freq/(1+z)).to(u.GHz).value))
        elif 'wave' in format:
            print("{}: {:.4f}um".format(line, (const.c/freq).to(u.um).value))
    for line, freq in H2O.items():
        if 'freq' in format:
            print("{}: {:.4f}GHz".format(line, (freq/(1+z)).to(u.GHz).value))
        elif 'wave' in format:
            print("{}: {:.4f}um".format(line, (const.c/freq).to(u.um).value))
    return


def stacking(dlist, plot=True, norm=True, norm_func=np.mean):
    """
    """
    first_item = dlist[0]
    if isinstance(first_item, str):
        data = np.loadtxt(df)
    elif isinstance(dat, (list, np.ndarray)):
        data = dat
    dtype = type(dat)
    stacked_flux = np.zeros(len(data))

    if dtype == str:
        data_list = []
        for df in dlist:
            if type(df) is not dtype:
                raise ValueError("Data type not consistant!")
            data_list.append(np.loadtxt(df))
    else:
        data_list = dlist
    for data in data_list:
        if norm == True:
            stacked_flux += data[:,1] / norm_func(data[:,1]) / np.std(data[:,1]/norm_func(data[:,1]))
        else:
            stacked_flux += data[:,1]

    if plot:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        for i,data in enumerate(data_list):
            if norm:
                ax.plot(data[:,0], data[:,1]/norm_func(data[:,1]), label="data{}".format(i))
            else:
                ax.plot(data[:,0], data[:,1], label='data{}'.format(i))
        #plot the stacked data
        ax.plot(data[:,0], stacked_flux, label='Stacked')
        plt.show()

                






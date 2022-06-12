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


def print_lines(format='freq', z=0.0, mode='simple'):
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
                              ('13C16O_3-2', 330.58796682*u.GHz), ('13C16O_4-3', 440.76517539*u.GHz),
                              ('13C18O_1-0', 104.711393*u.GHz), ('13C17O_1-0', 107.288945*u.GHz),
                              ('13C17O_2-1', 214.574077*u.GHz)])
    CO_family3 = OrderedDict([('12C17O_1-0_multi', 112.358988*u.GHz), ('12C17O_2-1_multi', 224.714187*u.GHz),
                              ('12C18O_3-2_multi', 337.0609880*u.GHz),
                              ('12C18O_1-0', 109.782176*u.GHz), ('12C18O_2-1', 219.560358*u.GHz),
                              ('12C18O_3-2', 329.330552*u.GHz), ('12C18O_6-5', 658.553278*u.GHz)])
    C_ion = OrderedDict([('CI_1-0', 492.16065100*u.GHz), ('CI_2-1', 809.34197*u.GHz), 
                         ('CII', 1900.5369*u.GHz)])
    H2O = OrderedDict([('H2O_110-101', 556.93598770*u.GHz), ('H2O_211-202', 752.03314300*u.GHz),
                       ('H2O_422-331', 916.17158000*u.GHz), ('H2O_423-330', 448.00107750*u.GHz),
                       ('H2O_414-321', 380.19735980*u.GHz), ('H2O_532-441', 620.70095490*u.GHz)])
    HCN = OrderedDict([('HCN_1-0', 88.63160230*u.GHz), ('HCN_2-1', 177.26111150*u.GHz), 
                       ('HCN_3-2', 265.8864343*u.GHz), ('HCN_4-3', 354.50547790*u.GHz),
                       ('HCN_5-4', 443.1161493*u.GHz)])
    Other = OrderedDict([('CN_1-0_multi', 113.490982*u.GHz), ('CS_2-1', 97.9809533*u.GHz),
                         ('CS_3-2', 146.9690287*u.GHz), ('CS_4-3', 195.9542109*u.GHz),
                         ('CS_5-4', 244.9355565*u.GHz), ('CS_6-5', 293.9120865*u.GHz),
                         ('CS_7-6', 342.8828503*u.GHz), ('CS_8-7', 391.8468898*u.GHz),
                         ('CS_9-8', 440.8032320*u.GHz), ('CS_10-9', 489.7509210*u.GHz)])
    if mode == 'simple':
        family_select = [CO_family, C_ion]
    else:
        family_select = [CO_family, CO_family2, CO_family3, C_ion, H2O, HCN, Other]
    for family in family_select:
        for line, freq in family.items():
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
        data = np.loadtxt(first_item)
    elif isinstance(dat, (list, np.ndarray)):
        data = dat
    dtype = type(first_item)
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
                ax.step(data[:,0], data[:,1]/norm_func(data[:,1]), 
                        label="data{}".format(i), where='mid')
            else:
                ax.step(data[:,0], data[:,1], label='data{}'.format(i), where='mid')
        #plot the stacked data
        if norm:
            ax.step(data[:,0], stacked_flux/norm_func(stacked_flux), color='black',
                    label='Stacked', where='mid', lw=4, alpha=0.8)
        else:
            ax.step(data[:,0], stacked_flux, label='Stacked', where='mid', lw=4, color='black', alpha=0.8)
        plt.legend()
        plt.show()

                






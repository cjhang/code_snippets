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
    CO_13C = OrderedDict([('13C16O_1-0', 110.20135487*u.GHz), ('13C16O_2-1', 220.39868527*u.GHz),
                              ('13C16O_3-2', 330.58796682*u.GHz), ('13C16O_4-3', 440.76517539*u.GHz),
                              ('13C16O_5-4', 550.92628510*u.GHz), ('13C16O_6-5', 661.06727660*u.GHz),
                              ('13C16O_7-6', 771.18412500*u.GHz), ('13C16O_8-7', 881.27280800*u.GHz)])
                              # ('13C18O_1-0', 104.711393*u.GHz), ('13C17O_1-0', 107.288945*u.GHz),
                              # ('13C17O_2-1', 214.574077*u.GHz)])
    CO_17O = OrderedDict([('12C17O_1-0_multi', 112.358988*u.GHz), ('12C17O_2-1_multi', 224.714187*u.GHz),])

    CO_18O = OrderedDict([('12C18O_1-0', 109.78217340*u.GHz), ('12C18O_2-1', 219.56035410*u.GHz),
                          ('12C18O_3-2', 329.33055250*u.GHz), ('12C18O_4-3', 439.08876580*u.GHz),
                          ('12C18O_5-4', 548.83100550*u.GHz), ('12C18O_6-5', 658.55327820*u.GHz),
                          ('12C18O_7-6', 768.25159330*u.GHz), ('12C18O_8-7', 877.92195530*u.GHz)])
    C_ion = OrderedDict([('CI_1-0', 492.16065100*u.GHz), ('CI_2-1', 809.34197*u.GHz), 
                         ('CII', 1900.5369*u.GHz)])
    H2O = OrderedDict([
                       ('H2O_414-321', 380.19735980*u.GHz), 
                       ('H2O_423-330', 448.00107750*u.GHz),
                       ('H2O_110-101', 556.93598770*u.GHz), 
                       ('H2O_532-441', 620.70095490*u.GHz),
                       ('H2O_211-202', 752.03314300*u.GHz),
                       ('H2O_422-331', 916.17158000*u.GHz), 
                       ('H2O_202-111', 987.92675900*u.GHz),
                       ('H2O_312-303', 1097.36479000*u.GHz),
                       ('H2O_111-000', 1113.34300700*u.GHz),
                       ('H2O_321-312', 1162.91160200*u.GHz),
                       ('H2O_422-413', 1207.63873000*u.GHz),
                       ('H2O_220-211', 1228.78871900*u.GHz),
                       ('H2O_523-514', 1410.61806900*u.GHz),
                       ('H2O_413-404', 1602.21936900*u.GHz),
                       ('H2O_221-212', 1661.00763700*u.GHz), 
                       ('*H2O_212-101', 1669.90477500*u.GHz),
                       ('*H2O_303-212', 1716.76963300*u.GHz),
                       ('H2O_532-523', 1867.74859400*u.GHz),
                       ('H2O_322-313', 1919.35953100*u.GHz),
                       ('H2O_431-422', 2040.47681000*u.GHz),
                       ('H2O_413_322', 2074.43230500*u.GHz),
                       ('*H2O_313_202', 2164.13198000*u.GHz),
                       ('H2O_330-321', 2196.34575600*u.GHz),
                       ('H2O_514-505', 2221.75050000*u.GHz),
                       ('*H2O_423-414', 2264.14965000*u.GHz),
                       ('H2O_725-716', 2344.25033500*u.GHz),
                       ('H2O_331-322', 2365.89965900*u.GHz),
                       ('*H2O_404-313', 2391.57262800*u.GHz),
                       ])
    HCN = OrderedDict([('HCN_1-0', 88.63160230*u.GHz), ('HCN_2-1', 177.26111150*u.GHz), 
                       ('HCN_3-2', 265.8864343*u.GHz), ('HCN_4-3', 354.50547790*u.GHz),
                       ('HCN_5-4', 443.1161493*u.GHz)])
    Other = OrderedDict([('CN_1-0_multi', 113.490982*u.GHz), ('CS_2-1', 97.9809533*u.GHz),
                         ('CS_3-2', 146.9690287*u.GHz), ('CS_4-3', 195.9542109*u.GHz),
                         ('CS_5-4', 244.9355565*u.GHz), ('CS_6-5', 293.9120865*u.GHz),
                         ('CS_7-6', 342.8828503*u.GHz), ('CS_8-7', 391.8468898*u.GHz),
                         ('CS_9-8', 440.8032320*u.GHz), ('CS_10-9', 489.7509210*u.GHz)])
    Special = OrderedDict([('CH+', 835.08*u.GHz),])
    if mode == 'simple':
        family_select = [CO_family, C_ion]
    elif mode == 'water':
        family_select = [H2O]
    elif mode == 'all' or mode=='full':
        family_select = [CO_family, CO_13C, CO_18O, C_ion, H2O, HCN, Special, Other]
    else:
        family_select = []
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

                






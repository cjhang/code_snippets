# a collection of plots utils to inspect the data during calibratiion

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
#   2019.03.21, release
#   2020.04.11, update for less input parameters, drop dependence of analysisUtils


import os
import random
import numpy as np
from casa import tbtool, plotms, plotants


def group_antenna(vis, antenna_list=[], refant='', subgroup_member=6):
    """group a large number of antenna into small subgroup

    It can be help to plot several baseline and antenna into one figure.

    Parameters
    ----------
    antenna_list : list
        the antennas to be included
    refant : str
        the reference antenna, if it is set, the returned subgroups are baseline groups
        if the refant is None, the returned subgroups are antenna groups
    subgroup_member : int
        the member number if the subgroup
    
    Returns
    -------
    A list of subgroups
    """
    # Extract the antenna list from ms
    tb = tbtool()
    tb.open(vis+'/ANTENNA', nomodify=True)
    ants = tb.getcol('NAME')
    tb.close()
 
    subgroups = []
    # generate the baseline list
    if refant != '':
        # generates the baseline groups based on the refant
        refant_idx = np.where(ants == refant)
        # remove the refant from antenna_list
        ants_new = np.delete(ants, refant_idx)
        for i in range(0, len(ants_new), subgroup_member):
            antbaseline = ''
            for j in range(0, subgroup_member):
                try:
                    antbaseline += '{}&{};'.format(refant, ants_new[i+j])
                except:
                    pass
            subgroups.append(antbaseline[:-1]) #remove the last
    elif refant == '' or refant is None:
        # generates the antenna groups
        for i in range(0, len(ants), subgroup_member):
            ants_group = ''
            for j in range(0, subgroup_member):
                try:
                    ants_group += '{},'.format(ants[i+j])
                except:
                    pass
            subgroups.append(ants_group[:-1])
 
    return subgroups

def check_info(vis=None, showgui=False, plotdir='./plots', spw='',
               show_ants=True, show_mosaic=False, show_uvcoverage=True,
               show_elevation=True, show_allobs=False, refant='1', overwrite=True,
               bcal_field=None, gcal_field=None, target_field=None, 
               ):
    """ plot the basic information of the observation.

    Plots include: antenna positions, UV coverage, mosaic coverage(if possible)

    Examples
    --------
    To give a quick overview

        check_info(vis=msfile, spw='')

    if you also want to give a glance for the calibrators and science target:

        check_info(vis=msfile, spw='', bcal_field='bcal', gcal_field='gcal', target_field='fcal', refant='1')

    Parameters
    ----------
    vis : str
        measurement file
    showgui : bool
        show the plot window, the plotmosaic does not support yet, 
        default: False
    plotdir : str
        the base directory for all the info plots. 
        default: './plots'
    spw : str
        the spectral windows, if unset or empty string all the spws will plot at the same time;
        specify the spws in stings can plot them one by one through the loop in spws.split(',')
        default: ''
    show_ants : bool
        plot the positions of the antennas
        default: True
    show_mosaic : bool
        deprecated, need analysisUtils. Plot the relative positions of the mosaic
        default: False
    show_uvcoverage : bool
        plot the uv coverage of different field. Also depends the field number
        default: True
    show_elevation : bool
        plot the elevation with time for all the fields
        default: True
    show_allobs : bool
        plot amplitude vs time for all the fields together
        default: False
    refant : str
        the reference antenna, set to '' to plot all the antennas
        default: '1'
    overwrite : bool
        set to True to overwrite all the existing plots
        default: True
    bcal_field : str
        the bandpass calibrator fields
        default: None
    gcal_field : str
        the gain calibrator field
        default: None
    target_field : str
        the target field
        default: None
    """
    os.system('mkdir -p {}/info'.format(plotdir))
    plotdir = plotdir + '/info'

    if show_ants and not os.path.exists('{}/antpos.png'.format(plotdir)):
        # Checking antenna position, choosing the reference antenna
        print('Plotting antenna positions...')
        plotants(vis=vis, figfile='{}/antpos.png'.format(plotdir))

    if show_mosaic:
        # TODO, adapt the function from analysisUtils
        print("Warning, mosaic plot deprecated!")
        pass

    # if show_uvcoverage and not os.path.exists('{}/uvcoverage.png'.format(plotdir)):
        # print("Plotting u-v coverage...")
        # plotms(vis=vis, xaxis='U', yaxis='V', coloraxis='field', 
               # spw=spw, showgui=showgui, avgchannel='1e6', 
               # plotfile='{}/all_uvcoverage.png'.format(plotdir),
               # overwrite=overwrite)
    if show_elevation and not os.path.exists('{}/elevation.png'.format(plotdir)):
        print("Plotting elevation with time...")
        # Checking the evevation, determine the need of elivation calibration
        plotms(vis=vis, xaxis='time', yaxis='elevation', spw=spw,
               avgchannel='1e6', coloraxis='field', 
               plotfile='{}/elevation.png'.format(plotdir), 
               showgui=showgui, overwrite=overwrite)

    if show_allobs and not os.path.exists('{}/all_observations.png'.format(plotdir)):
        print("Plotting amplitude with time for all obs...")
        plotms(vis=vis, xaxis='time', yaxis='amp', spw=spw, 
               avgchannel='1e8', avgtime='60', coloraxis='field',
               plotfile='{}/all_observations.png'.format(plotdir),
               showgui=showgui, overwrite=overwrite)
    
    if bcal_field:
        print('Plotting bandpass calibrator...')
        plotms(vis=vis, xaxis='U', yaxis='V', field=bcal_field, 
               spw=spw, showgui=showgui,avgchannel='1e6', 
               plotfile='{}/bcal_uvcoverage.png'.format(plotdir),
               overwrite=overwrite)

        # phase change with time
        plotms(vis=vis, field=bcal_field, xaxis='time', yaxis='phase', antenna=refant,
               avgchannel='1e8', spw=spw, coloraxis='corr', ydatacolumn='data',
               plotfile='{}/bcal_{}_time.png'.format(plotdir, 'phase'),
               showgui=False, overwrite=overwrite)
        # phase change with freq
        for spw_single in spw.split(','):
            for yaxis in ['amp','phase']:
                plotms(vis=vis, field=bcal_field, xaxis='freq', yaxis=yaxis, antenna=refant,
                       avgtime='1e8',spw=spw_single, coloraxis='corr', ydatacolumn='data',
                       plotfile='{}/bcal_spw{}_{}_freq.png'.format(plotdir, spw_single, yaxis),
                       showgui=False, overwrite=overwrite)
    if gcal_field:
        print('Plotting gain calibrator...')
        plotms(vis=vis, xaxis='U', yaxis='V', field=gcal_field, 
               spw=spw, showgui=showgui,avgchannel='1e6', 
               plotfile='{}/gcal_uvcoverage.png'.format(plotdir),
               overwrite=overwrite)
        for yaxis in ['amp','phase']:
            plotms(vis=vis, field=gcal_field, spw=spw, ydatacolumn='data',
                   xaxis='time', yaxis=yaxis, avgchannel='1e8', 
                   coloraxis='corr', showgui=showgui, antenna=refant,
                   plotfile='{}/gcal_{}_time.png'.format(plotdir, yaxis),
                   overwrite=overwrite)
    if target_field:
        print('Plotting science target...')
        plotms(vis=vis, xaxis='U', yaxis='V', field=target_field, 
               spw=spw, showgui=showgui,avgchannel='1e6', 
               plotfile='{}/target_uvcoverage.png'.format(plotdir),
               overwrite=overwrite)
        plotms(vis=vis, xaxis='time', yaxis='amplitude', avgchannel='1e8', 
               spw=spw, field=target_field, coloraxis='corr', 
               antenna=refant, ydatacolumn='data',
               plotfile='{}/target_amp_time.png'.format(plotdir),
               showgui=showgui, overwrite=overwrite)

def check_tsys(vis=None, tdmspws=None, ants_subgroups=None, gridcols=2, 
               gridrows=3, plotdir='./plots', showgui=False):
    """the stand alone plot function for tsys
    
    
    Parameters
    ----------
    vis : str
        visibility of the measurement file
    tdmspws : str
        time domain mode spws
    ants_subgroups : list
        the list contains the subgroups to be plot into one figure
    tdmspws : str
        time domain mode spws 
        for example: '2,3,4' or '2~3'
    gridrows : int
        the rows for subplots
    gridcols : int
        the columns for subplots
    plotdir : str
        the directory where to generate the plots
    showgui : bool
        set to "True" to open the gui window
    """
    if ants_subgroups is None:
        # Extract the antenna list from ms
        tb = tbtool()
        tb.open(vis+'/ANTENNA', nomodify=True)
        ants = tb.getcol('NAME')
        tb.close()
        
        ants_subgroups = group_antenna(ants, subgroup_member=gridrows*gridcols)

    os.system('mkdir -p {}/tsys/'.format(plotdir))

    # plot tsys vs time, to pick out bad antenna
    for page,antenna in enumerate(ants_subgroups):
        plotms(vis=vis+'.tsys', xaxis='time', yaxis='Tsys', 
               coloraxis='spw', antenna=antenna,
               gridcols=2, gridrows=gridrows, iteraxis='antenna',
               showgui=showgui,
               plotfile='{}/tsys/Tsys_vs_time.page{}.png'.format(plotdir, page))

        # plot tsys vs frequency
        if tdmspws is None:
            raise ValueError("No tdmspws founded!")
        for spw in spw_expand(tdmspws):
            plotms(vis=vis+'.tsys', xaxis='freq', yaxis='Tsys', spw=spw,
                   gridcols=2, gridrows=gridrows, iteraxis='antenna',
                   coloraxis='corr', antenna=antenna, showgui=showgui,
                   plotfile='{}/tsys/spw{}_tsys_vs_freq.page{}.png'.format(plotdir, spw, page))

def check_cal(vis='', spw='', refant='', ydatacolumn='corrected',
              cal_fields='', target_field='',
              plot_freq=True, plot_time=True, plot_uvdist=True,
              overwrite=True, showgui=False, 
              gridrows=2, gridcols=3, dpi=600, plotdir='./plots'):
    """
    Check the calibrated data after `applycal`
    
    The wrapped tools based on `plotms` for manual calibration.

    Examples
    --------
    check the amplitude and phase of calibrator change with time and frequency:
    (set refant to plot with baseline, which is much faster)
        # to iterate the baseline of refant
        check_cal(vis=msfile, spw='', cal_fields='0,1', refant='ea01')
        # to include all the antenna
        check_cal(vis=msfile, spw='', cal_fields='0,1', refant='ea01')

    for the science target:
    
        check_cal(vis=msfile, spw='', target_field='2')

    Parameters
    ----------
    vis : str
        measurements file
    spw : str
        same as casa
        default: ''
    refant : str
        the reference antenna, like 'CM03', 'DV48'
        if the refant is specified, the iteration axis will change to baseline, otherwise it will iterate through antenna
        default: ''
    ydatacolumn : str
        the ydatacolumn of `plotms`
        default: 'corrected'
    cal_fileds : str
        a string contains all the fields of calibrators
        example: '0,2,3' or 'J1427-4206,Mars'
        default: ''
    science_field : str
        the fields of the science target
        example: 'Cen*' or '4~22'
    plot_freq : bool
        Plot the amplitude-vs-frequency, phase-vs-frequency for both data.
        default: True
    plot_time : bool
        plot the amplitude-vs-time, phase-vs-time for both data column
        default: True
    plot_uvdist : bool
        plot amplitude vs uvdist 
        default: True
    overwrite : bool
        default True
    showgui : bool
        default: True
    gridrows, gridcols : int
        the number of row and columns
        default: gridrows=2, gridcols=3
    dpi : int
        the resolution of the saved png files
        default: 600
    plotdir : str
        the root directory to put all the generated plots
    """
    # checking the validation of parameters
    if vis == '':
        raise ValueError("You must give the measurements file name `vis`!")

    # plot the frequency related scatter
    if plot_freq and cal_fields != '':
        print("Plot frequency related calibration for fields: {} ...".format(cal_fields))
        os.system('mkdir -p {}/freq_{}/'.format(plotdir, ydatacolumn))
        for field in cal_fields.split(','):
            print(">> field: {}".format(field))
            for yaxis in ['amplitude', 'phase']:
                if refant != '':
                    iteraxis = 'baseline' 
                else:
                    iteraxis = 'antenna'
                    # generate all the antenna firstly
                    for spw_single in spw.split(','):
                        plotms(vis=vis, field=field, xaxis='frequency', yaxis=yaxis,
                               spw=spw_single, avgtime='1e8', avgscan=True, coloraxis='corr',
                               ydatacolumn=ydatacolumn, showgui=showgui,
                               dpi = dpi, overwrite=overwrite, verbose=False,
                               plotfile='{}/freq_{}/field{}-spw{}-{}_vs_freq.all.png'.format(
                                        plotdir, ydatacolumn, field, spw_single, yaxis))
                subgroups = group_antenna(vis, refant=refant, subgroup_member=gridrows*gridcols)
                # print(subgroups)
                for page, subgroup in enumerate(subgroups):
                    for spw_single in spw.split(','):
                        plotms(vis=vis, field=field, xaxis='frequency', yaxis=yaxis,
                               spw=spw_single, avgtime='1e8', avgscan=True, coloraxis='corr',
                               antenna=subgroup, iteraxis=iteraxis, ydatacolumn=ydatacolumn,
                               showgui=showgui, gridrows=gridrows, gridcols=gridcols,
                               dpi = dpi, overwrite=overwrite, verbose=False,
                               plotfile='{}/freq_{}/field{}-spw{}-{}_vs_freq.page{}.png'.format(
                                         plotdir, ydatacolumn, field, spw_single, yaxis, page))

    # the phase should be significantly improved after bandpass calibration
    # especially for the phase calibrator
    if plot_time and cal_fields != '':
        print("Plot time related calibration for fields: {} ...".format(cal_fields))
        os.system('mkdir -p {}/time_{}/'.format(plotdir, ydatacolumn))
        for yaxis in ['amplitude', 'phase']:
            # plot the general consistency of each field
            for field in cal_fields.split(','):
                print(">> field: {}".format(field))
                if refant != '':
                    iteraxis = 'baseline' 
                else:
                    iteraxis = 'antenna'
                    for spw_single in spw.split(','):
                        plotms(vis=vis, field=field, xaxis='time', yaxis=yaxis, spw=spw_single, 
                               avgchannel='1e8', coloraxis='corr', ydatacolumn=ydatacolumn,
                               plotfile='{}/time_{}/field{}_{}_vs_time.png'.format(plotdir, ydatacolumn, field, yaxis),
                               showgui=showgui, dpi = dpi, overwrite=overwrite)
                subgroups = group_antenna(vis, refant=refant, subgroup_member=gridrows*gridcols)
                
                for page, subgroup in enumerate(subgroups):
                    for spw_single in spw.split(','):
                        plotms(vis=vis, field=field, xaxis='time', yaxis=yaxis,
                               spw=spw_single, avgchannel='1e8', coloraxis='corr',
                               antenna=subgroup, iteraxis=iteraxis, ydatacolumn=ydatacolumn,
                               showgui=showgui, gridrows=gridrows, gridcols=gridcols,
                               plotfile='{}/time_{}/field{}_spw{}_{}_vs_time.page{}.png'.format(
                                         plotdir, ydatacolumn, field, spw_single, yaxis, page),
                               dpi = dpi, overwrite=overwrite)

    if plot_uvdist and cal_fields != '':
        # well behaved point source should show flat amplitude with uvdist
        print("Plot uvdist related calibration for {} ...".format(cal_fields))
        os.system('mkdir -p {}/uvdist/'.format(plotdir))
        for field in cal_fields.split(','):
            print('>> field: {}'.format(field))
            plotms(vis=vis, field=field, xaxis='uvdist', yaxis='amp',
                   avgchannel='1e8', coloraxis='corr', ydatacolumn=ydatacolumn,
                   plotfile='{}/uvdist/field{}_amp_vs_uvdist.png'.format(plotdir, field),
                   dpi=dpi, overwrite=overwrite, showgui=showgui)
       
    if target_field != '':
        print("Giving the science target a glance ...")
        os.system('mkdir -p {}/target/'.format(plotdir))
        for field in target_field.split(','):
            if plot_uvdist:
                print(">> Plotting amplitude vs uvdist for science target...")
                plotms(vis=vis, xaxis='uvdist', yaxis='amp',
                       ydatacolumn=ydatacolumn, field=field,
                       avgchannel='1e8', coloraxis='corr',
                       plotfile = '{}/target/target_amp_vs_uvdist.png'.format(plotdir),
                       showgui=showgui, dpi=dpi, overwrite=overwrite)
            for spw in spw.split(','):
                if plot_freq:
                    print(">> Plotting amplitude vs frequency for science target...")
                    plotms(vis=vis, xaxis='freq', yaxis='amp', spw=spw,
                           ydatacolumn=ydatacolumn, field=field,
                           avgtime='1e8', avgscan=True, coloraxis='corr',
                           plotfile='{}/target/target_amp_vs_{}.png'.format(plotdir, 'freq'),
                           showgui=showgui, dpi=dpi, overwrite=overwrite)
                if plot_time:
                    print(">> Plotting amplitude vs time for science target...")
                    plotms(vis=vis, xaxis='time', yaxis='amp', spw=spw,
                           ydatacolumn=ydatacolumn, field=field,
                           avgchannel='1e8', avgscan=True, coloraxis='corr',
                           plotfile='{}/target/target_amp_vs_{}.png'.format(plotdir, 'time'),
                           showgui=showgui, dpi=dpi, overwrite=overwrite)

def check_bandpass(fgcal='bpphase.gcal', fbcal='bandpass.bcal', 
                   dpi=600, gridrows=2, gridcols=2, plotdir='./plots'):
    """check the data quality of bandpass calibrator and the calibration
    """
    os.system('mkdir -p {}/bandpass'.format(plotdir))
    if os.path.exists(fgcal):
        print('Plotting {}'.format(fgcal))
        subgroups = group_antenna(fgcal, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            # after casa 6, consider using pathlib
            plotms(vis=fgcal, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/bandpass/amp_page{}.png'.format(plotdir, page),
                   plotrange=[0,0,-180,180], showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct bandpass gain table! Set the fgcal parameter")

    if os.path.exists(fbcal):
        print('Plotting {}'.format(fbcal))
        subgroups = group_antenna(fbcal, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            for yaxis in ['amp', 'phase']:
                plotms(vis=fbcal, gridrows=gridrows, gridcols=gridcols, xaxis='freq',
                       yaxis=yaxis, antenna=antenna, iteraxis='antenna', coloraxis='corr',
                       plotfile='{}/bandpass/{}_page{}.png'.format(plotdir, yaxis, page),
                       showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct bandpass calibration table! Set the fbcal parameter")

def check_gain(phase_int='phase_int.gcal', phase_scan='phase_scan.gcal', amp_scan='amp_scan.gcal', 
               dpi=600, gridrows=2, gridcols=2, plotdir='./plots'):
    """check the gain table
    In order to use this function, one should follow the same naming 
    """
    os.system('mkdir -p {}/gain_solution/'.format(plotdir))
    if os.path.exists(phase_int):
        print('Plotting {}'.format(phase_int))
        subgroups = group_antenna(phase_int, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            plotms(vis=phase_int, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/gain_solution/phase_int_page{}.png'.format(plotdir, page),
                   plotrange=[0,0,-180,180], showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct integrated phase table! Set the phase_int parameter")

    if os.path.exists(phase_scan):
        print('Plotting {}'.format(phase_scan))
        subgroups = group_antenna(phase_scan, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            plotms(vis=phase_scan, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/gain_solution/phase_scan_page{}.png'.format(plotdir, page),
                   plotrange=[0,0,-180,180], showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct scan averaged phase table! Set the phase_scan parameter")

    if os.path.exists(amp_scan):
        print('Plotting {}'.format(amp_scan))
        subgroups = group_antenna(amp_scan, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            for yaxis in ['amp', 'phase']:
                plotms(vis=amp_scan, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                       yaxis=yaxis, antenna=antenna, iteraxis='antenna', coloraxis='corr',
                       plotfile='{}/gain_solution/{}_page{}.png'.format(plotdir, yaxis, page),
                       showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct amp_scan calibration table! Set the amp_scan parameter")

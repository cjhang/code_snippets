# a collection of plots utils to inspect the data during calibration

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
#   2019.03.21, release
#   2020.04.11, update for less input parameters, drop dependence of analysisUtils


import os
import random
import numpy as np

try:
    from casa import tbtool, plotms, plotants
except:
    from casatools import table as tbtool
    from casaplotms import plotms
    from casatasks import plotants

def spw_expand(spws):
    """expand casa spws string format into single spw"""
    spw_list = []
    for spw in spws.split(','):
        spw_list.append(spw.strip()) 
    return spw_list

def list2str(ll):
    """convert list into string with seperator ','
    Parameters
    ----------
    ll : list
        the input list or numpy ndarray
    """
    string = ''
    for i in ll:
        string += str(i)+','
    return string[:-1]

def group_antenna(vis=None, ants=None, refant='', subgroup_member=6):
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
    if ants is None:
        # Extract the antenna list from ms
        tb = tbtool()
        tb.open(vis+'/ANTENNA', nomodify=True)
        ants = tb.getcol('NAME')
        tb.close()
 
    subgroups = []
    # generate the baseline list
    if refant != '':
        # generates the baseline groups based on the refant
        if refant not in ants:
            raise ValueError("Refant {} is not a valid antenna!".format(refant))
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

def check_info(vis=None, showgui=False, plotdir='./plots', basename=None,
               spw='', show_ants=True, show_mosaic=False, show_uvcoverage=True,
               show_elevation=True, show_allobs=True, show_uv=True,
               refant='1', overwrite=True, avgtime='1e8', avgchannel='1e8',
               intents=['*BANDPASS*','*FLUX*','*PHASE*','*TARGET*'],
               **kwargs):
    """ plot the basic information of the observation.

    Plots include: antenna positions, UV coverage, mosaic coverage(if possible)

    Examples
    --------
    To give a quick overview

        check_info(vis=msfile, spw='')

    if you also want to give a glance for the calibrators and science target:

        plot_utils.check_info(obs, refant=refant, correlation='XX,YY', avgchannel='64', datacolumn='data')

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
    """
    if basename is None:
        basename = os.path.basename(vis)
    outdir = os.path.join(plotdir, "info-"+basename)
    os.system('mkdir -p {}'.format(outdir))

    if show_ants and not os.path.exists('{}/antpos.png'.format(outdir)):
        # Checking antenna position, choosing the reference antenna
        print('Plotting antenna positions...')
        plotants(vis=vis, figfile='{}/antpos.png'.format(outdir))

    if show_mosaic:
        # TODO, adapt the function from analysisUtils
        print("Warning, mosaic plot deprecated!")
        pass

    # if show_uvcoverage and not os.path.exists('{}/uvcoverage.png'.format(outdir)):
        # print("Plotting u-v coverage...")
        # plotms(vis=vis, xaxis='U', yaxis='V', coloraxis='field', 
               # spw=spw, showgui=showgui, avgchannel=avgchannel, 
               # plotfile='{}/all_uvcoverage.png'.format(outdir),
               # overwrite=overwrite)
    if show_elevation and not os.path.exists('{}/elevation.png'.format(outdir)):
        print("Plotting elevation with time...")
        # Checking the evevation, determine the need of elivation calibration
        plotms(vis=vis, xaxis='time', yaxis='elevation', spw=spw,
               avgchannel=avgchannel, coloraxis='field', highres=True,
               plotfile='{}/elevation.png'.format(outdir), 
               showgui=showgui, overwrite=overwrite)

    if show_allobs and not os.path.exists('{}/all_observations.png'.format(outdir)):
        print("Plotting amplitude with time for all obs...")
        plotms(vis=vis, xaxis='time', yaxis='amp', spw=spw,  highres=True,
               avgchannel=avgchannel, avgtime='60', coloraxis='field',
               plotfile='{}/all_observations.png'.format(outdir),
               showgui=showgui, overwrite=overwrite, **kwargs)
    if show_uv:
        plotms(vis=vis, xaxis='U', yaxis='V', coloraxis='field', highres=True, 
               spw=spw, showgui=showgui, avgchannel=avgchannel, 
               plotfile='{}/uvcoverage.png'.format(outdir),
               overwrite=overwrite)
    for myintent in intents:
        myintent_name = myintent[1:-1]
        # amp change with time
        plotms(vis=vis, intent=myintent, xaxis='time', yaxis='amp', antenna=refant,
               avgchannel=avgchannel, spw=spw, coloraxis='spw', ydatacolumn='data',
               plotfile='{}/{}_amp_time.png'.format(outdir, myintent_name), 
               highres=True, showgui=showgui, overwrite=overwrite, **kwargs)
        # amplitude change with freq
        plotms(vis=vis, intent=myintent, xaxis='freq', yaxis='amp', antenna=refant,
               avgtime=avgtime,spw=spw, coloraxis='scan', ydatacolumn='data',
               plotfile='{}/{}_amp__freq.png'.format(outdir, myintent_name), 
               highres=True, showgui=showgui, overwrite=overwrite, **kwargs)
        # amp change with uvdist
        plotms(vis=vis, intent=myintent, xaxis='uvdist', yaxis='amp', antenna=refant,
               avgchannel=avgchannel, spw=spw, coloraxis='spw', ydatacolumn='data',
               plotfile='{}/{}_amp_uvdist.png'.format(outdir, myintent_name), 
               highres=True, showgui=showgui, overwrite=overwrite, **kwargs)

def check_tsys(tsystable=None, spws=None, ants_subgroups=None, gridcols=2, gridrows=3,
               basename=None, basedir='./', plotdir='./plots', showgui=False,
               dpi=400):
    """the stand alone plot function for tsys
    
    Example
    -------

        check_tsys(tsystable='msfile1.tsys')
        check_tsys(tsystable='msfile1.tsys', spws='13,15,17,19')
    
    Parameters
    ----------
    vis : str 
        visibility of the measurement file
    tdmspws : str
        time domain mode spws, seperated with comma
    ants_subgroups (list): the list contains the subgroups to be plot into one figure
    spws : str
        time domain mode spws, for example: '2,3,4' or '2~3'
    gridrows : int
        the rows for subplots
    gridcols : int
        the columns for subplots
    plotdir : str
        the directory where to generate the plots
    showgui : bool
        set to "True" to open the gui window

       
    """
    if basename is None:
        basename = os.path.basename(tsystable)
    if ants_subgroups is None:
        ants_subgroups = group_antenna(vis=tsystable, subgroup_member=gridrows*gridcols)

    outdir =  '{}/tsys-{}'.format(plotdir, basename)
    os.system('mkdir -p {}'.format(outdir))

    # plot tsys vs time, to pick out bad antenna
    for page,antenna in enumerate(ants_subgroups):
        plotms(vis=tsystable, xaxis='time', yaxis='Tsys', 
               coloraxis='spw', antenna=antenna,
               gridcols=2, gridrows=gridrows, iteraxis='antenna',
               showgui=showgui, dpi=dpi, highres=True,
               plotfile='{}/Tsys_vs_time.page{}.png'.format(outdir, page))

        # plot tsys vs frequency
        if spws is not None:
            for spw in spw_expand(spws):
                plotms(vis=tsystable, xaxis='channel', yaxis='Tsys', spw=spw,
                       gridcols=2, gridrows=gridrows, iteraxis='antenna',
                       coloraxis='corr', antenna=antenna, showgui=showgui,
                       plotfile='{}/spw{}_tsys_vs_freq.page{}.png'.format(outdir, spw, page),
                       dpi=dpi, highres=True)
        else:
            plotms(vis=tsystable, xaxis='freq', yaxis='Tsys', spw='',
                   gridcols=2, gridrows=gridrows, iteraxis='antenna',
                   coloraxis='corr', antenna=antenna, showgui=showgui,
                   plotfile='{}/tsys_vs_freq.page{}.png'.format(outdir, page),
                   dpi=dpi, highres=True)

def check_cal(vis='', spw='', refant='', ydatacolumn='corrected', basename=None,
              field='', yaxis=['amplitude', 'phase'],
              plot_freq=True, plot_time=True, plot_uvdist=True,
              overwrite=True, showgui=False, 
              avgtime='1e8', avgchannel='1e8',
              gridrows=2, gridcols=3, dpi=600, plotdir='./plots',
              **kwargs):
    """
    Check the calibrated data after `applycal`
    
    The wrapped tools based on `plotms` for manual calibration.

    Examples
    --------
    check the amplitude and phase of calibrator change with time and frequency:
    (set refant to plot with baseline, which is much faster)
        # to iterate the baseline of refant
        check_cal(vis=msfile, spw='', field='bcal,gcal', refant='ea01')
        # to include all the antenna
        check_cal(vis=msfile, spw='', field='bcal,gcal')

    for the science target:
    
        check_cal(vis=msfile, spw='', field='science_field')

    Parameters
    ----------
    vis : str
        measurements file
    spw : str
        same as casa
        default: ''
    refant : str
        the reference antenna, like 'CM03', 'DV48', 'ea01'
        if the refant is specified, the iteration axis will change to baseline, otherwise it will iterate through antennas
        default: ''
    ydatacolumn : str
        the ydatacolumn of `plotms`
        default: 'corrected'
    field : str
        the field or fileds to be plotted
        example: '0,2,3' or 'J1427-4260,Mars', not support '~'
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
        default: True
    showgui : bool
        default: True
    gridrows : int 
        the number of rows
        default: gridrows=2 
    gridcols : int
        the number of columns
        default: gridcols=3
    dpi : int
        the resolution of the saved png files
        default: 600
    plotdir : str
        the root directory to put all the generated plots
    """
    # checking the validation of parameters
    if vis == '':
        raise ValueError("You must give the measurements file name `vis`!")
    if basename is None:
        basename = os.path.basename(vis)
    outdir = os.path.join(plotdir, "checkcal-"+basename)
    os.system('mkdir -p {}'.format(outdir))

    tb = tbtool()
    if field == '':
        # read all the fields
        tb.open(vis)
        field_list = np.unique(tb.getcol('FIELD_ID'))
        field = list2str(field_list)
        tb.close()
    # plot the frequency related scatter
    if plot_freq:
        print("Plot frequency related calibration for field: {} ...".format(field))
        os.system('mkdir -p {}/freq_{}/'.format(outdir, ydatacolumn))
        for field_single in field.split(','):
            print(">> field: {}".format(field_single))
            for yx in yaxis:
                if refant == 'all':
                    pass
                    # for spw_single in spw.split(','):
                        # plotms(vis=vis, field=field_single, xaxis='frequency', yaxis=yx,
                               # spw=spw_single, avgtime=avgtime, avgscan=False, coloraxis='corr',
                               # ydatacolumn=ydatacolumn, showgui=showgui,
                               # dpi = dpi, overwrite=overwrite, verbose=False,
                               # plotfile='{}/freq_{}/field{}-spw{}-{}_vs_freq.all.png'.format(
                                        # outdir, ydatacolumn, field_single, spw_single, yx))
                else: 
                    if refant == '':
                        iteraxis = 'antenna'
                    else:
                        iteraxis = 'baseline' 
                    # generate all the antenna firstly
                    subgroups = group_antenna(vis, refant=refant, subgroup_member=gridrows*gridcols)
                    # print(subgroups)
                    for page, subgroup in enumerate(subgroups):
                        for spw_single in spw.split(','):
                            if spw == '':
                                mycoloraxis = 'spw'
                            else:
                                mycoloraxis = 'corr'
                            plotms(vis=vis, field=field_single, xaxis='frequency', yaxis=yx,
                                   spw=spw_single, avgtime=avgtime, avgscan=False, 
                                   coloraxis=mycoloraxis,
                                   antenna=subgroup, iteraxis=iteraxis, ydatacolumn=ydatacolumn,
                                   showgui=showgui, gridrows=gridrows, gridcols=gridcols,
                                   dpi = dpi, overwrite=overwrite, verbose=False, highres=True,
                                   plotfile='{}/freq_{}/field{}-spw{}-{}_vs_freq.page{}.png'.format(
                                             outdir, ydatacolumn, field_single, spw_single, yx, page),
                                   **kwargs)

    # the phase should be significantly improved after bandpass calibration
    # especially for the phase calibrator
    if plot_time:
        print("Plot time related calibration for field: {} ...".format(field))
        os.system('mkdir -p {}/time_{}/'.format(outdir, ydatacolumn))
        for field_single in field.split(','):
            print(">> field: {}".format(field_single))
            for yx in yaxis:
                # plot the general consistency of each field
                if refant == 'all':
                    pass
                    # for spw_single in spw.split(','):
                        # plotms(vis=vis, field=field_single, xaxis='time', yaxis=yx, spw=spw_single, 
                               # avgchannel=avgchannel, coloraxis='corr', ydatacolumn=ydatacolumn,
                               # plotfile='{}/time_{}/field{}_{}_vs_time.png'.format(outdir, ydatacolumn, field_single, yx),
                               # showgui=showgui, dpi = dpi, overwrite=overwrite)
                else:
                    if refant == '':
                        iteraxis = 'antenna'
                    else:
                        iteraxis = 'baseline' 
                    subgroups = group_antenna(vis, refant=refant, subgroup_member=gridrows*gridcols)
                    for page, subgroup in enumerate(subgroups):
                        for spw_single in spw.split(','):
                            if spw == '':
                                mycoloraxis = 'spw'
                            else:
                                mycoloraxis = 'corr'
                            plotms(vis=vis, field=field_single, xaxis='time', yaxis=yx,
                                   spw=spw_single, avgchannel=avgchannel, coloraxis=mycoloraxis,
                                   antenna=subgroup, iteraxis=iteraxis, ydatacolumn=ydatacolumn,
                                   showgui=showgui, gridrows=gridrows, gridcols=gridcols,
                                   plotfile='{}/time_{}/field{}_spw{}_{}_vs_time.page{}.png'.format(
                                             outdir, ydatacolumn, field_single, spw_single, yx, page),
                                   dpi = dpi, overwrite=overwrite, highres=True, **kwargs)

    if plot_uvdist:
        # well behaved point source should show flat amplitude with uvdist
        print("Plot uvdist related calibration for {} ...".format(field))
        os.system('mkdir -p {}/uvdist/'.format(outdir))
        for field_single in field.split(','):
            print('>> field: {}'.format(field_single))
            for spw_single in spw.split(','):
                if spw == '':
                    mycoloraxis = 'spw'
                else:
                    mycoloraxis = 'corr'
                plotms(vis=vis, field=field_single, xaxis='uvdist', yaxis='amp', spw=spw_single,
                       avgchannel=avgchannel, coloraxis='corr', ydatacolumn=ydatacolumn,
                       plotfile='{}/uvdist/field{}_spw{}_amp_vs_uvdist.png'.format(outdir, field_single, spw_single),
                       dpi=dpi, overwrite=overwrite, showgui=showgui, highres=True, **kwargs)
                plotms(vis=vis, field=field_single, xaxis='U', yaxis='V', spw=spw_single,
                       avgchannel=avgchannel, coloraxis=mycoloraxis, ydatacolumn=ydatacolumn,
                       plotfile='{}/uvdist/field{}_spw{}_uvcoverage.png'.format(outdir, field_single, spw_single),
                       dpi=dpi, overwrite=overwrite, showgui=showgui, highres=True, **kwargs)

def check_bandpass(phaseint_cal='bpphase.gcal', bandpass_cal='bandpass.bcal', basename=None,
                   dpi=600, gridrows=2, gridcols=2, plotdir='./plots/bandpass'):
    """check the data quality of bandpass calibrator and the calibration

    it can plot two solution: the short time gain (fgcal) and the bandpass solution (fbcal)

    Parameters
    ----------
    phasecal_int : str
        the file of gain solution for bandpass calibrator
    bpcal : str
        the file of the bandpass solution
    gridrows : int 
        the number of rows
        default: 2 
    gridcols : int
        the number of columns
        default: 2
    dpi : int
        the resolution of the saved png files
        default: 600
    plotdir : str
        the root directory to put all the generated plots
    """
   
    if os.path.exists(phaseint_cal):
        if basename is None:
            outname = os.path.basename(phaseint_cal)
        else:
            outname = basename+'phaseint'
        outdir = os.path.join(plotdir, outname)
        os.system('mkdir -p {}'.format(outdir))
 
        print('Plotting {}'.format(phaseint_cal))
        subgroups = group_antenna(phaseint_cal, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            # after casa 6, consider using pathlib
            plotms(vis=phaseint_cal, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/phase_page{}.png'.format(outdir, page),
                   plotrange=[0,0,-180,180], 
                   showgui=False, dpi=dpi, overwrite=True, highres=True)
    else:
        print("Warning: you should give the correct bandpass gain table! Set the fgcal parameter")

    if os.path.exists(bandpass_cal):
        if basename is None:
            outname = os.path.basename(bandpass_cal)
        else:
            outname = basename+'bandpass'
        outdir = os.path.join(plotdir, outname)
        os.system('mkdir -p {}'.format(outdir))
        print('Plotting {}'.format(bandpass_cal))
        subgroups = group_antenna(bandpass_cal, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            for yaxis in ['amp', 'phase']:
                plotms(vis=bandpass_cal, gridrows=gridrows, gridcols=gridcols, xaxis='freq',
                       yaxis=yaxis, antenna=antenna, iteraxis='antenna', coloraxis='corr',
                       plotfile='{}/{}_page{}.png'.format(outdir, yaxis, page),
                       showgui=False, dpi=dpi, overwrite=True, highres=True)
    else:
        print("Warning: you should give the correct bandpass calibration table! Set the fbcal parameter")

def check_gain(phasecal_int='phase_int.gcal', phasecal_scan='phase_scan.gcal', ampcal_scan='amp_scan.gcal', 
               dpi=600, gridrows=2, gridcols=2, plotdir='./plots'):
    """check the gain table
    In order to use this function, one should follow the same naming 
    It is designed to plot the phase and amplitude splution from the gain calibrator

    Parameters:
    phase_int : str
        the file of integrated time (or specified short time) solution for phase variation
    phase_scan : str
        the file of scan based solution for gain calibrator, used for science target
    ampcal_scan : str
        the file of amplitude solution based on scan integration
    gridrows : int 
        the number of rows
        default: 2 
    gridcols : int
        the number of columns
        default: 2
    dpi : int
        the resolution of the saved png files
        default: 600
    plotdir : str
        the root directory to put all the generated plots
    """
    os.system('mkdir -p {}/gain_solution/'.format(plotdir))
    if os.path.exists(phasecal_int):
        print('Plotting {}'.format(phasecal_int))
        subgroups = group_antenna(phasecal_int, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            plotms(vis=phasecal_int, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/gain_solution/phase_int_page{}.png'.format(plotdir, page),
                   plotrange=[0,0,-180,180], showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct integrated phase table! Set the phase_int parameter")

    if os.path.exists(phasecal_scan):
        print('Plotting {}'.format(phasecal_scan))
        subgroups = group_antenna(phasecal_scan, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            plotms(vis=phasecal_scan, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                   yaxis='phase', antenna=antenna, iteraxis='antenna', coloraxis='corr',
                   plotfile='{}/gain_solution/phase_scan_page{}.png'.format(plotdir, page),
                   plotrange=[0,0,-180,180], showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct scan averaged phase table! Set the phase_scan parameter")

    if os.path.exists(ampcal_scan):
        print('Plotting {}'.format(ampcal_scan))
        subgroups = group_antenna(ampcal_scan, subgroup_member=gridrows*gridcols)
        for page, antenna in enumerate(subgroups):
            for yaxis in ['amp', 'phase']:
                plotms(vis=ampcal_scan, gridrows=gridrows, gridcols=gridcols, xaxis='time',
                       yaxis=yaxis, antenna=antenna, iteraxis='antenna', coloraxis='corr',
                       plotfile='{}/gain_solution/{}_page{}.png'.format(plotdir, yaxis, page),
                       showgui=False, dpi=dpi, overwrite=True)
    else:
        print("Warning: you should give the correct amp_scan calibration table! Set the amp_scan parameter")

def check_Dterm(Dtermtable, spw='', showgui=False, plotdir='./plots', basename=None,
                overwrite=True, gridrows=2, gridcols=3, dpi=600, 
               **kwargs):
    if basename is None:
        basename = os.path.basename(Dtermtable)
    outdir = os.path.join(plotdir, "Dterm-"+basename)
    os.system('mkdir -p {}'.format(outdir))

    for yaxis in ['amp', 'real', 'imag']:
        subgroups = group_antenna(Dtermtable, subgroup_member=gridrows*gridcols)
        for page, subgroup in enumerate(subgroups):
            plotms(vis=Dtermtable, xaxis='frequency', yaxis=yaxis, antenna=subgroup,
                   spw=spw, iteraxis='antenna', coloraxis='corr', 
                   plotfile='{}/freq_{}.page{}.png'.format(outdir, yaxis, page), 
                   showgui=showgui, highres=True, **kwargs)

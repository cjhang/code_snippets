# ALMA Data Reduction Script
# $Id: scriptForPolCalibration_template.py,v 1.18 2022/10/20 15:20:20 dpetry Exp $

# POLARIZATION CALIBRATION - ORIGINAL VERSION DATE: 14-12-2015 - Rev. 5
# REVISED by gmoellen (2016Sep12) to calibrate the X/Y amp ratio
#  in the science target and phase calibrator
# REVISED by R. Paladino and D. Petry, Jan-Mar 2019, Oct 2019
# REVISED by gmoellen (2019Jul05, 2020Jan20) to use new task feature (less almapolhelpers)
# REVISED by D. Petry to be CASA 6 compatible, Mar 2020
# REVISED by D. Petry to accept also PL-calibrated MSs with CORRECTED_DATA column
# REVISED by R. Paladino, Nov 2020, D. Petry, Dec 2020
# REVISED by T. Carozzi, R. Paladino, D. Petry, Nov 2021

# REVISED by T. Carozzi, R. Paladino, D. Petry, Oct 2022 

# Calibration

thesteps = []
step_title = {0: 'Concatenation of the EBs per Session, separation into TDM and FDM if needed',
              1: 'Make a first gain table to check the data - TDM',
              2: 'Additional flagging  mostly based on Gpol1 from step 1 ',
              3: 'Polarization calibration ',
              4: 'Save flags before applycal ',
              5: 'Apply polarization calibration to the polarization calibrator',
              6: 'Image of the polarization calibrator - to check',
              7: 'Apply polarization calibration to all targets',
              8: 'Save flags after applycal',
              9: 'Split out corrected column',
              10: 'Make a first gain table to check the data - FDM',
              11: 'Additional flagging  mostly based on Gpol1 from step 10 ',
              12: 'Polarization calibration -FDM',
              13: 'Save flags before applycal -FDM ',
              14: 'Apply polarization calibration to the polarization calibrator -FDM',
              15: 'Image of the polarization calibrator - to check -FDM',
              16: 'Apply polarization calibration to all targets - FDM',
              17: 'Save flags after applycal - FDM',
              18: 'Split out corrected column - FDM'
              }             

try:
  print('List of steps to be executed ...'+str(mysteps))
  thesteps = mysteps
except:
  print('global variable mysteps not set.')

# The Python variable 'mysteps' will control which steps
# are executed when you start the script using
#   execfile('scriptForCalibration.py')
# e.g. setting
#   mysteps = [2,3,4]# before starting the script will make the script execute
# only steps 2, 3, and 4
# Setting mysteps = [] will make it execute all steps necessary for processing
# from scratch the given tdm and fdm spws.

import re
import pylab as pl
import numpy as np
import os
import glob
import math

# the msnames and EBs for each session
EBs = { 'concat_S1.ms': ['uid___A002_Xd4b9e4_Xa7e5.ms.split.cal','uid___A002_Xd4b9e4_Xab8d.ms.split.cal','uid___A002_Xd4b9e4_Xb11e.ms.split.cal'],
        'concat_S2.ms': ['uid___A002_Xd50463_X95f7.ms.split.cal','uid___A002_Xd50463_X981f.ms.split.cal']
        }

refants = {'S1': 'DV20',
           'S2': 'DV20'
           }

# the TDM and FDM SPWs in the concatenated MSs
# NOTE: if there are no SPWs of one kind, set the variable to '' (empty string)!
tdmspws = '0,1,2,3'
fdmspws = ''

targets = ['B335'] # enumerate all science targets here  

diagnostic_image_cellsize = '1.13arcsec'



####################################################

if (thesteps==[]):
  thesteps = [0]
  if len(tdmspws) > 0 :
    thesteps += range(1,10)
  else:
    print("There are no TDM SPWs in this dataset. Steps 1 to 9 of the script will not be executed.")
  if len(fdmspws) > 0 :
    thesteps += range(10,19)
  else:
    print("There are no FDM SPWs in this dataset. Steps 10 to 18 of the script will not be executed.")
  print('Executing steps: '+str(thesteps))


####################################################

# Determine calibrator fields if not yet done

if not ('polcalib' in locals() and 'phasecal' in locals() and 'bandpasscal' in locals()):
  phasecal = {}
  polcalib = {}
  bandpasscal = {}

  for mymsname in sorted(EBs.keys()):
    msmd.open(EBs[mymsname][0])
    lookup = msmd.fieldnames()
    phasecal[mymsname+'.TDM'] = lookup[msmd.fieldsforintent('*PHASE*')[0]]
    bandpasscal[mymsname+'.TDM'] = lookup[msmd.fieldsforintent('*BANDPASS*')[0]]
    polcalib[mymsname+'.TDM'] = lookup[msmd.fieldsforintent('*POLARIZATION*')[0]]
    phasecal[mymsname+'.FDM'] = phasecal[mymsname+'.TDM']
    bandpasscal[mymsname+'.FDM'] = bandpasscal[mymsname+'.TDM']
    polcalib[mymsname+'.FDM'] = polcalib[mymsname+'.TDM']
    msmd.close()

    print("Session "+str(mymsname))
    print("   Bandpass Calibrator "+str(bandpasscal[mymsname+'.TDM']))
    print("   Phase Calibrator "+str(phasecal[mymsname+'.TDM']))
    print("   Polarisation Calibrator "+str(polcalib[mymsname+'.TDM']))


####################################################
####################################################

def polcalcore(msname, refant, polcalname, avgbw=''):
    """
    polcalcore(msname, refant, avgbw, polcalname)

      msname = the concatenated MS containing only the science SPWs (reindexed)
      refant = name of the reference antenna as used in the calibration
      polcalname = field name of the polarisation calibrator
      avgbw = apply a spectral averaging over this bandwidth during the calibration, e.g. '5MHz'

    """


    avgstring=''
  
    if avgbw!='': # we need to do averaging
      print("Will increase the spectral solution interval for the pol solutions to "+avgbw)
      avgstring = ','+avgbw

    # Figure out typical scan duration of the polcalib scans:

    ms.open(msname+'_polcalib.ms')
    summary = ms.summary()
    ms.close()

    tb.open(msname+'_polcalib.ms')
    scnum = tb.getcol('SCAN_NUMBER')
    stid = tb.getcol('STATE_ID')
    tb.close()

    tb.open(msname+'_polcalib.ms'+'/STATE')
    obsm = tb.getcol('OBS_MODE')
    tb.close()

    polst = [i for i in range(len(obsm)) if 'CALIBRATE_POLARIZATION' in obsm[i]]
    polscans = ['scan_%i'%i for i in np.unique(scnum[np.in1d(stid,polst)])]
    scandur = int(np.median([summary[ss]['0']['EndTime'] - summary[ss]['0']['BeginTime'] for ss in polscans])*86400.)
    print('Will pre-average pol. data by %i seconds for calibration'%scandur)

    list_scans= list([i for i in np.unique(scnum[np.in1d(stid,polst)])])
    pol_scans=",".join(str(e) for e in list_scans)

    ###########################

    # Re-calibrate polarization calibrator:
    os.system('rm -rf '+msname+'.Gpol1')

    gaincal(vis = msname+'_polcalib.ms',
        caltable = '%s.Gpol1'%msname,
        field = polcalname,
        spw='', # all SPWs
        scan = pol_scans,
        solint = 'int',
        gaintype = 'G', 
        refant = refant, 
        refantmode='strict' )

    #plotcal(msname+'.Gpol1','scan','amp',field='',poln='/',subplot=111, figfile=msname+'.Gpol1.png')
    plotms(vis=msname+'.Gpol1', field='', xaxis='scan', yaxis='amp', correlation='/', coloraxis='spw', plotfile=msname+'.Gpol1.png', overwrite=True)



    ##########
    os.system('rm -rf '+msname+'.Gpol1a')

    # Save QU output in external file:
    orig_stdout = sys.stdout
    f = open('%s.QUfromGain.txt'%msname, 'w')
    sys.stdout = f

    # Rough estimate of QU:
    S1=polfromgain(vis=msname+'_polcalib.ms',
                   tablein='%s.Gpol1'%msname,
                   caltable='%s.Gpol1a'%msname)

    print(S1)

    sys.stdout = orig_stdout
    f.close()

    f = open('%s.QUfromGain.txt'%msname)
    print(f.read())
    f.close()
    ##########



    ##################################################
    # We search for the scan where the polarization signal is minimum in XX and YY
    # (i.e., maximum in XY and YX):

    tb.open('%s.Gpol1'%msname)
    scans = tb.getcol('SCAN_NUMBER')
    gains = np.squeeze(tb.getcol('CPARAM'))
    tb.close()
    scanlist = np.array(list(set(scans)))
    ratios = np.zeros(len(scanlist))
    for si, s in enumerate(scanlist):
        filt = scans == s
        ratio = np.sqrt(np.average(np.power(np.abs(gains[0,filt])/np.abs(gains[1,filt])-1.0,2.)))
        ratios[si] = ratio


    bestscidx = np.argmin(ratios)
    bestscan = scanlist[bestscidx]
    print('Scan with highest expected X-Y signal: '+str(bestscan))
    #####################################################



    # Cross-hand delay:
    os.system('rm -rf %s.Kcrs'%msname)
    gaincal(vis=msname+'_polcalib.ms',
        caltable='%s.Kcrs'%msname,
        selectdata=True,
        field = polcalname,
        spw='', # all
        scan=str(bestscan),
        gaintype='KCROSS',
        solint='inf'+avgstring,
        refant=refant, refantmode='strict', 
        smodel = [1,0,1,0],
        gaintable=['%s.Gpol1'%msname],
        interp=['linear'])


    # Cross-hand phase:
    os.system('rm -rf %s.Xfparang'%msname)

    orig_stdout = sys.stdout
    f = open('%s.PolFromPolcal.txt'%msname, 'w')
    sys.stdout = f

    S2=polcal(vis=msname+'_polcalib.ms',
              caltable='%s.Xfparang'%msname, 
              field= polcalname,
              spw='', # all
              scan = pol_scans,
              poltype='Xfparang+QU',
              solint='inf'+avgstring,
              combine='scan,obs',
              preavg=scandur, 
              smodel=S1[polcalname]['SpwAve'],
              gaintable=['%s.Gpol1'%msname,'%s.Kcrs'%msname],
              interp=['linear','nearest'])

    print(S2)
    
    sys.stdout = orig_stdout
    f.close()

    f = open('%s.PolFromPolcal.txt'%msname)
    print(f.read())
    f.close()

    #plotcal(msname+'.Xfparang','freq','phase',antenna='0',subplot=211)
    #plotcal(msname+'.Xfparang','chan','phase',antenna='0',subplot=212, figfile=msname+'.Xfparang.png')
    plotms(vis=msname+'.Xfparang', xaxis='freq', yaxis='phase', antenna='0', coloraxis='spw', gridrows=2, rowindex=0) 
    plotms(vis=msname+'.Xfparang', xaxis='chan', yaxis='phase', antenna='0', coloraxis='spw', gridrows=2, rowindex=1, plotindex=1, clearplots=False,
           plotfile=msname+'.Xfparang.png', overwrite=True)

    ## Re-calibrate polarization calibrator (with right pol. model): 
    os.system('rm -rf %s.Gpol2'%msname) 
    gaincal(vis=msname+'_polcalib.ms',
            caltable='%s.Gpol2'%msname, 
            field=polcalname,
            spw='', # all
            scan = pol_scans,
            solint='int',
            refant=refant, refantmode='strict',
            gaintype = 'G', 
            smodel=S2[polcalname]['SpwAve'],
            parang=True)

    # the revised polarisation calibration assuming the correct pol model (plot should be flat)
    #plotcal(msname+'.Gpol2','scan','amp',field='',poln='/',subplot=111, figfile=msname+'.Gpol2.png')
    plotms(vis=msname+'.Gpol2', field='', xaxis='scan', yaxis='amp', correlation='/', coloraxis='spw', plotfile=msname+'.Gpol2.png', overwrite=True)

    ##########
    ## Check for any residual polarization signal:
    os.system('rm -rf %s.Gpol2a'%msname) 

    orig_stdout = sys.stdout
    f = open('%s.QUfromGain.txt'%msname, 'a')
    f.write('\n\n USING POLCAL MODEL:\n\n')
    sys.stdout = f

    S1null=polfromgain(vis=msname+'_polcalib.ms',
                       tablein='%s.Gpol2'%msname,
                       caltable='%s.Gpol2a'%msname)

    print(S1null)

    sys.stdout = orig_stdout
    f.close()

    f = open('%s.QUfromGain.txt'%msname)
    print(f.read())
    f.close()
    ##########


    # Plot RMS of gain ratios around 1.0:

    tb.open('%s.Gpol2'%msname)
    scans2 = tb.getcol('SCAN_NUMBER')
    gains = np.squeeze(tb.getcol('CPARAM'))
    tb.close()

    scanlist2 = np.array(list(set(scans2)))
    ratios2 = np.zeros(len(scanlist2))
    for si, s in enumerate(scanlist2):
      filt = scans2 == s
      ratio = np.sqrt(np.average(np.power(np.abs(gains[0,filt])/np.abs(gains[1,filt])-1.0,2.)))
      ratios2[si] = ratio


    os.system('rm -rf %s.GainRatiosPol.png'%msname)
    pl.ioff()
    fig = pl.gcf()
    pl.clf() #pl.figure()
    sub = fig.add_subplot(111)
    sub.plot(scanlist, ratios, 'or', label = 'No Polcal')
    sub.plot(scanlist2, ratios2, 'ob', label = 'Polcal')
    sub.plot([scanlist[bestscidx]], [ratios[bestscidx]], 'xk', label = 'Best Pol. Scan')
    sub.set_xlabel('Scan Number')
    sub.set_ylabel('Gain Ratio RMS')
    pl.legend(numpoints=1)
    pl.savefig('%s.GainRatiosPol.png'%msname)
    pl.clf()

    pl.ion()


    # Calibrate D-terms:
    os.system('rm -rf %s.Df0gen*'%msname) 
    polcal(vis=msname+'_polcalib.ms',
       caltable='%s.Df0gen'%msname, 
       spw='', # all
       field= polcalname, 
       solint='inf'+avgstring, 
       scan = pol_scans,
       combine='obs,scan',
       preavg=scandur,
       poltype='Dflls',
       refant='', #solve absolute D-term
       smodel=S2[polcalname]['SpwAve'],
       gaintable=['%s.Gpol2'%msname, '%s.Kcrs'%msname, '%s.Xfparang'%msname],
       gainfield=['', '', ''],
       interp=['linear','nearest','nearest'])
      

    # Allow applying solutions to the parallel hands too:
    #aph.Dgen(dtab='%s.Df0'%msname, dout='%s.Df0gen'%msname)


    # commute Df0 from antenna to sky frame (for examination only)
    #aph.dxy(dtab='%s.Df0'%msname, xytab='%s.XY0'%msname, dout='%s.Df0sky'%msname)

    #  plotcal(msname+'.Df0sky', 'freq','real', spw='0,1', figfile='%s.Dterm_sky_real_spw0-1.png'%msname)
    #  plotcal(msname+'.Df0sky', 'freq','real', spw='2,3', figfile='%s.Dterm_sky_real_spw2-3.png'%msname)
    #  plotcal(msname+'.Df0sky', 'freq','imag', spw='0,1', figfile='%s.Dterm_sky_imag_spw0-1.png'%msname)
    #  plotcal(msname+'.Df0sky', 'freq','imag', spw='2,3', figfile='%s.Dterm_sky_imag_spw2-3.png'%msname)
      #    plotcal(msname+'.Df0sky', 'freq','imag', spw='4,5', figfile='%s.Dterm_sky_imag_spw4-5.png'%msname)

      # Save D-term plots for all antennas:
      #  plotcal(msname+'.Df0','chan','real', spw='0,1,2,3',
      #       iteration='spw',subplot=221,figfile='%s.Df0.plot.REAL.png'%(msname))

      #  plotcal(msname+'.Df0','chan','real', spw='0,1,2,3',
      #       iteration='spw',subplot=221,figfile='%s.Df0.plot.IMAG.png'%(msname))


    tb.open(msname+'/ANTENNA')
    allants = tb.getcol('NAME')
    tb.close()
    os.system('rm -rf %s.Df0.plots'%msname)
    os.system('mkdir %s.Df0.plots'%msname)
    for antnam in allants:
      #plotcal(msname+'.Df0gen','chan','real', antenna=antnam, spw='', # all 
      #        figfile='%s.Df0.plots/%s.Dterm.REAL.png'%(msname,antnam))
      
      #plotcal(msname+'.Df0gen','chan','imag', antenna=antnam, spw='', # all
      #        figfile='%s.Df0.plots/%s.Dterm.IMAG.png'%(msname,antnam))

      plotms(vis=msname+'.Df0gen', xaxis='chan', yaxis='real', antenna=str(antnam),  coloraxis='spw',
             plotfile='%s.Df0.plots/%s.Dterm.REAL.png'%(msname,antnam), overwrite=True, title=antnam+': Gain Real vs. Channel')

      plotms(vis=msname+'.Df0gen', xaxis='chan', yaxis='imag', antenna=str(antnam),  coloraxis='spw',
             plotfile='%s.Df0.plots/%s.Dterm.IMAG.png'%(msname,antnam), overwrite=True, title=antnam+': Gain Imag vs. Channel')


    # solve for global normalized gain amp (to get X/Y ratios) on pol calibrator

    # amp-only and normalized, so only X/Y amp ratios matter
    gaincal(vis=msname+'_polcalib.ms',
        caltable='%s.Gxyamp'%msname, 
        field=polcalname,
        solint='inf',
        combine='scan,obs',
        refant=refant, refantmode='strict',
        gaintype='G',
        scan = pol_scans,
        smodel=S2[polcalname]['SpwAve'],
        calmode='a',
        gaintable=['%s.Kcrs'%msname, '%s.Xfparang'%msname, '%s.Df0gen'%msname],
        gainfield=['', '', ''],
        interp=['nearest','nearest','nearest'],
        solnorm=True,
        parang=True)

    #plotcal(msname+'.Gxyamp', 'antenna','amp', spw='', iteration='spw',subplot=231, figfile='%s.Gxyamp.png'%msname)
    #plotcal(msname+'.Gxyamp', 'antenna','amp', spw='',poln='/', iteration='spw',subplot=231,figfile='%s.GxyampRatio.png'%msname)
    plotms(vis=msname+'.Gxyamp', xaxis='antenna1', yaxis='amp', coloraxis='antenna1', iteraxis='spw', gridrows=2, gridcols=3, 
           plotfile=msname+'.Gxyamp.png', overwrite=True)
    plotms(vis=msname+'.Gxyamp', xaxis='antenna1', yaxis='amp', correlation='/', coloraxis='antenna1', iteraxis='spw', gridrows=2, gridcols=3, 
           plotfile='%s.GxyampRatio.png'%msname, overwrite=True) 


#####################################################

def imagepolcal(fmode, cell, polcalnames, spw=''):
    """
    imagepolcal(fmode, cell, polcalnames, spw)

      Image the polarisation calibrator data of the given spectral mode.

      fmode = 'FDM' or 'TDM' - a marker to identify the data to image
      cell = the image cell size to use, e.g. '0.2arcsec'
      polcalnames = dictionary of the polcal field name for each MS
      spw = the SPWs to image (default: all)
    """

    msnames=glob.glob('*'+fmode)
    f = open('results_img1_'+fmode+'.txt','w')
    for msname in sorted(msnames):
      os.system('rm -rf '+msname+'_img1*')
      print('cleaning '+fmode+' '+msname)
      tclean(vis = msname,
             imagename = msname+'_img1',
             field = polcalnames[msname],
             stokes = 'IQUV',
             spw = '',
             outframe = 'LSRK',
             specmode = 'cont',
             nterms = 1,
             imsize = [256, 256],  # if you change this, also change the imfit parameters below
             cell = cell,
             deconvolver = 'clarkstokes',
             niter = 200,
             weighting = 'briggs',
             robust = 0.5,
             mask = '',
             gridder = 'standard',
             interactive = False
           )
      calstat=imstat(imagename=msname+'_img1.residual',
            axes=[0,1])
      rms=(calstat['rms'])
      prms = (rms[1]**2. + rms[2]**2.)**0.5

      os.system('rm -rf '+msname+'_Polcal.POLI')
      immath(outfile=msname+'_Polcal.POLI',
         mode='poli',
         imagename=msname+'_img1.image',
                    sigma='0.0Jy/beam')

      os.system('rm -rf '+msname+'_Polcal.POLA')
      immath(outfile=msname+'_Polcal.POLA',
         mode='pola',
         imagename=msname+'_img1.image',
         polithresh='%.8fJy/beam'%(5.0*prms))


      cleanimage=msname+'_img1.image'
      os.system('rm -rf '+msname+'.StokesI')
      immath(imagename=cleanimage,outfile=msname+'.StokesI',expr='IM0',stokes='I')
      os.system('rm -rf '+msname+'.StokesQ')
      immath(imagename=cleanimage,outfile=msname+'.StokesQ',expr='IM0',stokes='Q')
      os.system('rm -rf '+msname+'.StokesU')
      immath(imagename=cleanimage,outfile=msname+'.StokesU',expr='IM0',stokes='U')
      os.system('rm -rf '+msname+'.StokesV')
      immath(imagename=cleanimage,outfile=msname+'.StokesV',expr='IM0',stokes='V')


      resI=imfit(imagename = msname+'.StokesI', box = '110,110,145,145')
      resQ=imfit(imagename = msname+'.StokesQ', box = '115,115,130,130')
      resU=imfit(imagename = msname+'.StokesU', box = '110,110,145,145')

      # and then we extract the flux and error values for each Stokes

      fluxI=resI['results']['component0']['flux']['value'][0]
      errorI=resI['results']['component0']['flux']['error'][0]

      fluxQ=resQ['results']['component0']['flux']['value'][1]
      errorQ=resQ['results']['component0']['flux']['error'][1]

      fluxU=resU['results']['component0']['flux']['value'][2]
      errorU=resU['results']['component0']['flux']['error'][2]


      #Now we use these values to compute polarization intensity, angle and ratio, and their errors:


      fluxPI  = math.sqrt( fluxQ**2 + fluxU**2 )
      errorPI = math.sqrt( (fluxQ*errorU)**2 + (fluxU*errorQ)**2 ) / fluxPI

      fluxPImjy = 1000*fluxPI
      errPImjy  = 1000*errorPI

      polRatio = fluxPI / fluxI
      errPRat  = polRatio * math.sqrt( (errorPI/fluxPI)**2 + (errorI/fluxI)**2 )

      polAngle = 0.5 * math.degrees( math.atan2(fluxU,fluxQ) )
      errPA    = 0.5 * math.degrees( errorPI / fluxPI )


      print('Pol ratio of Polarization calibrator: '+str(polRatio))
      print('Pol angle of Polarization calibrator: '+str(polAngle))

      #Pol ratio of Polarization calibrator:
      #Pol angle of Polarization calibrator:

      f.write('\n Image: '+msname+'\n')
      f.write( 'Pol ratio %8.5f \n'%(polRatio))
      f.write( 'Pol angle %8.5f \n'%(polAngle))


    f.close()
    os.system('cat results_img1_'+fmode+'.txt')


####################################################
####################################################


mystep = 0
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  for mymsname in sorted(EBs.keys()):
    print("Working on "+mymsname)
    os.system('rm -rf %s*'%mymsname)
    concat(vis=EBs[mymsname], concatvis=mymsname)
    listobs(vis=mymsname, listfile=mymsname+'.listobs')

    #check which datacolumn to use
    tb.open(mymsname)
    if 'CORRECTED_DATA' in tb.colnames():
      thedatacol = 'corrected'
    else:
      thedatacol = 'data'
    tb.close()

    os.system('rm -rf %s.TDM'%mymsname) 
    os.system('rm -rf %s.FDM'%mymsname) 

    if len(tdmspws)>0 and len(fdmspws)>0:
      # separate FDM and TDM
      print("Separating FDM and TDM SPWs ...")
      mstransform(vis=mymsname, spw=tdmspws, outputvis='%s.TDM'%mymsname, datacolumn=thedatacol)
      mstransform(vis=mymsname, spw=fdmspws, outputvis='%s.FDM'%mymsname, datacolumn=thedatacol)
    elif len(tdmspws)>0: # only have TDM
      mstransform(vis=mymsname, spw=tdmspws, outputvis='%s.TDM'%mymsname, datacolumn=thedatacol)
    else: # only have FDM
      mstransform(vis=mymsname, spw=fdmspws, outputvis='%s.FDM'%mymsname, datacolumn=thedatacol)

    # plots of amp vs. parangle for checking total intensity calibration and parangle coverage
    for myspw in tdmspws.split(',')+fdmspws.split(','):
      plotms(vis=mymsname, field=polcalib[mymsname+'.TDM'], spw=str(myspw), xaxis='parang', yaxis='amp', 
             correlation='XX,YY', ydatacolumn=thedatacol, showgui=True,
             averagedata=True, avgchannel='9999', avgbaseline=True, coloraxis='corr',
             plotfile=mymsname+".amp-vs-pa.spw"+str(myspw)+".png"
             )


mystep = 1
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    os.system('rm -rf '+msname+'_polcalib.ms')

    #check which datacolumn to use (for the case there was no mstransform in step 0)
    tb.open(msname)
    if 'CORRECTED_DATA' in tb.colnames():
      thedatacol = 'corrected'
    else:
      thedatacol = 'data'
    tb.close()

    mstransform(vis=msname, field=polcalib[msname], outputvis=msname+'_polcalib.ms',
                datacolumn=thedatacol)


    # Re-calibrate polarization calibrator:
    os.system('rm -rf %s.Gpol1'%msname)
    mysession = msname[msname.find('_S')+1:msname.find('.ms')]
    print("Using refant "+refants[mysession]+" for session "+mysession)
    gaincal(vis = msname+'_polcalib.ms',
        caltable = '%s.Gpol1'%msname,
        field = polcalib[msname],
        solint = 'int',
        gaintype = 'G', 
        refant = refants[mysession],
        refantmode='strict' )

    plotms(vis=msname+'.Gpol1', xaxis='scan', yaxis='amp', correlation='/', coloraxis='spw', plotfile=msname+'.Gpol1.png', overwrite=True)


# Additional flagging (for outliers in cross-hand amplitude and/or phase):
# mostly based on the Gpol1 gains 

mystep = 2
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  ## Example code:
  # msname1 = 'concat_S1.ms.TDM'
  #msname2 = 'concat_S2.ms.TDM'
  #
  #flagmanager(vis=msname1,mode='save',versionname='Original')
  #flagmanager(vis=msname2,mode='save',versionname='Original')
  #
  #flagdata(vis=msname1, mode='manual', scan='73') # flag out scan for test
  #flagdata(vis=msname1, mode='manual', spw='', antenna='DA42, DA51,DA65, DV06,DV13,DV25', scan='46')
  #
  #flagdata(vis=msname2, mode='manual', spw='', antenna='DA57')
  #flagdata(vis=msname2, mode='manual', spw='', antenna='DA62')
 
  # comment in the following mstransform if you have any flagging here
 
  #nameTDM=glob.glob('*TDM')
  #for msname in sorted(nameTDM):
   # print("Working on "+msname)
  # os.system('rm -rf '+msname+'_polcalib.ms')
  #mstransform(vis=msname, field=polcalib[msname], outputvis=msname+'_polcalib.ms',
   #             datacolumn='data') # or 'corrected' if present

# Polarization calibration:
mystep = 3
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    mysession = msname[msname.find('_S')+1:msname.find('.ms')]
    print("Using refant "+refants[mysession]+" for session "+mysession)
    polcalcore(msname, refants[mysession], polcalib[msname], '')


# Save flags before applycal
mystep = 4
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    flagmanager(vis = msname,
                mode = 'save',
                versionname = 'BeforeApplycal')
  
# Application of the polarization calibration to the calibrator
mystep = 5
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    applycal(vis = msname,
      field = polcalib[msname],
      gaintable = ['%s.Gpol2'%msname,'%s.Kcrs'%msname, '%s.Xfparang'%msname, '%s.Df0gen'%msname],
      gainfield = [polcalib[msname],polcalib[msname], polcalib[msname], polcalib[msname]],
      interp = ['linear','nearest','nearest','nearest'],
      calwt = False,
      parang = True,
             flagbackup = False)

    os.system('rm -rf %s.polcal.XXYY.revsim.corrected.png'%msname)
    plotms(vis = msname, xaxis = 'real', xdatacolumn = 'corrected', yaxis = 'imag', ydatacolumn = 'corrected', field = polcalib[msname], spw = '', correlation = 'XX,YY', averagedata = True, avgchannel = '4000', avgtime = '1000', coloraxis = 'corr', plotfile = msname+'.polcal.XXYY.revsim.corrected.png')
    os.system('rm -rf %s.polcal.XYYX.revsim.corrected.png'%msname)
    plotms(vis = msname, xaxis = 'real', xdatacolumn = 'corrected', yaxis = 'imag', ydatacolumn = 'corrected', field = polcalib[msname], spw = '', correlation = 'XY,YX', averagedata = True, avgchannel = '4000', avgtime = '1000', coloraxis = 'corr', plotfile = msname+'.polcal.XYYX.revsim.corrected.png')  

  

# Image of the polarization calibrator
mystep = 6
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  imagepolcal(fmode='TDM', cell=diagnostic_image_cellsize, polcalnames=polcalib, spw='')


# Application of the polarization calibration to all targets
mystep = 7
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])
  
  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    for i in  set(targets+[phasecal[msname], bandpasscal[msname]]): 
      applycal(vis = msname,
               field = str(i),
               gaintable = ['%s.Gxyamp'%msname, 
                   '%s.Kcrs'%msname, '%s.Xfparang'%msname, '%s.Df0gen'%msname],
               gainfield = [polcalib[msname],polcalib[msname], polcalib[msname], polcalib[msname]],
               interp = ['nearest','nearest','nearest','nearest'],
               calwt = False,
               parang = True,
               flagbackup = False)


# Save flags after applycal
mystep = 8
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)
    flagmanager(vis = msname,
                mode = 'save',
                versionname = 'AfterApplycal')


# Split out corrected column
mystep = 9
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameTDM=glob.glob('*TDM')
  for msname in sorted(nameTDM):
    print("Working on "+msname)

    os.system('rm -rf %s.cal'%msname) 
    os.system('rm -rf %s.cal.flagversions'%msname) 
    split(vis = msname,
          field = ','.join(set(targets+[phasecal[msname],bandpasscal[msname],polcalib[msname]])),
          outputvis = '%s.cal'%msname,
          datacolumn = 'corrected',
          keepflags = True)
  
  
###################################################



mystep = 10
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])


  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
    os.system('rm -rf '+msname+'_polcalib.ms')

    #check which datacolumn to use (for the case there was no mstransform in step 0)
    tb.open(msname)
    if 'CORRECTED_DATA' in tb.colnames():
      thedatacol = 'corrected'
    else:
      thedatacol = 'data'
    tb.close()

    mstransform(vis=msname, field=polcalib[msname], outputvis=msname+'_polcalib.ms',
                datacolumn=thedatacol)
            
    # Re-calibrate polarization calibrator:
    os.system('rm -rf %s.Gpol1'%msname)
    mysession = msname[msname.find('_S')+1:msname.find('.ms')]
    print("Using refant "+refants[mysession]+" for session "+mysession)
    gaincal(vis = msname+'_polcalib.ms',
        caltable = '%s.Gpol1'%msname,
        field = polcalib[msname],
        solint = 'int',
        gaintype = 'G', 
        refant = refants[mysession], 
        refantmode='strict' )

    plotms(vis=msname+'.Gpol1', xaxis='scan', yaxis='amp', correlation='/', coloraxis='spw', plotfile=msname+'.Gpol1.png', overwrite=True)



# Additional flagging (for outliers in cross-hand amplitude and/or phase):
# mostly based on the Gpol1 gains 

mystep = 11
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  # Examples: 
  #flagmanager(vis='concat_S1.ms.FDM',mode='save',versionname='beforeStep11')
  #flagdata(vis='concat_S1.ms.FDM', mode='manual', scan='73') # flag for test 

  # comment in the following mstransform if you have any flagging here

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
  
    os.system('rm -rf '+msname+'_polcalib.ms')
    mstransform(vis=msname, field=polcalib[msname], outputvis=msname+'_polcalib.ms',
                datacolumn='data')



mystep = 12
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
    mysession = msname[msname.find('_S')+1:msname.find('.ms')]
    print("Using refant "+refants[mysession]+" for session "+mysession)
    polcalcore(msname, refants[mysession], polcalib[msname], '5MHz')


mystep = 13
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
    flagmanager(vis = msname,
                mode = 'save',
                versionname = 'BeforeApplycal')
  
# Application of the polarization calibration to the calibrator
mystep = 14
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
    applycal(vis = msname,
      field = polcalib[msname],
      gaintable = ['%s.Gpol2'%msname,'%s.Kcrs'%msname, '%s.Xfparang'%msname, '%s.Df0gen'%msname],
      gainfield = [polcalib[msname],polcalib[msname], polcalib[msname], polcalib[msname]],
      interp = ['linear','nearest','nearest','nearest'],
      calwt = False,
      parang = True,
      flagbackup = False)

    os.system('rm -rf %s.polcal.XXYY.revsim.corrected.png'%msname)
    plotms(vis = msname, xaxis = 'real', xdatacolumn = 'corrected', yaxis = 'imag', ydatacolumn = 'corrected', field = polcalib[msname], spw = '', correlation = 'XX,YY', averagedata = True, avgchannel = '4000', avgtime = '1000', coloraxis = 'corr', plotfile = msname+'.polcal.XXYY.revsim.corrected.png')
    os.system('rm -rf %s.polcal.XYYX.revsim.corrected.png'%msname)
    plotms(vis = msname, xaxis = 'real', xdatacolumn = 'corrected', yaxis = 'imag', ydatacolumn = 'corrected', field = polcalib[msname], spw = '', correlation = 'XY,YX', averagedata = True, avgchannel = '4000', avgtime = '1000', coloraxis = 'corr', plotfile = msname+'.polcal.XYYX.revsim.corrected.png')  

  
# Image of the polarization calibrator
mystep = 15
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  imagepolcal(spw='', cell=diagnostic_image_cellsize, polcalnames=polcalib, fmode='FDM')


# Application of the polarization calibration to all targets
mystep = 16
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)

    for i in  set(targets+[phasecal[msname], bandpasscal[msname]]): 
      applycal(vis = msname,
               field = str(i),
               gaintable = ['%s.Gxyamp'%msname, 
                   '%s.Kcrs'%msname, '%s.Xfparang'%msname, '%s.Df0gen'%msname],
               gainfield = [polcalib[msname], polcalib[msname], polcalib[msname], polcalib[msname]],
               interp = ['nearest','nearest','nearest','nearest'],
               calwt = False,
               parang = True,
               flagbackup = False)
  


# Save flags after applycal
mystep = 17
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)
    flagmanager(vis = msname,
                mode = 'save',
                versionname = 'AfterApplycal')


# Split out corrected column
mystep = 18
if(mystep in thesteps):
  casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
  print('Step '+str(mystep)+' '+step_title[mystep])

  nameFDM=glob.glob('*FDM')
  for msname in sorted(nameFDM):
    print("Working on "+msname)

    os.system('rm -rf %s.cal'%msname) 
    os.system('rm -rf %s.cal.flagversions'%msname) 
    split(vis = msname,
          field = ','.join(set(targets+[phasecal[msname],bandpasscal[msname],polcalib[msname]])),
          outputvis = '%s.cal'%msname,
          datacolumn = 'corrected',
          keepflags = True)
  
  


# This is the template file for manual calibration

# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2019.09.30: first release
#   2020.04.11: update for 

import re
import os

from plot_utils import check_cal, check_info

################ Basic Information ###############
# CALIBRATE_AMPLI: 
# CALIBRATE_ATMOSPHERE: 
# CALIBRATE_BANDPASS: 
# CALIBRATE_FLUX: Mars
# CALIBRATE_FOCUS: 
# CALIBRATE_PHASE: 
# CALIBRATE_POINTING: 
# OBSERVE_TARGET: 


# Pre-requisite
step_title = {0: 'Import of the ASDM',
              1: 'Fix old cycle problems',
              2: 'listobs',
              3: 'A priori flagging',
              4: 'Generation and time averaging of the WVR cal table',
              5: 'Generation of the Tsys cal table',
              6: 'Generation of the antenna position cal table',
              7: 'Application of the WVR, Tsys and antpos cal tables (+plots)',
              8: 'Split out science SPWs and time average',
              9: 'Listobs, clear pointing table, and save original flags',
              10: 'Initial flagging',
              11: 'Putting a model for the flux calibrator(s)',
              12: 'Save flags before bandpass cal',
              13: 'Bandpass calibration',
              14: 'Save flags before gain cal',
              15: 'Gain calibration',
              16: 'Save flags before applycal',
              17: 'Application of the bandpass and gain cal tables (+plots)',
              18: 'Split out corrected column',
              19: 'Save flags after applycal'}
thesteps = step_title.keys()

# Pre-defined variable
target_name = 'Centaurus_A' # the name used as the field name
refer_antenna = 'CM05' # the reference antenna
uidname = 'uid___A002_X83f101_X49a'
uidshortname = 'X49a'
plotdir = './'+uidshortname
msfile = uidname + '.ms'
splitmsfile = uidname + '.ms.split'
calibrator_name = 'Mars'
plot_results = True

# variable defined for interactive runing
try:
    mysteps
except:
    mysteps = []


# pre-define helper function
def step2log(mystep):
    # put the step information into the log and terminal
    if mystep in thesteps:
        casalog.post('Step '+str(mystep)+' '+step_title[mystep],'INFO')
        print('Step ', mystep, step_title[mystep])

def initial_runstep(mysteps):
    """generate the runsteps from input
    
    Parameters
    ----------
    mystep : list or str
        Can either be a list contains integers or a string for the steo range
        example: [0,1,2,3], '3~19'
    """
    if mysteps == []:
        runsteps = list(range(0, 20))
    if isinstance(mysteps, list):
        for i in mysteps:
            if (not isinstance(i, int)) or (i > 19) or (i < 0):
                raise ValueError("Invalide steps parameter! Checking the doc.")
            else:
                runsteps = mysteps
    if isinstance(mysteps, str):
        try: 
            steps_range = mysteps.split('~')
            runsteps = list(range(int(steps_range[0]), int(steps_range[-1])+1))
        except:
            raise ValueError("Invalide steps parameter! Checking the doc.")

    return runsteps

########################################################
# star runsteps
runsteps = initial_runstep(mysteps)

######################## Setp 0 #########################
# Import of the ASDM
runstep = 0
if runstep in runsteps:
    step2log(runstep)
    if os.path.exists(msfile) == False:
        importasdm(asdm='../raw/'+uidname + '.asdm.sdm', vis=msfile, 
                    asis='Antenna Station Receiver Source CalAtmosphere CalWVR')

######################## Step 1 ###########################
# Fix old cycle problems
runstep = 1
if runstep in runsteps:
    step2log(runstep)
    
    if False: # No issue found
        es.fixForCSV2555(msfile)

    # Fix of SYSCAL table times
    from recipes.almahelpers import fixsyscaltimes
    fixsyscaltimes(vis = msfile)
    
    # Fix the 0,0 coordinates of planets
    fixplanets(vis = msfile,
        field = '2', # flux calibrator
        fixuvw = True)
  
######################## Step 2 ###########################
# listobs
runstep = 2
if runstep in runsteps:
    step2log(runstep)

    os.system('rm -rf {}.ms.listobs.txt'.format(uidname))
    listobs(vis = msfile,
            listfile = msfile+'.listobs.txt')

    if True:
        check_info(msfile)


######################## Step 3 ###########################
# A priori flagging
runstep = 3
if runstep in runsteps:
    step2log(runstep)

    # flag shadowed data by nearby antennas
    flagdata(vis = msfile, mode = 'shadow', flagbackup = False)
    # flag autocorrelation data, only cross-corrected data is needed
    flagdata(vis = msfile, autocorr = True, flagbackup = False)
    # flag POINTING and ATMOSPHERE calibrations
    flagdata(vis = msfile, mode = 'manual', flagbackup = False, 
             intent = '*POINTING*,*SIDEBAND_RATIO*,*ATMOSPHERE*')
    # Store the priori flags
    flagmanager(vis = msfile, mode = 'save', versionname = 'Priori')
    
######################## Step 4 ###########################
# Generation and time averaging of the WVR cal table
runstep = 4
if runstep in runsteps:
    step2log(runstep)

    # Already calculated the WVR
    # wvrgcal(vis=msfile,
    #          caltable = msfile+'.wvrgal')

######################## Step 5 ###########################
# Generation of the Tsys cal table
runstep = 5
if runstep in runsteps:
    step2log(runstep)

    os.system('rm -rf {}.ms.tsys'.format(uidname))
    gencal(vis = msfile,
           caltable = msfile+'.tsys',
           caltype = 'tsys')
     
######################## Step 6 ###########################
# Generation of the antenna position cal table
runstep = 6
if runstep in runsteps:
    step2log(runstep)

    # Position for antenna CM05 is derived from baseline run made on 2014-06-02 04:15:36.
    os.system('rm -rf {}.ms.antpos'.format(uidname)) 
    gencal(vis = msfile,
           caltable = '{}.ms.antpos'.format(uidname),
           caltype = 'antpos',
           antenna = refer_antenna,
           parameter = [0,0,0])
           #  parameter = [8.01868736744e-07,-9.60193574429e-07,-5.20143657923e-07])
  

######################## Step 7 ###########################
# Application of the WVR, Tsys and antpos cal tables
runstep = 7
if runstep in runsteps:
    step2log(runstep)

    from recipes.almahelpers import tsysspwmap
    tsysmap = tsysspwmap(vis = '{}.ms'.format(uidname), 
                         tsystable = '{}.ms.tsys'.format(uidname))
    
    applycal(vis = msfile,
             field = '0',
             spw = '16,18,20,22',
             gaintable = [msfile+'.tsys', msfile+'.antpos'],
             gainfield = ['0', ''],
             interp = ['linear', ''],
             spwmap = [tsysmap,[]],
             calwt = True,
             flagbackup = False)
  
    # Note: J1256-0547 didn't have any Tsys measurement, and I couldn't find any close measurement. But this is not a science target, so this is probably Ok.
  
    applycal(vis = msfile,
             field = '2',
             spw = '16,18,20,22',
             gaintable = [msfile+'.tsys', msfile+'.antpos'],
             gainfield = ['2', ''],
             interp = ['linear', ''],
             spwmap = [tsysmap,[]],
             calwt = True,
             flagbackup = False)
  
    # Note: J1321-4342 didn't have any Tsys measurement, so I used the one made on Centaurus_A. This is probably Ok.
    applycal(vis = msfile,
             field = '3',
             spw = '16,18,20,22',
             gaintable = [msfile+'.tsys', msfile+'.antpos'],
             gainfield = ['4', ''],
             interp = ['linear', ''],
             spwmap = [tsysmap,[]],
             calwt = True,
             flagbackup = False)
  
    applycal(vis = msfile,
             field = '4~22',
             spw = '16,18,20,22',
             gaintable = [msfile+'.tsys', msfile+'.antpos'],
             gainfield = ['4', ''],
             interp = ['linear', ''],
             spwmap = [tsysmap,[]],
             calwt = True,
             flagbackup = False)

    # check the data after the WVR, Tsys and Antenna calibration
    if plot_results:
        check_cal(vis=msfile, fdmspw='16',
                  calibrator_fields='0,2,3',
                  refant='CM03', plotdir=plotdir+'/AfterTsysCalibration', 
                  science_field='Cen*', flux_calibrator='Mars', 
                  tdmspws=['8','10','12','14'], 
                  plot_time=True, plot_freq=True, plot_tsys=True) 

######################## Step 8 ###########################
# Split out science SPWs and time average
runstep = 8
if runstep in runsteps:
    step2log(runstep)

    os.system('rm -rf {}.split'.format(msfile)) 
    os.system('rm -rf {}.split.flagversions'.format(msfile)) 
    split(vis = msfile,
          outputvis = splitmsfile,
          datacolumn = 'corrected',
          spw = '16,18,20,22',
          keepflags = False)


######################## Step 9 ###########################
# Listobs, clear pointing table, and save original flags
runstep = 9
if runstep in runsteps:
    step2log(runstep)

    os.system('rm -rf {}.listobs.txt'.format(splitmsfile))
    listobs(vis = splitmsfile,
            listfile = splitmsfile + '.listobs.txt')
  
    # Removing the POINTING tables
    if True:
        tb.open(splitmsfile+'/POINTING', nomodify = False)
        a = tb.rownumbers()
        tb.removerows(a)
        tb.close()
  
    if not os.path.exists('{}.flagversions/Original.flags'.format(splitmsfile)):
        flagmanager(vis = splitmsfile,
                    mode = 'save',
                    versionname = 'Original')

######################## Step 10 ###########################
# Initial flagging
runstep = 10
if runstep in runsteps:
    step2log(runstep)
    try:
      if interactive_flag != True:
        interactive_flag = False
    except:
      interactive_flag = False

    # Flagging edge channels
    flagdata(vis = splitmsfile,
             mode = 'manual',
             spw = '1:0~6;116~123,2:0~6;116~123,3:0~6;116~123', # 5% of the total channels
             flagbackup = False)
  
    ########## starting addtional flagging here ############
    # flagging spectral line in Mars 
    flagdata(vis = splitmsfile,
             mode = 'manual',
             field = '2', #Mars
             spw = '0:2400~3200',
             flagbackup = False)
    
######################## Step 11 ###########################
# Putting a model for the flux calibrator(s)
runstep = 11
if runstep in runsteps:
    step2log(runstep)

    setjy(vis = splitmsfile,
          field = '2', # Mars
          spw = '0,1,2,3',
          standard = 'Butler-JPL-Horizons 2012')
  
######################## Step 12 ###########################
# Save flags before bandpass cal
runstep = 12
if runstep in runsteps:
    step2log(runstep)
  
    flagmanager(vis = splitmsfile,
                mode = 'save',
                versionname = 'BeforeBandpassCalibration')
  

######################## Step 13 ###########################
# Bandpass calibration
runstep = 13
if runstep in runsteps:
    step2log(runstep)

    refer_antenna = 'CM03' # change antenna matters?
  
    os.system('rm -rf {}.bpphase_int.cal'.format(splitmsfile)) 
    gaincal(vis = splitmsfile,
            caltable = splitmsfile+'.bpphase_int.cal',
            field = '0', # bandpass calibrator
            spw = '0:1638~2457,1:49~74,2:49~74,3:49~74',
            solint = 'int',
            refant = refer_antenna,
            calmode = 'p')
  
    #if applyonly != True: es.checkCalTable('uid___A002_X83f101_X165.ms.split.ap_pre_bandpass', msName='uid___A002_X83f101_X165.ms.split', interactive=False)

    os.system('rm -rf {}.bandpass.cal'.format(splitmsfile)) 
    bandpass(vis = splitmsfile,
             caltable = splitmsfile+'.bandpass.cal',
             field = '0', # bandpass calibrator
             solint = 'inf',
             combine = 'scan',
             refant = refer_antenna,
             solnorm = True,
             bandtype = 'B',
             gaintable = splitmsfile+'.bpphase_int.cal')
  
     #if applyonly != True: es.checkCalTable('uid___A002_X83f101_X165.ms.split.bandpass', msName='uid___A002_X83f101_X165.ms.split', interactive=False) 
  
######################## Step 14 ###########################
# Save flags before gain cal
runstep = 14
if runstep in runsteps:
    step2log(runstep)
  
    flagmanager(vis = splitmsfile,
                mode = 'save',
                versionname = 'BeforeGainCalibration')
    
######################## Step 15 ###########################
# Gain calibration
runstep = 15
if runstep in runsteps:
    step2log(runstep)

    #refer_antenna = 'CM03' # change antenna matters?

    os.system('rm -rf {}.phase_int.cal'.format(splitmsfile)) 
    gaincal(vis = splitmsfile,
            caltable = splitmsfile+'.phase_int.cal',
            field = '0~0,2~3', # J1427-4206,Mars,J1321-4342
            solint = 'int',
            refant = refer_antenna,
            calmode = 'p',
            gaintable = splitmsfile+'.bandpass.cal')
  
    #if applyonly != True: es.checkCalTable('uid___A002_X83f101_X165.ms.split.phase_int', msName='uid___A002_X83f101_X165.ms.split', interactive=False) 
  
    os.system('rm -rf {}.phase_inf.cal'.format(splitmsfile)) 
    gaincal(vis = splitmsfile,
            caltable = splitmsfile+'.phase_inf.cal',
            field = '0~0,2~3', # J1427-4206,Mars,J1321-4342
            solint = 'inf',
            refant = refer_antenna,
            calmode = 'p',
            gaintable = splitmsfile+'.bandpass.cal')
    #if applyonly != True: es.checkCalTable('uid___A002_X83f101_X165.ms.split.ampli_inf', msName='uid___A002_X83f101_X165.ms.split', interactive=False) 

    os.system('rm -rf {}.amp.cal'.format(splitmsfile)) 
    gaincal(vis = splitmsfile,
            caltable = splitmsfile+'.amp.cal',
            field = '0~0,2~3', # J1427-4206,Mars,J1321-4342
            solint = 'inf',
            refant = refer_antenna,
            calmode = 'ap',
            gaintable = [splitmsfile+'.bandpass.cal', splitmsfile+'.phase_int.cal'])
  
    #if applyonly != True: es.checkCalTable('uid___A002_X83f101_X165.ms.split.phase_inf', msName='uid___A002_X83f101_X165.ms.split', interactive=False) 
  
    os.system('rm -rf {}.flux.cal'.format(splitmsfile)) 
    fluxscale(vis = splitmsfile,
              caltable = splitmsfile+'.amp.cal',
              fluxtable = splitmsfile+'.flux.cal',
              reference = '2') # Mars
  
    #if applyonly != True: es.fluxscale2(caltable = 'uid___A002_X83f101_X165.ms.split.ampli_inf', removeOutliers=True, msName='uid___A002_X83f101_X165.ms', writeToFile=True, preavg=10000)
  
######################## Step 16 ###########################
# Save flags before applycal
runstep = 16
if runstep in runsteps:
    step2log(runstep)
  
    flagmanager(vis = splitmsfile,
                mode = 'save',
                versionname = 'BeforeApplycal')
  
######################## Step 17 ###########################
# Application of the bandpass and gain cal tables
runstep = 17
if runstep in runsteps:
    step2log(runstep)

    for i in ['0', '2']: # J1427-4206,Mars
        applycal(vis = splitmsfile,
                 field = i,
                 gaintable = [splitmsfile+'.bandpass.cal', 
                              splitmsfile+'.phase_int.cal', 
                              splitmsfile+'.flux.cal'],
                 gainfield = ['', i, i],
                 interp = ['nearest', 'nearest', 'nearest'],
                 calwt = False,
                 flagbackup = False)
    applycal(vis = splitmsfile,
             field = '3', # J1321-4342, Centaurus_A
             gaintable = [splitmsfile+'.bandpass.cal', 
                          splitmsfile+'.phase_inf.cal', 
                          splitmsfile+'.flux.cal'],
             gainfield = ['', '3', '3'], # J1321-4342
             interp = ['nearest', 'nearest', 'nearest'],
             calwt = False,
             flagbackup = False)
    applycal(vis = splitmsfile,
             field = '4~22', # J1321-4342, Centaurus_A
             gaintable = [splitmsfile+'.bandpass.cal', 
                          splitmsfile+'.phase_inf.cal', 
                          splitmsfile+'.flux.cal'],
             gainfield = ['', '3', '3'], # J1321-4342
             interp = ['nearest', 'linear', 'linear'],
             calwt = False,
             flagbackup = False)
  
    if plot_results:
        check_cal(vis=splitmsfile, fdmspw='0', calibrator_fields=['0','2','3'], 
                  refant=refer_antenna, 
                  plotdir=plotdir+'/AfterBandpassCalibration', 
                  science_field='Cen*',
                  flux_calibrator='Mars', plot_time=True, plot_freq=True, 
                  plot_uvdist=True, plot_solutions=True, plot_target=True)

######################## Step 18 ###########################
# Split out corrected column
runstep = 18
if runstep in runsteps:
    step2log(runstep)

    os.system('rm -rf {}.calibrated'.format(splitmsfile)) 
    split(vis = splitmsfile,
          outputvis = splitmsfile+'.calibrated',
          datacolumn = 'corrected',
          keepflags = True)
  

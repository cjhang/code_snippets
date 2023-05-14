# main calibration for ALMA
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2019.09.30 First release for learning casa
#   2020.04.11 Update for reduce Centaurus_A
#   2022.03.26 Update for casa6 and include polarisation calibration


################ Basic Information ###############
# CALIBRATE_BANDPASS: 
# CALIBRATE_GAIN: 
# CALIBRATE_FLUX:
# OBSERVE_TARGET: 

#######################################################
#                  Data Preparation
#######################################################
data_dir = './data_cal'
name = '*.ms'
vis = os.path.join(data_dir, name)

#> define the basic information
polcal_field = 'J0006-0623'
bandpass_field = 'J2258-2758'
fluxcal_field = 'J2258-2758'
gaincal_field = 'J0038-2459'
target_field = 'NGC_253'
fields_unique = [bandpass_field, gaincal_field, polcal_field, target_field]
fields_tied_wvr = [target_field, gaincal_field] # the first one has wvr info, check CALIBRATE_ATMOSPHERE

refant = 'DV23'
mysolint = 'int'    # integration time
scan_kcross = '27'  # the scan with maximum cross-hand contribution


#######################################################
#           Pre-defined functions   
#######################################################
def search_scan(gainpol, mode='kcross'):
    """search scan with given mode
    Params:
        kcross: search the scan where the polarisation signal is minimum in XX and YY
    """
    if mode == 'kcross':
        tb.open(gainpol)
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
    return bestscan
    

#######################################################
#           Polarisation calibration   
#######################################################

from casarecipes.almapolhelpers import polfromgain


if True:
    obs = os.path(data_dir, name)

    # gaincal to absorb polarisation signal of polcal as function of parallactic angle
    obs_apint_pcal = name + '.apint.pcal'
    os.system('rm -rf {}'.format(obs_apint_pgcal))
    gaincal(vis = obs,
            caltable = obs_apint_pgcal,
            field = polcal_field,
            # scan = '', set the scan to polarisation scans
            solint = 'int',
            refant = refant,
            refantmode = 'strict',
            gaintype = 'G',
            smodel=[1,0,0,0],)
    # apply the calibration
    # applycal(vis=obs, 
             # field=polcal_field,
             # calwt=True,
             # gaintable=[obs_apint_pcal],
             # interp=['linear'], parang=False)

    # plot the complex polarisation ratio
    plotms(vis=obs_apint_pcal, xaxis='scan', yaxis='amp',
           coloraxis='spw', correlation='/', showgui=True)
    # test the combine two gain solution

    # extract the polarisation signal from gain table, rough estimation
    S1 = polfromgain(vis=obs_pol,
                     tablein=obs_apint_pcal,
                     #caltable=obs_apint_pcal+'2'
                     )

    # TODO, add the code to find the scan with minimm polarisation
    tb.open(obs_apint_pgcal)
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
    bestscan = str(scanlist[bestscidx])



    # cross-hand delay correction, derive a linear phase slope as a function of frequency
    # present when there are more than one channels, cross-hand delay is a function of frequency
    # plot the cross-hand delay
    # plotms(vis=obs, xaxis='freq', yaxis='hase', avgtime='1e9', correlation='XY.YX', spw='', 
    #        antenna=refant, iteraxis='baseline', coloraxis='corr')
    #
    obs_kcross_pcal = name + '.kcross.pcal'
    os.system('rm -rf {}'.format(obs_kcross_pcal))
    gaincal(vis=obs,
            caltable=obs_kcross_pcal,
            selectdata=True,
            scan=bestscan,
            gaintype='KCROSS',
            solint='inf',refant=refant, 
            refantmode='strict',
            smodel=[1,0,1,0], # force a polarized model
            gaintable=[obs_apint_pcal,],
            interp=['linear'])
    
    # applycal(vis=obs, 
         # field=polcal_field,
         # calwt=True,
         # gaintable=[obs_apint_pcal, obs_kcross_pcal],
         # interp=['linear', 'nearest'])
    
    # correct XY-phase absolute offset
    # the kcross correction have solved the frequency related variation, so the channels averaging is
    # coherent
    obs_XYfQU_pcal = name + '.XYfQU.pcal'
    os.system('rm -rf {}'.format(obs_XYfQU_pcal)) 
    gaincal(vis=obs, caltable=obs_XYfQU_pcal, 
              field=polcal_field,
              gaintype='XYf+QU',
              solint='inf',
              combine='scan,obs',
              preavg=300, # in seconds
              refant=refant,
              refantmode='strict',
              smodel=[1,0,1,0],
              gaintable=[obs_apint_pcal, obs_kcross_pcal],
              interp=['linear','nearest'])
    obs_XY0_pcal = name + '.XY0.pcal'
    # use the proximated solution to resolve the QU ambiguity with 180deg
    # and it also return a updated version of source model
    S = xyamb(xytab=obs_XYfQU_pcal, qu=qu[0], xyout=obs_XY0_pcal)
    
    applycal(vis=obs, 
         field=polcal_field,
         calwt=[True, True, False, False],
         gaintable=[obs_apint_pcal, obs_kcross_pcal, obs_XY0_pcal],
         interp=['nearest','linear', 'nearest', 'nearest'])
 

    # Revise the gain with good source pol estimate
    obs_apint_pcal2 = name + '.apint.pcal2'
    os.system('rm -rf {}'.format(obs_apint_pcal2)) 
    gaincal(vis=obs,
            caltable=obs_apint_pcal2, 
            field=polcal_field,
            solint='int',
            refant=refant,
            refantmode='strict',
            smodel=S,
            gaintable=[],interp=['nearest',],
            parang=True)
 

    # just to check
    qufromgain(obs_apint_pcal2)

    # solve for leakage terms
    obs_Dfinf_pcal = name + '.Dfinf.pcal'
    os.system('rm -rf {}'.format(obs_Dfinf_pcal)) 
    polcal(vis=obs,
           caltable=obs_Dfinf_pcal, 
           field=polcal_field,
           solint='inf',combine='obs,scan',
           preavg=300,
           poltype='Dflls',
           refant='', #solve absolute D-term
           smodel=S,
           gaintable=[obs_apint_pcal2, obs_kcross_pcal, obs_XY0_pcal],
           gainfield=['', '', '', ''],
           interp=['nearest','linear','nearest','nearest'])
    # write the function, plot_utils.check_polcal
    plot_utils.check_Dterm(obs_Dfinf_pcal)


    # Modify D-term solutions to be applied to the parallel-hands and cross-hands
    obs_Dfgen_pcal = name + '.Dfgen.pcal'
    Dgen(dtab=obs_Dfinf_pcal, dout=obs_Dfgen_pcal)
 
    # apply all the calibration tables
    obs_kcross_pcal = name + '.kcross.pcal'
    obs_XY0_pcal = name + '.XY0.pcal'
    obs_Dfgen_pcal = name + '.Dfgen.pcal'

    applycal(vis=polcalvis, 
             field=','.join([gaincal_field, target_field]), 
             calwt=[False,False,False],
             gaintable = [obs_kcross_pcal,obs_XY0_pcal,obs_Dfgen_pcal],
             interp=['nearest','nearest','nearest'],  
             gainfield=['', '', ''],
             parang=True)

    # apply the calibration to the polarised calibrator only
    applycal(vis=polcalvis,
      field = polcal_field,
      gaintable = ['%s.Gpol2'%msname, obs_kcross_pcal, obs_XY0_pcal, obs_Dfgen_pcal],
      gainfield = [polcal_field,polcal_field, polcal_field, polcal_field],
      interp = ['linear','nearest','nearest','nearest'],
      calwt = False, parang = True, flagbackup = False)
    check_polcal(polcalvis)


    # apply the polarisation calibration to the science targets
    applycal(vis=vis,
      field = science_field,
      gaintable = ['%s.Gpol2'%msname, obs_kcross_pcal, obs_XY0_pcal, obs_Dfgen_pcal],
      gainfield = [polcal_field,polcal_field, polcal_field, polcal_field],
      interp = ['linear','nearest','nearest','nearest'],
      calwt = False, parang = True, flagbackup = False)

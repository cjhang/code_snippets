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
fluxdensity = [2.25,0,0,0] # check the flux density of fluxcal
reffreq = '337.5GHz' # the reference frequency
spw_bpphase = '*:20~45'     # the spw window to get the time variation for bandpass calibrator
mysolint = 'int'    # integration time

os.system('rm {}.listobs.txt'.format(vis))
listobs(vis=vis, listfile=vis+'.listobs.txt', verbose=True)
try:
    # only plot the XX, YY correlation
    plot_utils.check_info(vis, refant=refant, correlation='XX,YY', avgchannel='64', datacolumn='data')
except:
    continue
# initial flagging
if False:
    # bulk flagging
    # flag edge channels
    flagdata(vis=vis, spw='*:0~3,*:60~63')
    flagmanager(vis=vis, mode='save', versionname='Beforebandpass_flag')
    # specific flagging
    flagdata(vis='', mode='manual', antenna='', timerange='', flagbackup=False)


#######################################################
#             Bandpass and Gain Calibration
#######################################################

vis_pint_bcal = '{}.pint.bcal'.format(name)
vis_apinf_bcal = '{}.apinf.bcal'.format(name)

# bandpass calibration
os.system('rm -rf {}'.format(vis_pint_bcal))
gaincal(vis=vis,
        caltable=vis_pint_bcal, 
        field=bandpass_field,
        gaintype='G', solint='int', calmode='p',
        spw=spw_bpphase, # channels with flat amp
        refantmode='strict',
        refant=refant,
        smodel=[1,0,0,0], # assuming unpolarized model, [I,Q,U,V]
        )
os.system('rm -rf {}'.format(vis_apinf_bcal))
bandpass(vis=vis, 
         caltable=vis_apinf_bcal,
         field=bandpass_field,
         solint='inf', combine='scan,obs', # vis should be there for combined observations
         refant=refant, solnorm=True, minsnr=2,
         gaintable=[vis_pint_bcal,],interp=['nearest'])

try:
    plot_utils.check_bandpass(phaseint_cal=vis_pint_bcal, 
                          bandpass_cal=vis_apinf_bcal)
except:
    continue


# setJy, find the closest value from ALMA calibrator database
setjy(vis='./data_cal/NGC253_April28th.ms',
      field=fluxcal_field,
      spw='',
      standard='manual',
      fluxdensity = fluxdensity,
      reffreq = reffreq,)

# new caltables
vis_pint_gcal = '{}.pint.gcal'.format(name)
vis_aint_gcal = '{}.aint.gcal'.format(name)
vis_flux_fcal = '{}.flux.fcal'.format(name)

os.system('rm -rf {}'.format(vis_pint_gcal))
gaincal(vis=vis,
        caltable=vis_pint_gcal, 
        gaintype='G',
        field=','.join([bandpass_field, gaincal_field]),
        solint='int',refant=refant,
        refantmode='strict',
        calmode = 'p', 
        gaintable=[vis_apinf_bcal],
        interp=['nearest'])

os.system('rm -rf {}'.format(vis_aint_gcal))
gaincal(vis=vis,
        caltable=vis_aint_gcal, 
        field=','.join([bandpass_field, gaincal_field]),
        solint='int',refant=refant,
        refantmode='strict',
        gaintype = 'T',
        calmode = 'a', 
        gaintable=[vis_apinf_bcal, vis_pint_gcal],
        interp=['nearest', 'nearest'])

os.system('rm -rf {}'.format(vis_flux_fcal))
fluxscale(vis=vis, 
          caltable=vis_aint_gcal,
          fluxtable = vis_flux_fcal,
          reference=fluxcal_field,
          transfer=','.join([bandpass_field,])) 

# applying the gain calibration temperary and checking the validity
applycal(vis=vis, 
         field=','.join([bandpass_field, gaincal_field]),
         calwt=True,
         gaintable=[vis_apinf_bcal,vis_pint_gcal,vis_flux_fcal],
         interp=['nearest','linear','linear'], parang=False)
try:
    plot_utils.check_cal(vis=vis, correlation='XX,YY', field=','.join([bandpass_field, gaincal_field]))
except:
    print("Load plot_utils for plots!")
    continue



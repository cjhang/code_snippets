# Simple manual prior calibration for ALMA data
# This script works for prior calibration with local ms files
#
# The input files for this script is the every associated executive block, where in the end they will
# be merged into one visibility that only contains the scientific spw
#
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
#                  Data Importing
#######################################################
# rawdata_dir = '../raw'
# data_dir = './data'
# #> Prepare the ms data
# #>> import asdm into local data folder
# asdm_list = glob.glob('../raw/*.asdm.sdm')
# for asdm in asdm_list:
    # basename = os.path.basename(asdm)[:-9]
    # msfile = basename + '.ms'
    # vis_fullpath = os.path.join(data_dir, msfile)
    # if not os.path.isdir(vis_fullpath):
        # importasdm(asdm=asdm, vis=vis_fullpath)
    # else:
        # print('Using existing: {}'.format(vis_fullpath))
# #>> generate info file for every observation
# obs_list = glob.glob(os.path.join(data_dir, '*.ms'))
# for obs in obs_list:
    # os.system('rm {}.listobs.txt'.format(obs))
    # listobs(vis=obs, listfile=obs+'.listobs.txt', verbose=True)

#>>>>>>>>>>>>
# #> Or, with casa_utils 
# #>> import_rawdata(rawdir='../raw', outdir=data_dir) 

#######################################################
#                  Data Preparation
#######################################################
#> define the basic information
polcal_field = 'J0006-0623'
bandpass_field = 'J2258-2758'
fluxcal_field = 'J2258-2758'
gaincal_field = 'J0038-2459'
target_field = 'NGC_253'
fields_unique = [bandpass_field, gaincal_field, target_field] #,polcal_field,] 
fields_tied_wvr = [target_field, gaincal_field] # the first one has wvr info, check CALIBRATE_ATMOSPHERE

myrefant = 'DV23'
science_spw = '17,19,21,23'
tsys_spw = '9,11,13,15'
mysolint = 'int'    # integration time
#> mapping tsys spw to science spw, check CALIBRATE_ATMOSPHERE
# example:
# >> tsysmap = list(range(0,24)) # 24 is total number of spw
# >> tsysmap[17:25] = list(range(9,17)) # map science spw to tsys spw

# deal with old ALMA data
is_fixsyscaltimes = False # Fix the ASDM SYSCal table issue, data before 2015
toffset = 0  # wvr time offset, -1 for cycle 0

data_dir = './data'
cal_dir = './prior_cal'  # the directory to store the calibration tables
caldata_dir = './calibrated'

try:
    import plot_utils
except:
    print("Warning: no plotter is found! Will not generate summary plots for calibration.")


#> Flagging the useless data
obs_list = glob.glob(os.path.join(data_dir, '*.ms'))
for obs in obs_list:
    # flagdata(vis=msfile, mode='unflag', flagbackup=False)
    # flagmanager(vis=msfile, mode='list')
    flagdata(vis=obs, mode='manual', autocorr=True, flagbackup=False)
    flagdata(vis=obs, mode='manual', intent='*POINTING*', flagbackup=False)
    flagdata(vis=obs, mode='shadow', flagbackup=False)
    #>> Flag edge channel
    #> remove the first 5s and the end 5s data (optional, see the amp_vs_time)
    # flagdata(vis=msfile, mode='quack', quackinterval=3.0, quackmode='beg', flagbackup=False)
    # flagdata(vis=msfile, mode='quack', quackinterval=3.0, quackmode='endb', flagbackup=False)
    # flagdata(vis = msfile, spw='17:0~3,19:0~3,21:0~3,23:0~3')
    flagmanager(vis=obs, mode='save', versionname='Prior_flag')

    if is_fixsyscaltimes:
        #>> Fix the ASDM SYSCal table issue, data before 2015
        from recipes.almahelpers import fixsyscaltimes
        fixsyscaltimes(vis=msfile)
    
    try:
        # to get general info about the quality of the data (only calibrators)
        plot_utils.check_info(vis=msfile, spw=tsys_spw, refant=myrefant,
                              show_fields=True, fields=calibrators, plotdir='plots/rawdata_info')
        # check the staibility of baselines with freq and time, useful to setup the spw4bpphase and solint 
        plot_utils.check_cal(vis=msfile, refant=myrefant, ydatacolumn='data', 
                             field=calibrators, spw=myspw, plotdir='plots/rawdata_refant')
    except:
        print("Load plot_utils to generate the plots!")

#######################################################
#                  Prior Calibration
#######################################################
print("\n==============> Start Prior Calibration <=============\n")

#> Tsys Calibration
#>> It gives the first-order correction for atmospheric opacity as a funcction of time and freq
for obs in obs_list:
    basename = os.path.basename(obs)
    tsys_cal = f'{os.path.join(cal_dir,basename)}.tsys'
    os.system(f'rm -rf {tsys_cal}')
    gencal(vis=obs, caltable=tsys_cal, caltype='tsys')
    # check tsys   
    try:
        plot_utils.check_tsys(tsystable=tsys_cal)
    except:
        print("Load plot_utils to generate the plots for inspection!")


    #> save the output of WVR
    mylogfile = casalog.logfile()
    wvrcal_log = f'{os.path.join(cal_dir,basename)}.wvrgcal.log'
    os.system(f'rm -rf {wvrcal_log}')
    casalog.setlogfile(f'{wvrcal_log}')
    wvr_cal = f'{os.path.join(cal_dir,basename)}.wvr'
    rmtables(wvr_cal)
    wvrgcal(vis = obs, 
            caltable = wvr_cal,
            toffset = toffset, # ALMA cycle 0, up to 2013, was -1
            #> tie phasecal and science target, share CALIBRATE_ATMOSPHERE 
            tie = [','.join(fields_tied_wvr),], 
            # statsource = 'fieldname', # usually the bandpass
            wvrflag=[], # add the antenna with bad WVR
            )
    casalog.setlogfile(mylogfile)

if True:
    #>> specific flagging
    # flagdata(vis='uid___A002_Xdb7ab7_X11a58.ms.tsys', scan='15', antenna='DV02', flagbackup=False)
    for obs in obs_list:
        flagmanager(vis=obs, mode='save', versionname='Manual_flag')

#> Apply the Tsys and WVR calibration
#>> we could use the helper function to get the mapping of system table, but be careful 
## from recipes.almahelpers import tsysspwmap #casa5
## from casarecipes.almahelpers import tsysspwmap #casa6
## tsysmap = tsysspwmap(vis=msfile, tsystable=msfile+'.tsys')

for obs in obs_list:
    #> General calibrators that have their own Tsys
    basename = os.path.basename(obs)
    tsys_cal = f'{os.path.join(cal_dir,basename)}.tsys'
    wvr_cal = f'{os.path.join(cal_dir,basename)}.wvr'
    for field_name in list(set(fields_unique) - set(fields_tied_wvr[-1])):
        applycal(vis = obs,
                 field = field_name,
                 spw = science_spw,
                 gaintable = [tsys_cal, wvr_cal],
                 gainfield = [field_name, field_name],
                 interp = 'linear',
                 spwmap = [tsysmap,[]],
                 calwt = True,
                 flagbackup = False)

    # for the tied field, without their own wvr
    applycal(vis = obs,
             field = fields_tied_wvr[-1],
             spw = science_spw,
             gaintable = [tsys_cal, wvr_cal],
             gainfield = fields_tied_wvr,
             interp = 'linear',
             #spwmap = [tsysmap,[]], no need for spwmap
             calwt = True,
             flagbackup = False)

    #> Check the calibrated data
    try:
        plot_utils.check_cal(vis=obs, refant=myrefant, ydatacolumn='corrected', 
                             field=calibrators, spw=myspw, plotdir='plots/aprior_refant')
    except:
        print("Load plot_utils to generate the plots!")

os.system('mkdir -p {}'.format(caldata_dir))
for obs in obs_list:
    outvis = os.path.join(caldata_dir, os.path.basename(obs))
    split(obs, outputvis=outvis, spw=science_spw, datacolumn='corrected')

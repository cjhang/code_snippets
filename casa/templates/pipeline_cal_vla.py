# Designed for pipeline calibration

# History
# 2021-02-10: Copying from the NRAO officitial instruction

mySDM = ['']

__rethrow_casa_exceptions = True
context = h_init()
context.set_state('ProjectSummary', 'observatory', 'Karl G. Jansky Very Large Array')
context.set_state('ProjectSummary', 'telescope', 'EVLA')
try:
    hifv_importdata(vis=mySDM, createmms='automatic',\
                    asis='Receiver CalAtmosphere', ocorr_mode='co',\
                    nocopy=False, overwrite=False)
    hifv_hanning(pipelinemode="automatic")
    hifv_flagdata(tbuff=0.0, flagbackup=False, scan=True, fracspw=0.05,\
                  intents='*POINTING*,*FOCUS*,*ATMOSPHERE*,*SIDEBAND_RATIO*,\
                  *UNKNOWN*, *SYSTEM_CONFIGURATION*, *UNSPECIFIED#UNSPECIFIED*',\
                  clip=True, baseband=True, shadow=True, quack=True, edgespw=True,\
                  autocorr=True, hm_tbuff='1.5int', template=True, online=True)
    hifv_vlasetjy(fluxdensity=-1, scalebychan=True, spix=0, reffreq='1GHz')
    hifv_priorcals(tecmaps=False)
    hifv_testBPdcals(weakbp=False)
    hifv_checkflag(pipelinemode="automatic")
    hifv_semiFinalBPdcals(weakbp=False)
    hifv_checkflag(checkflagmode='semi')
    hifv_semiFinalBPdcals(weakbp=False)
    hifv_solint(pipelinemode="automatic")
    hifv_fluxboot2(fitorder=-1)
    hifv_finalcals(weakbp=False)
    hifv_applycals(flagdetailedsum=True, gainmap=False, flagbackup=True,\
                   flagsum=True)
    hifv_targetflag(intents='*CALIBRATE*,*TARGET*')
    hifv_statwt(datacolumn='corrected')
    hifv_plotsummary(pipelinemode="automatic")
    hif_makeimlist(nchan=-1, calcsb=False, intent='PHASE,BANDPASS', robust=-999.0,\
                   parallel='automatic', per_eb=False, calmaxpix=300,\
                   specmode='cont', clearlist=True)
    hif_makeimages(tlimit=2.0, hm_perchanweightdensity=False, hm_npixels=0,\
                   hm_dogrowprune=True, hm_negativethreshold=-999.0, calcsb=False,\
                   hm_noisethreshold=-999.0, hm_fastnoise=True, hm_masking='none',\
                   hm_minpercentchange=-999.0, parallel='automatic', masklimit=4,\
                   hm_nsigma=0.0, target_list={}, hm_minbeamfrac=-999.0,\
                   hm_lownoisethreshold=-999.0, hm_growiterations=-999,\
                   overwrite_on_export=True, cleancontranges=False,\
                   hm_sidelobethreshold=-999.0)
    #hifv_exportdata(gainmap=False, exportmses=False, exportcalprods=False)
finally:
    h_save()

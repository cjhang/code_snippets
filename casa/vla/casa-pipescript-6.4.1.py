# This CASA pipescript is meant for use with CASA 6.4.1 and pipeline 2022.2.0.64
context = h_init()
context.set_state('ProjectSummary', 'observatory', 'Karl G. Jansky Very Large Array')
context.set_state('ProjectSummary', 'telescope', 'EVLA')
try:
    hifv_importdata(vis=['mySDM'], createmms='automatic',\
                    asis='Receiver CalAtmosphere', ocorr_mode='co',\
                    nocopy=False, overwrite=False)
    hifv_hanning(pipelinemode="automatic")
    hifv_flagdata(hm_tbuff='1.5int', fracspw=0.01, intents='*POINTING*,*FOCUS*,\
                  *ATMOSPHERE*,*SIDEBAND_RATIO*, *UNKNOWN*, *SYSTEM_CONFIGURATION*,\
                  *UNSPECIFIED#UNSPECIFIED*')
    hifv_vlasetjy(pipelinemode="automatic")
    hifv_priorcals(pipelinemode="automatic")
    hifv_syspower(pipelinemode="automatic")
    hifv_testBPdcals(pipelinemode="automatic")
    hifv_checkflag(checkflagmode='bpd-vla')
    hifv_semiFinalBPdcals(pipelinemode="automatic")
    hifv_checkflag(checkflagmode='allcals-vla')
    hifv_solint(pipelinemode="automatic")
    hifv_fluxboot(pipelinemode="automatic")
    hifv_finalcals(pipelinemode="automatic")
    hifv_applycals(pipelinemode="automatic")
    hifv_checkflag(checkflagmode='target-vla')
    hifv_statwt(datacolumn='corrected')
    hifv_plotsummary(pipelinemode="automatic")
    hif_makeimlist(intent='PHASE,BANDPASS', specmode='cont')
    hif_makeimages(hm_masking='centralregion')
    #hifv_exportdata(pipelinemode="automatic")
finally:
    h_save()



context = h_init()
context.set_state('ProjectStructure', 'recipe_name', 'hifa_calimage')

# SDM and Session should has same length
mySDM = ['@1']

if False: # datafixing
    from casarecipes.almahelpers import fixsyscaltimes # SACM/JAO - Fixes
    from casatasks import fixplanets
    try:
        hifa_importdata(vis=['@1'], session=['session_1'], dbservice=True, )
        fixsyscaltimes(vis='@1.ms') # SACM/JAO - Fixes
        fixsyscaltimes(vis='@1.ms') # SACM/JAO - Fixes
        h_save() # SACM/JAO - Finish weblog after fixes
        h_init() # SACM/JAO - Restart weblog after fixes
    except:
        pass

try:
    hifa_importdata(vis=['@1'], session=['session_1'], 
                    dbservice=True) # use flux.csv
    hifa_flagdata() # uses *flagtemplate.txt
    hifa_fluxcalflag()
    hif_rawflagchans()
    hif_refant()
    h_tsyscal()
    hifa_tsysflag()
    # hifa_tsysflagcontamination()
    hifa_antpos() # use antennapos.csv
    hifa_wvrgcalflag()
    hif_lowgainflag()
    hif_setmodels()
    hifa_bandpassflag()
    hifa_bandpass()
    hifa_spwphaseup()
    hifa_gfluxscaleflag(minsnr=2.0)
    hifa_gfluxscale()
    hifa_timegaincal(calminsnr=2.0,)
    # hifa_renorm(createcaltable=True, atm_auto_exclude=True)
    hifa_targetflag()
    hif_applycal()
finally:
    h_save()


"""syntax of the helper text files

>> flux.csv 
ms,field,spw,I,Q,U,V,spix,uvmin,uvmax,comment
uid___@1.ms,0,17,0.8,0.0,0.0,0.0,-0.595,0.0,0.0,"# field=J0334-4008 intents=AMPLITUDE,ATMOSPHERE,BANDPASS,POINTING,WVR origin=DB age=-1.0 queried_at=2022-01-12 19:23:17 UTC"

>> antennapos.csv
name,antenna,xoff,yoff,zoff,comment
uid___@1.ms,DA41,-5.29597e-06,-1.16080e-05,-1.60051e-04,

>> uid___@1.flagtemplate.txt /no spaces in the reason string!
mode='manual' antenna='DV02;DV03&DA51' spw='22,24:150~175' reason='QA2:applycal_amplitude_frequenc
"""

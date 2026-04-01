    ...
    hifv_finalcals()

    # Reference Frequency for fit values
    reffreq = '33.0GHz'
    # Stokes I flux density
    I =         1.4953 
    # Spectral Index
    alpha =    [-0.7512, 0.1885]
    # Polarization Fraction (fractional polarizion)
    polfrac =[ 0.0412126,   0.02973067, -0.00598331]
    # Polarization Angle (Radians)
    polangle = [ 1.48599775,  0.37284829, -0.99768313]
    setjy(vis='mySDM.ms', field='0542+498=3C147', spw='2~65',
      standard="manual", fluxdensity=[I,0,0,0], spix=alpha, reffreq=reffreq,
      polindex=polfrac, polangle=polangle, rotmeas=0, usescratch=True)

    hifv_circfeedpolcal(run_setjy=False)
    hifv_applycals()
    ...

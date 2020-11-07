

def ppxf_fit(wavelength, flux, flux_err, z, quiet=False, **kwargs):
    """A wrapper for pPXF


    """
    tie_balmer=False
    limit_doublets=False
    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    #wavelength = wavelength
    #z = z

    # Only use the wavelength range in common between galaxy and stellar library.
    #
    mask = (wavelength > 3585*(1+z)) & (wavelength < 7340*(1+z))
    wave = wavelength[mask]
    muse_mask = (wavelength < 5970) & (wavelength > 5800)

    #flux = flux[mask]
    gal_lin = flux[mask]
    galaxy, logLam1, velscale = util.log_rebin([wave[0], wave[-1]], gal_lin)
    flux_scale = np.median(galaxy)
    galaxy = galaxy/np.median(galaxy)   # Normalize spectrum to avoid numerical issues

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0).
    # A constant noise is not a bad approximation in the fitted wavelength
    # range and reduces the noise in the fit.
    #
    #noise = 0.024#err[mask]  
    noise = np.full_like(galaxy, 0.0047) 
    #noise[muse_mask[mask]] = 1000
    
    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458  # speed of light in km/s
    #velscale = c*np.log(wave[1]/wave[0])  # eq.(8) of Cappellari (2017)
    FWHM_gal = 2.7  # instrumental resolution FWHM

    #------------------- Setup templates -----------------------

    pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

    # The templates are normalized to mean=1 within the FWHM of the V-band.
    # In this way the weights and mean values are light-weighted quantities
    miles = lib.miles(pathname, velscale, FWHM_gal)
    lam_miles = np.e**miles.log_lam_temp
    lamRangeTemp = (lam_miles[0], lam_miles[-1])

    # The stellar templates are reshaped below into a 2-dim array with each
    # spectrum as a column, however we save the original array dimensions,
    # which are needed to specify the regularization dimensions
    #
    reg_dim = miles.templates.shape[1:]
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

    # See the pPXF documentation for the keyword REGUL,
    regul_err = 0.013  # Desired regularization error

    # Estimate the wavelength fitted range in the rest frame.
    lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z)

    # Construct a set of Gaussian emission line templates.
    # The `emission_lines` function defines the most common lines, but additional
    # lines can be included by editing the function in the file ppxf_util.py.
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp, lam_range_gal, FWHM_gal,
        tie_balmer=tie_balmer, limit_doublets=limit_doublets)

    # Combines the stellar and gaseous templates into a single array.
    # During the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    #
    templates = np.column_stack([stars_templates, gas_templates])

    #-----------------------------------------------------------

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_EXAMPLE_KINEMATICS_SAURON and Sec.2.4 of Cappellari (2017)
    #
    c = 299792.458
    dv = c*(miles.log_lam_temp[0] - np.log(wave[0]))  # eq.(8) of Cappellari (2017)
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 180.]  # (km/s), starting guess for [V, sigma]
    if not quiet:
        print('start=', start)

    n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    # Assign component=0 to the stellar templates, component=1 to the Balmer
    # gas emission lines templates and component=2 to the forbidden lines.
    component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
    gas_component = np.array(component) > 0  # gas_component=True for gas templates

    # Fit (V, sig, h3, h4) moments=4 for the stars
    # and (V, sig) moments=2 for the two gas kinematic components
    moments = [4, 2, 2]

    # Adopt the same starting value for the stars and the two gas components
    start = [start, start, start]

    # If the Balmer lines are tied one should allow for gas reddeining.
    # The gas_reddening can be different from the stellar one, if both are fitted.
    gas_reddening = 0 if tie_balmer else None
    
    goodPixels = determine_goodpixels(wave, lamRangeTemp, z, flag_emlines=False)
    
    # Here the actual fit starts.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    t = clock()
    pp = ppxf(templates, galaxy, noise, velscale, start,
              moments=moments, degree=-1, mdegree=10, vsyst=dv,
              lam=wave, clean=False, regul=1./regul_err, reg_dim=reg_dim,
              component=component, gas_component=gas_component,
              gas_names=gas_names, gas_reddening=gas_reddening,
              mask=(galaxy>0.01), quiet=quiet, **kwargs)

    pp.flux_scale = flux_scale
    # When the two Delta Chi^2 below are the same, the solution
    # is the smoothest consistent with the observed spectrum.
    #
    if not quiet:
        print('Desired Delta Chi^2: %.4g' % np.sqrt(2*galaxy.size))
        print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1)*galaxy.size))
        print('Elapsed time in PPXF: %.2f s' % (clock() - t))

        weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
        weights = weights.reshape(reg_dim)/weights.sum()  # Normalized

        miles.mean_age_metal(weights)
        miles.mass_to_light(weights, band="r")

# ERIS Utility

`eris_jhchen_utils.py` is designed to automatically download and reduce ERIS data, yet with a compact file.

A complete help page for the cmd interface can be found by:

    python eris_jhchen_utils.py -h

You can make it executable by: `chmod +x eris_jhchen_utils.py` and put it into your system "PATH", then you can directly call it with `eris_jhchen_utils.py`.

You can also use it as a normal Python package:

    from eris_jhchen_utils import request_calib

Due to the limited user base of this script, there is currently no stand-alone documentation for all the functions. But you are welcome to dig into and modify the source code. The function and variable names should be reasonably easy to guess their meaning. The source code has four main sections for data reduction: 'DATA Retrieval', 'DATA Calibration', 'Flux Calibration', and 'DATA Combination'. Besides, "DATA Quicklook", "Quick Tools", and "Helper Functions" include some small functions/gadgets to make it easier to interact with the data.

## Requirement

`eris_jhchen_utils.py` is a purge Python wrapper of the `esorex` recipes, so you should have proper `esorex` installed. After installing `esorex`, make sure esorex is in your $PATH, so that `eris_jhchen_utils.py` can find it.

As a Python package, `eris_jhchen_utils.py` also requires several python packages; you can install them with:

    pip install numpy scipy astropy astroquery

Additional to the command line tool described here, you can also use the package as a normal python package. You will need to copy it into your "PYTHONPATH" and then:

    from eris_jhchen_utils import request_science, get_daily_calib, reduce_eris # and many others

The current version is fully tested with Python 3.10.


## Reduce the data for a whole project (automatic pipeline)

To download and reduce the data of a whole project (here we use galphys as an example, but it is similar to other projects as well) in an elegant way (just two steps):

    # galphys pipeline, to reduce the most recent data
    eris_jhchen_utils.py request_science --prog_id "112.25M3" --username <your_eso_user_name> --outdir science_raw --archive
    eris_jhchen_utils.py run_eris_pipeline -d science_raw --outdir science_reduced --calib_pool calibPool --calib_raw calib_raw

## Reducing data step-by-step

### Step 1: downloading raw data

To download the ERIS science data, take the galphy project "111.24YQ.001" as an example:

    eris_jhchen_utils.py request_science --prog_id "111.24YQ.001" --username <your_eso_user_name> --outdir science_raw --archive

you need to replace the `<your_eso_user_name>` to your eso user name. The `--archive` option tells the program to organise the downloaded raw data into different dates, so you will find you data been kept somewhere like "science_raw/2023-04-10/". If you remove the option `--archive`. The compressed fits files can be directly found in "science_raw".

You need to make sure all your required data, such as the PSF stars, are downloaded. It is usually fine, if you specify only the project code. If you only download the specified OB with `--ob_id`, then you need to also manually download other calibration raw files, such as these for telluric stars.

Now you can directly jump to the "Step 2", as the program can automatically download the calibration files for you. However, if you still want to download the calibration raw files manually, you can achive it through "request_calib". Check the details with `eris_jhchen_utils.py request_calib -h`.

### Step 2: reducing the science data

After downloading all the raw data, here we assume it is in "science_raw/2023-04-10". Then, you can simply reduce the download data:

    eris_jhchen_utils.py reduce_eris --datadir science_raw/2023-04-10 --outdir science_reduced 

The program will automatically search and download the calibration files for you. You can also specify the `--calib_pool` to speed-up the process, like:
    
    eris_jhchen_utils.py reduce_eris --datadir science_raw/2023-04-10 --outdir science_reduced_klow_25mas_reduced --calib_pool klow_25mas_calib

in this way, it is the user's responsibility to make sure the "calib_pool" contains the right calibration files for your science data. For instance, here we know the data are observed with "klow" and spaxel scale of "25mas". If the same folder includes data with different setups, it is also the user's responsibility to separate them into different folders. 

In general, it is recommended to run the command without specify the `--calib_pool`, the program will automatically download the correct calibration raw files and reduce them separately according to their configurations and dates, but this can double the required time to reduce the all the data as it needs also to reduce the calibration files first.

Check "Preparing for the observing runs" if you want to reduce your data as soon as possible when you are on the visitor mode. 

## Quick tools

"Quick tools" are designed to simplify the pipeline steps, and they can generate and organise the files into folder structures.

### "get_daily_calib"

For instance, if you are keen to reduce the date of a specific day, you can get the daily calibration files quickly by:

    eris_jhchen_utils.py get_daily_calib -d 2023-04-09 -b K_low -s 100mas -e 600

If you want to prepare the daily calibration files for an on-going observation, which can be later re-used by the `run_eris_pipeline`:

    obsdate="2024-08-25"
    eris_jhchen_utils get_daily_calib -d "$obsdate" -b K_low -s 100mas -e 600 --outdir calibPool/"$obsdate" --calib_raw calib_raw/"$obsdate"

If you have downloaded all the science data of your project, and the data are organised in the folder with their observing dates. Then, all the data can be reduced quickly by the `run_eris_pipeline`. Assuming all the raw data are in "science_raw":

    eris_jhchen_utils.py run_eris_pipeline -d science_raw 

### 'quick_pv_diagram'

If you want to check the PV diagram of your reduced cubes:
    
    eris_jhchen_utils.py quick_pv_diagram -d science_reduced/eris_ifu_jitter_twk_cube_000.fits -z 2.0

### 'quick_combine'

To combine the reduced data (taking 250mas observation of zc406690 as an example):

    eris_jhchen_utils.py quick_combine --datadir science_reduced --target zc406690 --band K_short --spaxel 250mas --outdir combined --suffix v1 

For detailed input parameters of other quick tools:

   eris_jhchen_utils.py -h
   eris_jhchen_utils.py <quick_tool_name> -h

## Drifts

A crucial step for 25mas observations is the correction for drifts. This step involves with some human intervention to derive the offset from the PSF stars. The following python code can be used to derive the drifts (taking d3a6004 clump-A as an example):

```python
    from eris_jhchen_utils import search_archive, summarise_eso_files, construct_drift_file, fit_eris_psf_star, interpolate_drifts, quick_fix_nan

    target = 'd3a6004'
    file_list,_,_ = search_archive('./science_reduced', target=target, band='K_middle', spaxel='25mas', target_type='SCIENCE')
    target_summary = summarise_eso_files(file_list)
    construct_drift_file(target_summary['tpl_start'], datadir='science_reduced', driftfile=target+'_clumpA_drifts.csv')
    fit_eris_psf_star(target+'_clumpA_drifts.csv', plotfile=target+'_clumpA_PSF.pdf', interactive=1)
    interpolate_drifts(target+'_clumpA_drifts.csv')
```

Then, the final "*_drift.csv" can be read by the `quick_combine` to properly account the drifts of the telescope (taking d3a6004 as an example).

```python
    from eris_jhchen_utils import quick_combine
    quick_combine(datadir='science_reduced', target='d3a6004', sigma_clip=5, weak_signal=True, band='K_middle', spaxel='25mas', outdir='combined', suffix='try1_withdrifts', drifts='d3a6004_drifts.csv', z=2.3871)
```

Or, if you prefer command line tool:

    eris_jhchen_utils.py quick_combine --datadir science_reduced --target d3a6004 --band K_middle --spaxel 25mas --outdir combined --suffix try1_withdrifts --z 2.3871 --drifts d3a6004_drifts.csv --weak_signal --sigma_clip 5

Here, we adopted the options `--weak_signal`, `--sigma_clip` to clipping the bad channels and pixels. The redshift is also important to preserve the possible spectral lines.

## Flux calibration

Flux calibration normally needs more human intervention if the star is offset from the field center or the star is not typical B or G type stars.

In normal situation, the flux calibration (zero point and transmission curve) can be quickly derived from the standard telluric stars using:

```python
    from eris_jhchen_utils import search_archive, get_telluric_calibration
    
    star_list, _, _ = search_archive('./science_reduced', band='K_short', spaxel='25mas', target_type='CALIBSTD')
    get_telluric_calibration(star_list, datadir='science_reduced', outdir='spectral_correction')
```

Then, you can get the correction in fits format in the folder of `spectral_correction`. For each fits file, there is also a qa file for quality assessment.

With the "*_qa.pdf", you can check the quality of the flux calibration. If more human interventions are needed. You can also break the task into individual steps:

```python
    from eris_jhchen_utils import extract_star, get_corrections
    
    # extract the spectrum from the standard star
    starfile = '' # e.g. science_reduced/<date>/<name>_CALIBSTD_<OB-ID>_<TPL-start>_<band>_<spaxel>_<exptime>'
    # then, you can specify all the paremters manually
    star_name = 'HD 22586'
    band_2mass = 'Ks' # possible are J, H, Ks
    star_info = query_star_VizieR(star_name)
    star_type = star_info['type'][0]
    T_star = guess_star_temperature(star_type) # or manually
    magnitude_2mass = star_info[band_2mass[0]+'mag'][0] # or manually
    # all the above information can be also automatically generated with
    # the function "read_starfile", check help(read_starfile)
    # read_starfile(starfile)
    wave, spec = extract_star(starfile, plot=True, sky_aperture=6)
    zero_point, transmission_normed, correction = get_corrections(
        wave, spec, T_star=T_star, band_2mass=band_2mass, 
        magnitude_2mass=magnitude_2mass)
```
Check for each function, epecially `extract_star` to fine-tune the extraction aperture and apply the background subtraction.

The "correction", other returned from the function `get_corrections` or the 'pdf' file in the directory of "spectral_correction", is the combined correction from zero_point and transmission. To correct the reduced science datacube, one only needs multiple the correction to the datacube:

    data_corrected = correction[:,None,None] * data

Or, to automatically aligned the spectral axis and apply the correction:

    from eris_jhchen_utils import correct_cube
    correct_cube(fitscube, correction, outdir='output')

## Preparing for the observing runs

During the observing run it is always important to get a quick reduced science observation to evaluate the performance. In this case, having pre-prepared calibration files will speed up the data reduction on site. If you already know what observing configurations you will have, you can prepare the calibration files using "get_daily_calib":

    eris_jhchen_utils.py get_daily_calib -d "2025-02-19" -b K_middle -s 100mas -e 600 --outdir calibPool/ 

If you expect to have various configurations, you can using a bash loop:

    obsdate="2025-02-19"
    declare -a band_list=("K_middle" "K_low")
    declare -a spaxel_list=("25mas" "100mas" "250mas")
    declare -a exptime_list=("10" "30" "600")

    for band in "${band_list[@]}"
    do
      for spaxl in "${spaxel_list[@]}"
      do
        for exptime in "${exptime_list[@]}"
        do
          echo "$obsdate" + "$band" + "$spaxl" + "$exptime"
          eris_jhchen_utils.py get_daily_calib -d "$obsdate" -b "$band" -s "$spaxl" -e "$exptime" --outdir calibPool --calib_raw calib_raw
        done
      done
    done


here we have defined '--calib_raw' to store the raw calibration files locally, but you can remove them when the script is finished.

On the mountains, you can dowload your new observation with:

    eris_jhchen_utils.py request_science --user <your_eso_user_name> --ob_id 4206913 --outdir science_raw/obs1 --metafile science_raw/obs1/metadata.csv

Or, you can download the same data directly from the internal server and put them into the folder 'science_raw/obs1'. Then, you can reduce your newly acquired data with:

    eris_jhchen_utils.py reduce_eris --datadir science_raw/obs1 --outdir science_reduced/obs1 --calib_pool calibPool/2025-02-19_K_middle_100mas_600s --categories SCIENCE

here we assume that your science data is stored in 'science_raw/obs1', and its instrument configuration is 'K_middle' + '100mas' + '600s'. We also specify the '--categories' to 'SCIENCE' only, so it will only reduce the science target. The date in the folder name is only used by 'get_daily_calib' to organize the data. For your quick reduction, the date is not so important.


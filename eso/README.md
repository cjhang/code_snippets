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

The current version is fully tested with Python 3.10.

## Examples

Download the raw calibration files for specific day(s)

    eris_jhchen_utils.py request_calib --start_date 2023-04-09 --band K_low --spaxel 100mas --exptime 600 --outdir ./raw --metafile raw/2023-04-09.metadata.csv

If you only want specific calibration raw files, just pass through the "steps":

    eris_jhchen_utils.py request_calib --steps dark detlin --start_date 2023-04-09 --band K_low --spaxel 100mas --exptime 600 --outdir ./raw --metafile raw/2023-04-09.metadata.csv

The supported steps are: 'dark', 'detlin', 'distortion', 'flat', 'wavecal'.

After downloading all the required files, you can also generate the corresponding calibration files:

    eris_jhchen_utils.py generate_calib --metadata raw/2023-04-09.metadata.csv --raw_pool raw --calib_pool calibPool

Again, if you only want part of the calibration files:

    eris_jhchen_utils.py generate_calib --metadata raw/2023-04-09.metadata.csv --raw_pool raw --calib_pool calibPool --steps dark detlin

To download the ERIS science data, take the galphy project "111.24YQ.001" as an example:

    eris_jhchen_utils.py request_science --prog_id "111.24YQ.001" --username <your_eso_user_name> --outdir science_raw --archive

the `--archive` option tells the program to organise the downloaded raw data into different dates.

Then, to reduce the download data:

    eris_jhchen_utils.py reduce_eris --datadir raw/2023-04-10 --outdir science_reduced 

The program will automatically search and download the calibration files for you. You can also specify the calib_pool, like:
    
    eris_jhchen_utils.py reduce_eris --datadir raw/klow_25mas --outdir klow_25mas_reduced--calib_pool klow_25mas_calib

in this way, it is the user's responsibility to make sure the calib_pool contains the right calibration files for your science data. It is thus recommended to keep the science data (also the corresponding calibration files) with different configurations separately.


## Quick tools

"Quick tools" are designed to simplify the pipeline steps, and they can generate and organise the files into folder structures.

For instance, if you are keen to reduce the date of a specific day, you can get the daily calibration files quickly by:

    eris_jhchen_utils.py get_daily_calib -d 2023-04-09 -b K_low -s 100mas -e 600

If you want to prepare the daily calibration files for an on-going observation, which can be later re-used by the `run_eris_pipeline`:

    obsdate="2024-08-25"
    eris_jhchen_utils get_daily_calib -d "$obsdate" -b K_low -s 100mas -e 600 --outdir calibPool/"$obsdate" --calib_raw calib_raw/"$obsdate"

If you have downloaded all the science data of your project, and the data are organised in the folder with their observing dates. Then, all the data can be reduced quickly by the `run_eris_pipeline`. Assuming all the raw data are in "science_raw":

    eris_jhchen_utils.py run_eris_pipeline -d science_raw 

If you want to check the PV diagram of your reduced cubes:
    
    eris_jhchen_utils.py quick_pv_diagram -d science_reduced/eris_ifu_jitter_twk_cube_000.fits -z 2.0

For detailed input parameters of the quick tools:

    eris_jhchen_utils.py <quick_tool_name> -h

## Reduce the data for a whole project

To download and reduce the data of a whole project (here we use galphys as an example, but it is similar to other projects as well) in an elegant way (just two steps):

    # galphys pipeline, to reduce the most recent data
    eris_jhchen_utils.py request_science --prog_id "112.25M3.002" --username <your_eso_user_name> --outdir science_raw --archive
    eris_jhchen_utils.py run_eris_pipeline -d science_raw --outdir science_reduced --calib_pool calibPool --calib_raw calib_raw

To combine the reduced data:

    eris_jhchen_utils.py quick_combine --datadir science_reduced --target zc406690 --band K_short --spaxel 25mas --outdir combined --suffix try1 #--drifts drifts_file 

A crucial step for 25mas observations is the correction for drifts. This step involves with some human intervention to derive the offset from the PSF stars. The following python code can be used to derive the drifts (taking d3a6004 as an example):

    from eris_jhchen_utils import search_archive, summarise_eso_files, construct_drift_file, fit_eris_psf_star, interpolate_drifts, quick_fix_nan

    target = 'd3a6004'
    file_list,_,_ = search_archive('./science_reduced', target=target, band='K_middle', spaxel='25mas', target_type='SCIENCE')
    target_summary = summarise_eso_files(file_list)
    construct_drift_file(target_summary['tpl_start'], datadir='science_reduced', driftfile=target+'_drift.csv')
    fit_eris_psf_star(target+'_drift.csv', plotfile=target+'_PSF.pdf', interactive=1)
    interpolate_drifts(target+'_drift.csv')

Then, the final "*_drift.csv" can be read by the `quick_combine` to properly account the drifts of the telescope (taking d3a6004 as an example).

    eris_jhchen_utils.py quick_combine --datadir science_reduced --target d3a6004 --band K_middle --spaxel 25mas --outdir combined --suffix try1_withdrifts --drifts d3a6004_drifts.csv

## Flux calibration

Coming later.

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

    eris_jhchen_utils.py request_science --user cjhang --ob_id 4206913 --outdir science_raw/obs1 --metafile science_raw/obs1/metadata.csv

Or, you can download the same data directly from the internal server and put them into the folder 'science_raw/obs1'. Then, you can reduce your newly acquired data with:

    eris_jhchen_utils.py reduce_eris --datadir science_raw/obs1 --outdir science_reduced/obs1 --calib_pool calibPool/2025-02-19_K_middle_100mas_600s --categories SCIENCE

here we assume that your science data is stored in 'science_raw/obs1', and its instrument configuration is 'K_middle' + '100mas' + '600s'. We also specify the '--categories' to 'SCIENCE' only, so it will only reduce the science target. The date in the folder name is only used by 'get_daily_calib' to organize the data. For your quick reduction, the date is not so important.


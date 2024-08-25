# ERIS Utility

`eris_jhchen_utils.py` is designed to automatically download and reduce ERIS data, yet with a compact file.

A complete help page for the cmd interface can be found by:

    python eris_jhchen_utils.py -h

You can also use it as a Python package:

    from eris_jhchen_utils import request_calib

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

    eris_jhchen_utils.py request_science --prog_id "111.24YQ.001" --username <your_eso_user_name> --archive

the `--archive` option tells the program to organise the downloaded raw data into different dates.


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

## Galphy project

To download the galphy project and reduce the data in an elegant way (just two steps):

    # galphy pipeline, to reduce the most recent data
    eris_jhchen_utils.py request_science --prog_id "112.25M3.002" --username <your_eso_user_name> --outdir science_raw --archive
    eris_jhchen_utils.py run_eris_pipeline -d science_raw --outdir science_reduced --calib_pool calibPool --calib_raw calib_raw

To combine the reduced data:

    eris_jhchen_utils.py quick_combine --datadir science_reduced --target zc406690 --band K_short --spaxel 25mas --outdir combined --suffix test1 #--drifts drifts_file 

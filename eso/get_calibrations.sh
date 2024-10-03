# load the python evironment
#source /home/jhchen/apps/miniconda3/bin/activate astro
# add the eris_jhchen_utils into the PATH
#export PATH="/home/jhchen/apps/bin:$PATH"

# quickly get the daily calibration files
# ---------------------------------------
# example:
# 
#   eris_jhchen_utils get_daily_calib -d 2023-04-09 -b K_low -s 100mas -e 600
# 
# options:
#   -h, --help            show this help message and exit
#   -d DATE, --date DATE  Observing date
#   -b BAND, --band BAND  Observation band
#   -s SPAXEL, --spaxel SPAXEL
#                         Pixel size
#   -e EXPTIME, --exptime EXPTIME
#                         Exposure time
#   --outdir OUTDIR       Output directory, default ./
#   --static_pool STATIC_POOL
#                         Static calibration pool directory
#   --steps STEPS [STEPS ...]
#                         Calibration steps to be proceeded
#   --calib_raw CALIB_RAW
#                         Directory for raw calibration files, default calib_raw
#   --max_days MAX_DAYS   The maxium days to search for calibration raw files
#   --overwrite           Overwrite the existing files
#   --rename              Rename output files with detailed configurations


obsdate="2024-08-29"
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
      eris_jhchen_utils get_daily_calib -d "$obsdate" -b "$band" -s "$spaxl" -e "$exptime" --outdir calibPool/"$obsdate" --calib_raw calib_raw/"$obsdate"
    done
  done
done


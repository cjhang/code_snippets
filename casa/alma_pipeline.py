#!/usr/bin/env python

# A collection of CASA routine related to ALMA pipeline
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# 
# Usage: due the requirement for all the casa internal tasks/tools this file should 
#        be execfile interactively
#       
#        execfile(path_to_this_file)
#        recover_pipeline(root_folder)
#
# History:
#   2022.8.7: first release

import os
import sys
import glob

# ALMA data retrieval


# global variables
# ALMA pipeline versions: https://almascience.nrao.edu/processing/science-pipeline
ALMA_PIPELINE_versions = {
    "6.2.1":['2021-05-10','2099-10-01','casa-6.2.1-7-pipeline-2021.2.0.128'],
    "5.6.1":['2019-10-01','2021-05-10','casa-pipeline-release-5.6.1-8.el7'],
    "5.4.0":['2018-10-01', '2019-10-01', 'casa-release-5.4.0-70.el7'],
    "5.1.1":['2017-10-01', '2018-10-01', 'casa-release-5.1.1-5.el7'],
    "4.7.2":['2016-10-25', '2017-10.01', 'casa-release-4.7.2-el7'],
    "4.5.3":['2016-02-10', '2016-10-25', 'casa-release-4.5.3-el6'],
    "4.3.1":['2015-08-07', '2016-02-10', 'casa-release-4.3.1-pipe-1-el6']
    }

def check_pipeline_version(mous_id=None, project_id=None, vis=None):
    """check the required casa pipeline version

    Args:
        mous_id: the member mous id of the observation 
        project_id: project id
        vis: the represent visibility or the raw file
    """
    pass

def auto_execute_file(member_dir, mous_id=None, casa_pipeline=None, overwrite=False):
    """automatic execute the "*.scriptForPI.py" in the given directory
    
    Args:
        basedir: the starting folder contains the untarred files
    """
    subfolders = os.listdir(member_dir)
    if "calibration" not in subfolders:
        raise ValueError("No calibration can be found!")
    if "script" not in subfolders:
        raise ValueError("No script can be found!")
    if "raw" not in subfolders:
        raise ValueError("No raw files can be found!")
    if "calibrated" in subfolders:
        if not overwrite:
            print("Found existing calibrations, set 'overwrite=True' to overwrite!")
            pass
        else:
            os.system('rm -rf {}'.format(os.path.join(member_dir, "calibrated")))
    print("memeber_dir", member_dir)
    os.chdir(os.path.join(member_dir, 'script'))
    scripts_PI = glob.glob('./*.scriptForPI.py')
    if len(scripts_PI) < 1:
        raise ValueError("No scripts for PI!")
    print('{} --pipeline -c *.scriptForPI.py'.format(casa_pipeline))
    os.system('{} --pipeline --nologger -c *.scriptForPI.py'.format(casa_pipeline))
    
    #for PI_script in scripts_PI:
    #    #exec(open(PI_script).read())
    #    execfile(PI_script)

def recover_pipeline(basedir, dry_run=False, overwrite=False, casadir=None, **kwargs):
    """recover the pipeline calibration for a whole project
    
    Args:
        basedir: the starting folder contains the untarred files
    """
    project_match = re.compile('(?P<year>\d+)\.\d+\.\d+\.\w')
    for item in os.listdir(basedir):
        if project_match.match(item):
            project_year = project_match(item).groupdict()['year']
            project_cycle
    abs_basedir = os.path.abspath(basedir)
    member_list = []
    for root, subfolders, files in os.walk(abs_basedir):
        for subfolder in subfolders:
            if subfolder == 'script':
                scripts_PI = glob.glob(os.path.join(root, subfolder, './*.scriptForPI.py'))
                if len(scripts_PI) > 0:
                    member_list.append(root)
    if dry_run:
        return member_list
    for member in member_list:
        auto_execute_file(member, overwrite=overwrite, **kwargs)




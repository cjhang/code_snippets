#!/bin/bash -i
# "casa.sh"
# A wrapper around casapy to enforce separate resource directories for each session.
#
# History:
#   Original: RHEL 8 version, D. Petry (ESO) 
#   2022-01-06: Change the default configuration folder as the working directory, Jianhang Chen
#   2022-06-08: Add general default configuration directory, Jianhang Chen
# 
#

echo '(casa.sh version 4.1 Jan 2020, JC)'
INVOCATION=$(basename $0)
echo $INVOCATION
DEFAULTCONFIG="/home/jchen/applications/casa/casa_configs/"
WRAPPERINSTALLDIR="/home/jchen/.local/bin"
CASAINSTALLDIR="/home/jchen/applications/casa"

USERSTRING=""
echo
echo "Please enter a  Session ID  containing _your_ user id, e.g. "`whoami`"1"
read -t 45 -a USERSTRING -p "(ID should be different for each of your concurrent CASA sessions): "
echo $USERSTRING
RCFILENAME=""
if  [ ! $USERSTRING"a" == "a" ]; then
    RCFILENAME=${PWD}"/."${USERSTRING}".mycasa"
else
    echo "No id entered."
    RCFILENAME=${PWD}"/.mycasa"
fi
START=0
if  [ ! $RCFILENAME"a" == "a" ]; then
    if [ -e $RCFILENAME ]; then
	if [ -d $RCFILENAME ]; then
	    echo "Using existing directory "${RCFILENAME}" as CASA resource directory ..."
	    START=1
	else
	    echo "ERROR: Cannot use existing file "${RCFILENAME}" as CASA resource directory."
        fi
    else
        echo "Will create new directory "${RCFILENAME}" as CASA resource directory ..."
        mkdir $RCFILENAME
        cp -r ${DEFAULTCONFIG}/* ${RCFILENAME}/
	START=1
    fi
fi
if [ $START = 1 ]; then
    if [ $INVOCATION == "casa-5.6.1" ]; then
        exec ${CASAINSTALLDIR}/casa-pipeline-release-5.6.1-8.el6/bin/casa --rcdir $RCFILENAME $@    
    elif [ $INVOCATION == "casa-5.6.2" ]; then
        exec ${CASAINSTALLDIR}/casa-pipeline-release-5.6.2-3.el6/bin/casa --rcdir $RCFILENAME $@    
    elif [ $INVOCATION == "casa-5.7.0" ]; then
        exec ${CASAINSTALLDIR}/casa-release-5.7.0-134.el7/bin/casa --rcdir $RCFILENAME $@    
    # Put the default casa5 (python2) here
    elif [ $INVOCATION == "casa5" ]; then
        exec ${CASAINSTALLDIR}/casa-release-5.7.0-134.el7/bin/casa --rcdir $RCFILENAME $@    
    # Put the ALMA and VLA pipelines here
    elif [ $INVOCATION == "casapipealma" ]; then
        exec ${CASAINSTALLDIR}/casa-pipeline-release-5.6.1-8.el7/bin/casa --pipeline --rcdir $RCFILENAME $@    
    elif [ $INVOCATION == "casapipevla" ]; then
        exec ${CASAINSTALLDIR}/casa-pipeline-release-5.6.2-3.el7/bin/casa --pipeline --rcdir $RCFILENAME $@    
    # This should always point to the latest version and should not start a pipeline if preset
    elif [ $INVOCATION == "casa" ]; then
        exec ${CASAINSTALLDIR}/casa-6.4.0-16/bin/casa --rcdir $RCFILENAME $@
    else
        echo "Not found "$INVOCATION
    fi
fi

#!/usr/bin/env python

# funtion to identify flagging data from the output in the logfile

# Author: Zhi-Yu Zhang, Jianhang Chen
# Email: pmozhang@gmail.com, cjhastro@gmail.com
# History:
#   21 Mar 2019, update by Zhiyu
#   30 Sep 2019, transfer into function from the original script, by Jianhang
#    3 Dec 2019, v0.0.2: match logs by regex, by Jianhang
#    1 Apr 2020, sort the ouput including more details


import re
from collections import Counter

version = '0.0.3'

def match_info(info_line, debug=False):
    """match the info of plotms selection
    """
    match_list = {}
    match_list['scan']  = 'Scan=(?P<scan>\d+)\s'
    match_list['field'] = 'Field=(?P<field>[\w\s+-]+)\s\[(?P<field_id>\d+)\]\s'
    match_list['time']  = 'Time=(?P<time>[\w/:.]+)\s'
    match_list['ant1']  = 'BL=(?P<ant1>\w+)@\w+\s&\s\w+@\w+\s\[[\d&]+\]\s'
    match_list['ant2']  = 'BL=\w+@\w+\s&\s(?P<ant2>\w+)@\w+\s\[[\d&]+\]\s'
    match_list['spw']   = 'Spw=(?P<spw>\d+)\s'
    match_list['chan']  = 'Chan=(?P<chan>(\d+)|(<\d+~\d+>))\s'
    match_list['freq']  = '(Avg\s)*Freq=(?P<freq>[\d.]+)\s'
    match_list['corr']  = 'Corr=(?P<corr>\w+)\s'
    match_list['poln']  = 'Corr=(?P<corr>\w+)\s'

    info_matched = {}
    for item in match_list:
        p_match = re.compile(match_list[item])
        try:
            p_matched = p_match.search(info_line).groupdict()
            info_matched.update({item: p_matched[item]})
        except:
            if debug:
                print("Failed: {:<10}".format(item))
    # recover the baseline from two antennas
    try:
        two_ants = sorted([info_matched['ant1'], info_matched['ant2']])
        info_matched['baseline'] = two_ants[0] + '&' + two_ants[1]
    except:
        if debug:
            print("Failed to generate the baseline!")

    return info_matched

def pretty_output(counter):
    """make Couter output more readable

    """
    output = ''
    for item in counter:
        output += "{}[{}] ".format(item[0], item[1])
    return output

def generate_timerange(time_list):
    """comparing the string, find the smallest time string and the largest
    """
    if len(time_list) <= 1:
        return time_list[0]

    start_time = time_list[0]
    end_time = time_list[0]
    for item in time_list[1:]:
        if item < smallest:
            start_time = item
        if item > end_time:
            end_time = item
    return start_time +'~'+ end_time



def locating_flag(logfile, n=5, debug=False, vis=''):
    """Searching flag information in logfile
    
    Example
    -------
    As a package:
        from locating_flag import locating_flag
        locating_flag(logfile, n=5)
    In casa:
        execfile locating_flag.py

    Parameters
    ----------
    logfile : str
        the filename of the log
    n : int
        the number of most common outputs

    """
    n_select_start = 0
    n_select_end = 0
    p_select = re.compile('Found (?P<n_select>\d+) points \((?P<n_unflagged>\d+) unflagged\)')
    p_overselect = re.compile('Only first (?P<n_over>\d+) points reported above')
    
    with open(logfile) as logfile:
        all_lines = logfile.readlines()
    for ind in range(len(all_lines) - 1, 0, -1):
        p_matched = p_select.search(all_lines[ind])
        if p_matched:
            p_overmatched = p_overselect.search(all_lines[(ind - 1)])
            if p_overmatched:
                n_select_start = ind - int(p_overmatched.groupdict()['n_over'])
                print("Warning: only show the {} lines in the logs!".format(p_overmatched.groupdict()['n_over']))
            else:
                n_select_start = ind - int(p_matched.groupdict()['n_select'])
            n_select_end = ind
            break

    match_stat = {'ant1&ant2':[], 'baselines':[], 'spws':[], 'corrs':[], 
            'chans':[], 'scans':[], 'fields':[], 'time':[]}
    ants_all = []
    baselines_all = []

    for line in all_lines[n_select_start:n_select_end]:
        info_matched = match_info(line, debug=debug)
        for item_stat in match_stat:
            for item_info in info_matched:
                if item_info in item_stat:
                    match_stat[item_stat].append(info_matched[item_info])
    for item in match_stat:
        print("{}:\n{}\n".format(item, pretty_output(Counter(match_stat[item]).most_common(n))))

    # generate flagdata command
    flag_baseline = ''
    for baseline in Counter(match_stat['baselines']).most_common(n):
        flag_baseline += "{};".format(baseline[0])
    flag_scan = ''
    for scan in Counter(match_stat['scans']).most_common(n):
        flag_scan += "{},".format(scan[0])
    flag_corr = ''
    for corr in Counter(match_stat['corrs']).most_common(n):
        flag_corr += "{},".format(corr[0])
    flag_timerange = generate_timerange(match_stat['time'])

    print("flagdata(vis='{}', mode='manual', antenna='{}', scan='{}', correlation='{}', timerange='{}', flagbackup=False)".format(vis ,flag_baseline[:-1], flag_scan[:-1], flag_corr[:-1], flag_timerange))


if __name__ == '__main__':
    locating_flag(casalog.logfile())

#!/usr/bin/env python

# funtion to identify flagging data from the output in the logfile

# Author: Zhi-Yu Zhang, Jianhang Chen
# Email: pmozhang@gmail.com, cjhastro@gmail.com
# History:
#   21 Mar 2019, update by Zhiyu
#   30 Sep 2019, transfer into function from the original script of Zhiyu, by Jianhang
#    3 Dec 2019, v0.0.2: match logs by regex, by Jianhang
#   24 Jun 2020, v0.1.0, locating_flag can generate the flagdata automatically


import re
from collections import Counter
from datetime import datetime, timedelta

version = '0.1.0'

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

def generate_timerange(time_list, intt=1.0, avgtime=None):
    """comparing the string, find the smallest time string and the largest
    
    Parameters
    ----------
    time_list : str or list
        the list of time from the match_info
    intt : int or float
        the integrational time of the data
        default: 5s
    avgtime : int or float
        the average time interval during the inspection
        default: None
    """
    if isinstance(time_list, str):
        time_list = [time_list,]
    if len(Counter(time_list)) <= 1:
        if avgtime:
            start_time = datetime.strptime(time_list[0], '%Y/%m/%d/%H:%M:%S.%f') - timedelta(seconds=avgtime/2.0)
            end_time = datetime.strptime(time_list[0], '%Y/%m/%d/%H:%M:%S.%f') + timedelta(seconds=avgtime/2.0)
            return start_time.strftime('%Y/%m/%d/%H:%M:%S.%f') +'~'+ end_time.strftime('%Y/%m/%d/%H:%M:%S.%f')
        else:
            return time_list[0]

    start_time = time_list[0]
    end_time = time_list[0]
    for item in time_list[1:]:
        if item < start_time:
            start_time = item
        if item > end_time:
            end_time = item
    if avgtime:
        intt = avgtime
    start_time = datetime.strptime(start_time, '%Y/%m/%d/%H:%M:%S.%f') - timedelta(seconds=intt/2.0)
    end_time = datetime.strptime(end_time, '%Y/%m/%d/%H:%M:%S.%f') + timedelta(seconds=intt/2.0)
    return start_time.strftime('%Y/%m/%d/%H:%M:%S.%f') +'~'+ end_time.strftime('%Y/%m/%d/%H:%M:%S.%f')

def generate_spw(chan_list):
    if isinstance(chan_list, str):
        chan_list = [chan_list,]
    if '<' in chan_list[0]:
        chan_match = re.compile('<(\d*)~(\d*)>')
    else:
        chan_match = re.compile('(\d*)')
    if len(Counter(chan_list)) <= 1:
        chan_nums = chan_match.search(chan_list[0]).groups()
        return '*:' + chan_nums[0] + '~' + chan_nums[-1]
    chan_nums = chan_match.search(chan_list[0]).groups()
    start_chan = int(chan_nums[0])
    end_chan = int(chan_nums[-1])
    for item in chan_list:
        chan_nums = chan_match.search(item).groups()
        for num in chan_nums:
            item_int = int(num)
            if item_int < start_chan:
                start_chan = item_int
            if item_int > end_chan:
                end_chan = item_int
    return '*:' + str(start_chan) + '~' + str(end_chan)

def locating_flag(logfile, n=5, vis='', intt=1.0, avgtime=None,
                  mode='stat', flagfile=None,
                  show_timerange=True, show_spw=False, debug=False,):
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
        default: 5
    vis : str
        the vis of generated flagdata command
        default: ''
    intt : int or float
        integration time, in seconds
        default: 5
    avgtime : int or float
        the avgtime in plotms
        default: None
    mode : str
        'stat' or 'statistics': print out the antenna and baseline statistics
        'single': generate flagdata command for each point in plotms
        default: 'stat'
    show_timerange : bool
        generate the timerange in the flagdata command
        default: True
    show_spw : bool
        generate spw in the flagdata command
        default: False
    debug : bool
        printing the matching information in the match_info function
        default: False
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
    if mode == 'stat' or mode == 'statistics':
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
        
        flag_cmd = "vis={}, mode='manual', antenna='{}', scan='{}', correlation='{}', ".format(vis ,flag_baseline[:-1], flag_scan[:-1], flag_corr[:-1])
        if show_timerange:
            flag_timerange = generate_timerange(match_stat['time'], intt=intt, avgtime=avgtime)
            flag_cmd += "timerange='{}', ".format(flag_timerange)
        if show_spw:
            flag_spw = generate_spw(match_stat['chans'])
            flag_cmd += "spw='{}', ".format(flag_spw)
        print("flagdata({}flagbackup=False)".format(flag_cmd))

    elif mode == 'single':
        for line in all_lines[n_select_start:n_select_end]:
            info_matched = match_info(line, debug=debug)
            if debug:
                print("\nOriginal flagInfo:", line)
                print("Matched info:", info_matched)

            flag_cmd = "vis={}, mode='manual', antenna='{}', scan='{}', correlation='{}', ".format(vis ,info_matched['baseline'], 
                    info_matched['scan'], info_matched['corr'])
            if show_timerange:
                flag_timerange = generate_timerange(info_matched['time'], intt=intt, avgtime=avgtime)
                flag_cmd += "timerange='{}', ".format(flag_timerange)
            if show_spw:
                flag_spw = generate_spw(info_matched['chan'])
                flag_cmd += "spw='{}', ".format(flag_spw)
            if flagfile:
                with open(flagfile, "a") as ff:
                    ff.write("flagdata({}flagbackup=False)".format(flag_cmd))
            else:
                print("flagdata({}flagbackup=False)".format(flag_cmd))



if __name__ == '__main__':
    locating_flag(casalog.logfile())

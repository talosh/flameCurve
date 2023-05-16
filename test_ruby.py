from io import TextIOWrapper
import os
import sys
import numpy as np
from pprint import pprint

def dictify(r, root=True):
    from copy import copy

    if root:
        return {r.tag: dictify(r, False)}

    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(dictify(x, False))

    return d

def decode_tw_setup(temp_setup_path):
    import xml.etree.ElementTree as ET
    import math
    
    # return value is a dict of int frames as keys
    frame_value_map = {}

    with open(temp_setup_path, 'r') as tw_setup_file:
        tw_setup_string = tw_setup_file.read()
        tw_setup_file.close()

    tw_setup_xml = ET.fromstring(tw_setup_string)
    tw_setup_dict = dictify(tw_setup_xml)

    start_frame = math.floor(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['Start']))
    end_frame = math.ceil(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['End']))
    TW_SpeedTiming_size = int(tw_setup_dict['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])
    TW_RetimerMode = int(tw_setup_dict['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])

    frame_value_map = bake_flame_tw_setup(temp_setup_path, start_frame, end_frame)

    return frame_value_map

def bake_flame_tw_setup(tw_setup_path, start, end):
    # parses tw setup from flame and returns dictionary
    # with baked frame - value pairs
    
    def extrapolate_linear(xa, ya, xb, yb, xc):
        m = (ya - yb) / (xa - xb)
        yc = (xc - xb) * m + yb
        return yc

    def dictify(r, root=True):
        from copy import copy

        if root:
            return {r.tag: dictify(r, False)}

        d = copy(r.attrib)
        if r.text:
            d["_text"] = r.text
        for x in r.findall("./*"):
            if x.tag not in d:
                d[x.tag] = []
            d[x.tag].append(dictify(x, False))
        return d

    import xml.etree.ElementTree as ET

    frame_value_map = {}

    with open(tw_setup_path, 'r') as tw_setup_file:
        tw_setup_string = tw_setup_file.read()
        tw_setup_file.close()
        tw_setup_xml = ET.fromstring(tw_setup_string)
        tw_setup = dictify(tw_setup_xml)

    # start = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
    # end = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
    # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])
    TW_SpeedTiming_size = int(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])
    TW_RetimerMode = int(tw_setup['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])
    parsed_and_baked_path = os.path.join(os.path.dirname(__file__), 'parsed_and_baked.txt')

    if sys.platform == 'darwin':
        parser_and_baker = os.path.join(os.path.dirname(__file__), 'flame_channel_parser', 'bin', 'bake_flame_channel_mac')
    else:
        parser_and_baker = os.path.join(os.path.dirname(__file__), 'flame_channel_parser', 'bin', 'bake_flame_channel')

    '''
    if TW_SpeedTiming_size == 1 and TW_RetimerMode == 0:
        # just constant speed change with no keyframes set       
        x = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Frame'][0]['_text'])
        y = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Value'][0]['_text'])
        ldx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dX'][0]['_text'])
        ldy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dY'][0]['_text'])
        rdx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dX'][0]['_text'])
        rdy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dY'][0]['_text'])

        for frame_number in range(start, end+1):
            frame_value_map[frame_number] = extrapolate_linear(x + ldx, y + ldy, x + rdx, y + rdy, frame_number)
    
        return frame_value_map
    '''
    
    # add point tangents from vecrors to match older version of setup
    # used by Julik's parser

    from xml.dom import minidom
    xml = minidom.parse(tw_setup_path)
    keys = xml.getElementsByTagName('Key')
    for key in keys:        
        frame = key.getElementsByTagName('Frame')
        if frame:
            frame = (frame[0].firstChild.nodeValue)
        value = key.getElementsByTagName('Value')
        if value:
            value = (value[0].firstChild.nodeValue)
        rdx = key.getElementsByTagName('RHandle_dX')
        if rdx:
            rdx = (rdx[0].firstChild.nodeValue)
        rdy = key.getElementsByTagName('RHandle_dY')
        if rdy:
            rdy = (rdy[0].firstChild.nodeValue)
        ldx = key.getElementsByTagName('LHandle_dX')
        if ldx:
            ldx = (ldx[0].firstChild.nodeValue)
        ldy = key.getElementsByTagName('LHandle_dY')
        if ldy:
            ldy = (ldy[0].firstChild.nodeValue)

        lx = xml.createElement('LHandleX')
        lx.appendChild(xml.createTextNode('{:.6f}'.format(float(frame) + float(ldx))))
        key.appendChild(lx)
        ly = xml.createElement('LHandleY')
        ly.appendChild(xml.createTextNode('{:.6f}'.format(float(value) + float(ldy))))
        key.appendChild(ly)
        rx = xml.createElement('RHandleX')
        rx.appendChild(xml.createTextNode('{:.6f}'.format(float(frame) + float(rdx))))
        key.appendChild(rx)
        ry = xml.createElement('RHandleY')
        ry.appendChild(xml.createTextNode('{:.6f}'.format(float(value) + float(rdy))))
        key.appendChild(ry)

    xml_string = xml.toxml()
    dirname, name = os.path.dirname(tw_setup_path), os.path.basename(tw_setup_path)
    xml_path = os.path.join(dirname, 'fix_' + name)
    with open(xml_path, 'a') as xml_file:
        xml_file.write(xml_string)
        xml_file.close()

    intp_start = start
    intp_end = end

    if TW_RetimerMode == 0:
        tw_speed = {}
        tw_speed_frames = []
        TW_Speed = xml.getElementsByTagName('TW_SpeedTiming')
        keys = TW_Speed[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_speed[int(index)] = {'frame': int(frame), 'value': float(value)}
            tw_speed_frames.append(int(frame))

            intp_start = min(start, min(tw_speed_frames))
            intp_end = max(end, max(tw_speed_frames))
    else:
        tw_timing = {}
        tw_timing_frames = []
        TW_Timing = xml.getElementsByTagName('TW_Timing')
        keys = TW_Timing[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_timing[int(index)] = {'frame': int(frame), 'value': float(value)}
            tw_timing_frames.append(int(frame))

            intp_start = min(start, min(tw_timing_frames))
            intp_end = max(end, max(tw_timing_frames))

    tw_channel_name = 'Speed' if TW_RetimerMode == 0 else 'Timing'

    cmd = parser_and_baker + ' -c ' + tw_channel_name
    cmd += ' -s ' + str(intp_start) + ' -e ' + str(intp_end)
    cmd += ' --to-file ' + parsed_and_baked_path + ' ' + xml_path
    pprint (cmd)
    os.system(cmd)

    if not os.path.isfile(parsed_and_baked_path):
        print ('can not find parsed channel file %s' % parsed_and_baked_path)
        input("Press Enter to continue...")
        sys.exit(1)

    tw_channel = {}
    with open(parsed_and_baked_path, 'r') as parsed_and_baked:
        import re
        
        # taken from Julik's parser

        CORRELATION_RECORD = re.compile(
        r"""
        ^([-]?\d+)            # -42 or 42
        \t                    # tab
        (
            [-]?(\d+(\.\d*)?) # "-1" or "1" or "1.0" or "1."
            |                 # or:
            \.\d+             # ".2"
        )
        ([eE][+-]?[0-9]+)?    # "1.2e3", "1.2e-3" or "1.2e+3"
        $
        """, re.VERBOSE)
    
        lines = parsed_and_baked.readlines()
        for i, line in enumerate(lines):
            line = line.rstrip()
            m = CORRELATION_RECORD.match(line)
            if m is not None:
                frame_number = int(m.group(1))
                value = float(m.group(2))
                tw_channel[frame_number] = value

    if TW_RetimerMode == 1:
        # job's done for 'Timing' channel
        return tw_channel

    else:
        # speed - based timewarp needs a bit more love
        # to solve frame values against speed channel
        # with the help of anchor frames in SpeedTiming channel

        tw_speed_timing = {}
        TW_SpeedTiming = xml.getElementsByTagName('TW_SpeedTiming')
        keys = TW_SpeedTiming[0].getElementsByTagName('Key')
        for key in keys:
            index = key.getAttribute('Index') 
            frame = key.getElementsByTagName('Frame')
            if frame:
                frame = (frame[0].firstChild.nodeValue)
            value = key.getElementsByTagName('Value')
            if value:
                value = (value[0].firstChild.nodeValue)
            tw_speed_timing[int(index)] = {'frame': int(frame), 'value': float(value)}
        
        if tw_speed_timing[0]['frame'] > start:
            # we need to extrapolate backwards from the first 
            # keyframe in SpeedTiming channel

            anchor_frame_value = tw_speed_timing[0]['value']
            for frame_number in range(tw_speed_timing[0]['frame'] - 1, start - 1, -1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                else:
                    step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                frame_value_map[frame_number] = anchor_frame_value - step_back
                anchor_frame_value = frame_value_map[frame_number]

        # build up frame values between keyframes of SpeedTiming channel
        for key_frame_index in range(0, len(tw_speed_timing.keys()) - 1):
            # The value from my gess algo is close to the one in flame but not exact
            # and error is accumulated. SO quick and dirty way is to do forward
            # and backward pass and mix them rationally

            range_start = tw_speed_timing[key_frame_index]['frame']
            range_end = tw_speed_timing[key_frame_index + 1]['frame']
            
            if range_end == range_start + 1:
            # keyframes on next frames, no need to interpolate
                frame_value_map[range_start] = tw_speed_timing[key_frame_index]['value']
                frame_value_map[range_end] = tw_speed_timing[key_frame_index + 1]['value']
                continue

            forward_pass = {}
            anchor_frame_value = tw_speed_timing[key_frame_index]['value']
            forward_pass[range_start] = anchor_frame_value

            for frame_number in range(range_start + 1, range_end):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step = tw_channel[max(list(tw_channel.keys()))] / 100
                else:
                    step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                forward_pass[frame_number] = anchor_frame_value + step
                anchor_frame_value = forward_pass[frame_number]
            forward_pass[range_end] = tw_speed_timing[key_frame_index + 1]['value']
            
            backward_pass = {}
            anchor_frame_value = tw_speed_timing[key_frame_index + 1]['value']
            backward_pass[range_end] = anchor_frame_value
            
            for frame_number in range(range_end - 1, range_start -1, -1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step_back = tw_channel[min(list(tw_channel.keys()))] / 100
                else:
                    step_back = (tw_channel[frame_number + 1] + tw_channel[frame_number]) / 200
                backward_pass[frame_number] = anchor_frame_value - step_back
                anchor_frame_value = backward_pass[frame_number]
            
            backward_pass[range_start] = tw_speed_timing[key_frame_index]['value']

            # create easy in and out soft mixing curve
            import numpy as np

            ctr =np.array( [(0 , 0), (0.1, 0), (0.9, 1),  (1, 1)])
            x=ctr[:,0]
            y=ctr[:,1]            
            
            from bruteforce import interpolate
            interp = interpolate.CubicSpline(x, y)

            work_range = list(forward_pass.keys())
            ratio = 0
            rstep = 1 / len(work_range)
            for frame_number in sorted(work_range):
                frame_value_map[frame_number] = forward_pass[frame_number] * (1 - interp(ratio)) + backward_pass[frame_number] * interp(ratio)
                ratio += rstep
        
        last_key_index = list(sorted(tw_speed_timing.keys()))[-1]
        if tw_speed_timing[last_key_index]['frame'] < end:
            # we need to extrapolate further on from the 
            # last keyframe in SpeedTiming channel
            anchor_frame_value = tw_speed_timing[last_key_index]['value']
            frame_value_map[tw_speed_timing[last_key_index]['frame']] = anchor_frame_value

            for frame_number in range(tw_speed_timing[last_key_index]['frame'] + 1, end + 1):
                if frame_number + 1 not in tw_channel.keys() or frame_number not in tw_channel.keys():
                    step = tw_channel[max(list(tw_channel.keys()))] / 100
                else:
                    step = (tw_channel[frame_number] + tw_channel[frame_number + 1]) / 200
                frame_value_map[frame_number] = anchor_frame_value + step
                anchor_frame_value = frame_value_map[frame_number]

        return frame_value_map

temp_setup_path = sys.argv[1]
if not temp_setup_path:
    print ('no file to parse')
    sys.exit()
keys = decode_tw_setup(temp_setup_path)

pprint (keys)
sys.exit()



HERMATRIX = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
print (HERMATRIX)
hermite_vector = np.array([9.0, 12.0, 18.78345865418239, -12.138369540337518])
print (hermite_vector)
hermite_basis = HERMATRIX.dot(hermite_vector)
print (hermite_basis)

start_frame = 1
end_frame = 8

for x in range(start_frame, end_frame+1):
    t = float(x - start_frame) / (end_frame - start_frame)
    print ("frame = %s" % x)
    print ("t = %s" % t)
    multipliers_vec = np.array([t ** 3,  t ** 2, t ** 1, t ** 0])
    print ("multipliers vec: %s" % multipliers_vec)
    sum = 0.0
    for i in range (0, 4):
        sum += hermite_basis[i] * multipliers_vec[i]
    print (sum)
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
    
    with open(temp_setup_path, 'r') as tw_setup_file:
        tw_setup_string = tw_setup_file.read()
        tw_setup_file.close()

    tw_setup_xml = ET.fromstring(tw_setup_string)
    tw_setup_dict = dictify(tw_setup_xml)

    start_frame = math.floor(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['Start']))
    end_frame = math.ceil(float(tw_setup_dict['Setup']['Base'][0]['Range'][0]['End']))
    TW_SpeedTiming_size = int(tw_setup_dict['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size'][0]['_text'])
    TW_RetimerMode = int(tw_setup_dict['Setup']['State'][0]['TW_RetimerMode'][0]['_text'])

    return start_frame, end_frame, tw_setup_string

def bake_flame_tw_setup(tw_setup_string, start, end):
    import numpy as np
    import xml.etree.ElementTree as ET
    import re

    # parses tw setup from flame and returns dictionary
    # with baked frame - value pairs
    
    def extrapolate_linear(xa, ya, xb, yb, xc):
        m = (ya - yb) / (xa - xb)
        yc = (xc - xb) * m + yb
        return yc

    def dictify(r, root=True):
        def string_to_value(s):
            if (s.find('-') <= 0) and s.replace('-', '', 1).isdigit():
                return int(s)
            elif (s.find('-') <= 0) and (s.count('.') < 2) and \
                    (s.replace('-', '', 1).replace('.', '', 1).isdigit()):
                return float(s)
            elif s == 'True':
                return True
            elif s == 'False':
                return False
            else:
                return s

        from copy import copy

        if root:
            return {r.tag: dictify(r, False)}

        d = copy(r.attrib)
        if r.text:
            # d["_text"] = r.text
            d = r.text
        for x in r.findall('./*'):
            if x.tag not in d:
                v = dictify(x, False)
                if isinstance (v, str):
                    d[x.tag] = string_to_value(v)
                else:
                    d[x.tag] = []
            if isinstance(d[x.tag], list):
                d[x.tag].append(dictify(x, False))
        return d

    class FlameChannellInterpolator:
        # An attempt of a python rewrite of Julit Tarkhanov's original
        # Flame Channel Parsr written in Ruby.

        class ConstantSegment:
            def __init__(self, from_frame, to_frame, value):
                self.NEG_INF = float('-inf')
                self.POS_INF = float('inf')
                self.start_frame = from_frame
                self.end_frame = to_frame
                self.v1 = value

            def defines(self, frame):
                return (frame < self.end_frame) and (frame >= self.start_frame)

            def value_at(self, frame):
                return self.v1

        class ConstantFunction(ConstantSegment):
            def __init__(self, value):
                super().__init__(float('-inf'), float('inf'), value)

            def defines(self, frame):
                return True

            def value_at(self, frame):
                return self.v1

        def __init__(self, channel):
            self.NEG_INF = float('-inf')
            self.POS_INF = float('inf')

            self.segments = []
            self.extrap = channel.get('Extrap', 'constant')

            if channel.get('Size', 0) == 0:
                self.segments = [ConstantFunction(channel.get('Value', 0))]
            else:
                self.segments = self.create_segments_from_channel(channel)

    def create_segments_from_channel(self, channel):
        # First the prepolating segment
        segments = [pick_prepolation(channel.extrapolation, channel[0], channel[1])]

        # Then all the intermediate segments, one segment between each pair of keys
        for index, key in enumerate(channel[:-1]):
            segments.append(self.key_pair_to_segment(key, channel[index + 1]))

        # and the extrapolator
        segments.append(pick_extrapolation(channel.extrapolation, channel[-2], channel[-1]))
        return segments

    

        def sample_from_segments(self, at_frame):
            for segment in self.segments:
                if segment.defines(at_frame):
                    return segment.value_at(at_frame)
            raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')


    frame_value_map = {}
    tw_setup_xml = ET.fromstring(tw_setup_string)
    tw_setup = dictify(tw_setup_xml)

    # start = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
    # end = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
    # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])

    TW_SpeedTiming_size = tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size']
    TW_RetimerMode = tw_setup['Setup']['State'][0]['TW_RetimerMode']

    if TW_SpeedTiming_size == 1 and TW_RetimerMode == 0:
        # just constant speed change with no keyframes set       
        x = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Frame'][0])
        y = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['Value'][0])
        ldx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dX'][0])
        ldy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['LHandle_dY'][0])
        rdx = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dX'][0])
        rdy = float(tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['KFrames'][0]['Key'][0]['RHandle_dY'][0])

        for frame_number in range(start, end+1):
            frame_value_map[frame_number] = extrapolate_linear(x + ldx, y + ldy, x + rdx, y + rdy, frame_number)
    
        return frame_value_map

    tw_channel = 'TW_Speed' if TW_RetimerMode == 0 else 'TW_Timing'
    channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
    channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Index'])}
    pprint (channel)
    sys.exit()

    # np.searchsorted(np.array(list(s.keys())), -40)

    # add point tangents from vecrors to match older version of setup
    # used by Julik's parser

    from xml.dom import minidom
    xml = minidom.parseString(tw_setup_string)
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

    tw_oldstyle_xml_string = xml.toprettyxml()

    intp_start = start
    intp_end = end

    if TW_RetimerMode == 0:
        tw_speed = {}
        tw_speed_frames = []
        TW_Speed = xml.getElementsByTagName('TW_Speed')
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

    options = {
        'channel': tw_channel_name,
        'start_frame': intp_start,
        'end_frame': intp_end,
        'on_curve_limits': False,
        'destination': 'destination'
    }

    # channels = FlameChannelParser.parse(tw_oldstyle_xml_string)
    interpolator = FlameChannelParser.Interpolator(tw_channel_name)
    tw_channel = {}

    for frame_number in range(intp_start, intp_end+1):
        tw_channel[frame_number] =interpolator.sample_at(frame_number)

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
            
            def spline(x_new):
                ctr =np.array( [(0 , 0), (0.1, 0), (0.9, 1),  (1, 1)])
                x=ctr[:,0]
                y=ctr[:,1]            

                # Find the index value s falls between in the array x
                idx = np.searchsorted(x, x_new)

                # Check if x_new is outside the range of x
                if idx == 0:
                    return y[0]
                elif idx == len(x):
                    return y[-1]

                # Calculate the coefficients of the cubic polynomial
                h = x[idx] - x[idx-1]
                a = (x[idx]-x_new)/h
                b = (x_new-x[idx-1])/h
                c = (a**3 - a)*h**2/6
                d = (b**3 - b)*h**2/6

                # Interpolate using the cubic polynomial
                y_new = a*y[idx-1] + b*y[idx] + c*y[idx-1] + d*y[idx]

                return y_new

            work_range = list(forward_pass.keys())
            ratio = 0
            rstep = 1 / len(work_range)
            for frame_number in sorted(work_range):
                frame_value_map[frame_number] = forward_pass[frame_number] * (1 - spline(ratio)) + backward_pass[frame_number] * spline(ratio)
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

def main():
    if len(sys.argv) < 2:
        print ('usage: %s flame_setup [start_frame] [end_frame]'% os.path.basename(__file__))
    temp_setup_path = sys.argv[1]
    if not temp_setup_path:
        print ('no file to parse')
        sys.exit()

    start, end, tw_setup_string = decode_tw_setup(temp_setup_path)

    if len(sys.argv) < 4:
        start_frame = start
        end_frame = end
    else:
        start_frame = int(sys.argv[2])
        end_frame = int(sys.argv[3])

    keys = bake_flame_tw_setup(tw_setup_string, start_frame, end_frame)

    pprint (keys)
    sys.exit()


if __name__ == '__main__':
    main()

'''
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
'''
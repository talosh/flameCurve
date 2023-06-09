from io import TextIOWrapper
import os
import sys
import numpy as np
from pprint import pprint, pformat

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

def bake_flame_tw_setup(tw_setup_string, start_frame, end_frame):
    import numpy as np
    import xml.etree.ElementTree as ET

    # parses tw setup from flame and returns dictionary
    # with baked frame - value pairs
    
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
                self._mode = 'constant'
                self.start_frame = from_frame
                self.end_frame = to_frame
                self.v1 = value

            def mode(self):
                return self._mode

            def defines(self, frame):
                return (frame < self.end_frame) and (frame >= self.start_frame)

            def value_at(self, frame):
                return self.v1

        class LinearSegment(ConstantSegment):
            def __init__(self, from_frame, to_frame, value1, value2):
                self.vint = (value2 - value1)
                super().__init__(from_frame, to_frame, value1)
                self._mode = 'linear'

            def value_at(self, frame):
                on_t_interval = (frame - self.start_frame) / (self.end_frame - self.start_frame)
                return self.v1 + (on_t_interval * self.vint)
            
            '''
            self.HERMATRIX = np.array([
                [2,  -2,  1,  1],
                [-3, 3,   -2, -1],
                [0,   0,  1,  0],
                [1,   0,  0,  0]
            ])
            '''

        class HermiteSegment(LinearSegment):
            def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                self.start_frame, self.end_frame = from_frame, to_frame
                frame_interval = (self.end_frame - self.start_frame)
                self._mode = 'hermite'

                self.HERMATRIX = np.array([
                    [0,  0,  0,  1],
                    [1,  1,  1,  1],
                    [0,  0,  1,  0],
                    [3,  2,  1,  0]
                ])
                self.HERMATRIX = np.linalg.inv(self.HERMATRIX)

                # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                # CC = {P1, P2, T1, T2}
                p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                self.hermite = np.array([p1, p2, t1, t2])
                self.basis = np.dot(self.HERMATRIX, self.hermite)

            def value_at(self, frame):
                if frame == self.start_frame:
                    return self.hermite[0]

                # Get the 0 < T < 1 interval we will interpolate on
                # Q[frame_] = P[ ( frame - 149 ) / (time_to - time_from)]
                t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                # S[s_] = {s^3, s^2, s^1, s^0}
                multipliers_vec = np.array([t ** 3, t ** 2, t ** 1, t ** 0])

                # P[s_] = S[s].h.CC
                interpolated_scalar = np.dot(self.basis, multipliers_vec)
                return interpolated_scalar

        class HermiteSegmentQuartic(LinearSegment):
            '''
            Make sure not to confuse them with quartic (= degree 4) Hermite splines, 
            which are defined by 5 values per segment: function value and first derivative at both ends, 
            and one of the second derivatives.
            '''
            '''
            P(x0) = y0    =>    a0 + a1*x0 + a2*x0^2 + a3*x0^3 + a4*x0^4 = y0   --(1)
            P(x1) = y1    =>    a0 + a1*x1 + a2*x1^2 + a3*x1^3 + a4*x1^4 = y1   --(2)
            
            P'(x0) = tx0    =>    a1 + 2*a2*x0 + 3*a3*x0^2 + 4*a4*x0^3 = tx0   --(3)
            P'(x1) = tx1    =>    a1 + 2*a2*x1 + 3*a3*x1^2 + 4*a4*x1^3 = tx1   --(4)

            We have four equations (equations (1)-(4)) and five unknowns (a0, a1, a2, a3, a4). 
            To solve this system of equations, we can rewrite it in matrix form:

            A * X = B

            where A is the coefficient matrix, 
            X is the column vector of unknowns, 
            and B is the column vector of constants.

            A = | 1   x0   x0^2   x0^3   x0^4 |
                | 1   x1   x1^2   x1^3   x1^4 |
                | 0   1    2*x0   3*x0^2 4*x0^3 |
                | 0   1    2*x1   3*x1^2 4*x1^3 |

            X = | a0 |
                | a1 |
                | a2 |
                | a3 |
                | a4 |

            B = | y0 |
                | y1 |
                | tx0 |
                | tx1 |

            To solve for X, we can compute X = inv(A) * B, where inv(A) is the inverse of matrix A.
            Once we have the values of a0, a1, a2, a3, and a4, we can substitute them back 
            into the quartic polynomial P(x) to obtain the interpolated values for any desired x.

            P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
            
            '''
            def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                self.start_frame, self.end_frame = from_frame, to_frame
                frame_interval = (self.end_frame - self.start_frame)
                self._mode = 'hermite'
                self.a = 0
                self.b = 0
                self.value1 = value1
                self.value2 = value2
                self.tangent1 = tangent1
                self.tangent2 = tangent2
                self.frame_interval = frame_interval

                self.HERMATRIX = np.array([
                            [2, -2, 1, 1],
                            [-3, 3, -2, -1],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [2, 0, 0, 0]
                        ])
                
                # self.HERMATRIX = np.linalg.inv(self.HERMATRIX)
                # pprint (self.HERMATRIX)

                # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                # CC = {P1, P2, T1, T2}
                p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                self.hermite = np.array([p1, p2, t1, t2])
                pprint (self.hermite)
                self.basis = np.dot(self.HERMATRIX, self.hermite)
                pprint (self.basis)

            def value_at(self, frame):
                if frame == self.start_frame:
                    return self.hermite[0]

                # Get the 0 < T < 1 interval we will interpolate on
                t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                # S[s_] = {s^4, s^3, s^2, s^1, s^0}
                multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])

                # cubic functions
                a0 = (1 - 3 * (t ** 2) + 2 * (t ** 3))
                a1 = (3 * (t ** 2) - 2 * (t ** 3))
                b0 = (t - 2* (t ** 2) + t**3)
                b1 = -1 * (t ** 2) + (t ** 3)

                # quatric functions
                alpha = self.a
                beta = self.b

                aa0 = 1 + (alpha - 3)*(t ** 2) + 2 * (1 - alpha) * (t ** 3) + alpha * (t ** 4)
                aa1 = (3 - alpha) * (t **2) + 2 * (alpha - 1) * (t ** 3) - alpha * (t ** 4)
                bb0 = t + (beta - 2) * (t ** 2) + (1 - 2 * beta) * (t ** 3) + beta * (t ** 4)
                bb1 = -1 * (beta + 1) * (t ** 2) + (2 * beta + 1) * (t ** 3) - beta * (t ** 4)

                # P[s_] = S[s].h.CC
                # interpolated_scalar = np.dot(self.basis, multipliers_vec)
                p1, p2, t1, t2 = self.value1, self.value2, self.tangent1 * self.frame_interval, self.tangent2 * self.frame_interval
                # interpolated_scalar = a0*p1 + a1*p2 + b0*t1 + b1*t2
                interpolated_scalar = aa0*p1 + aa1*p2 + bb0*t1 + bb1*t2
                return interpolated_scalar

        class HermiteSegmentQuartic5x5(LinearSegment):
            '''
            Make sure not to confuse them with quartic (= degree 4) Hermite splines, 
            which are defined by 5 values per segment: function value and first derivative at both ends, 
            and one of the second derivatives.
            '''
            '''
            P(x0) = y0    =>    a0 + a1*x0 + a2*x0^2 + a3*x0^3 + a4*x0^4 = y0   --(1)
            P(x1) = y1    =>    a0 + a1*x1 + a2*x1^2 + a3*x1^3 + a4*x1^4 = y1   --(2)
            
            P'(x0) = tx0    =>    a1 + 2*a2*x0 + 3*a3*x0^2 + 4*a4*x0^3 = tx0   --(3)
            P'(x1) = tx1    =>    a1 + 2*a2*x1 + 3*a3*x1^2 + 4*a4*x1^3 = tx1   --(4)

            We have four equations (equations (1)-(4)) and five unknowns (a0, a1, a2, a3, a4). 
            To solve this system of equations, we can rewrite it in matrix form:

            A * X = B

            where A is the coefficient matrix, 
            X is the column vector of unknowns, 
            and B is the column vector of constants.

            A = | 1   x0   x0^2   x0^3   x0^4 |
                | 1   x1   x1^2   x1^3   x1^4 |
                | 0   1    2*x0   3*x0^2 4*x0^3 |
                | 0   1    2*x1   3*x1^2 4*x1^3 |

            X = | a0 |
                | a1 |
                | a2 |
                | a3 |
                | a4 |

            B = | y0 |
                | y1 |
                | tx0 |
                | tx1 |

            To solve for X, we can compute X = inv(A) * B, where inv(A) is the inverse of matrix A.
            Once we have the values of a0, a1, a2, a3, and a4, we can substitute them back 
            into the quartic polynomial P(x) to obtain the interpolated values for any desired x.

            P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
            
            '''
            def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
                self.start_frame, self.end_frame = from_frame, to_frame
                frame_interval = (self.end_frame - self.start_frame)
                self._mode = 'hermite'

                self.HERMATRIX = np.array([
                            [0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0],
                            [4, 3, 2, 1, 0],
                            [0, 0, 2, 0, 0]
                        ])
                
                self.HERMATRIX = np.linalg.inv(self.HERMATRIX)
                pprint (self.HERMATRIX)

                # Default tangents in flame are 0, so when we do None.to_f this is what we will get
                # CC = {P1, P2, T1, T2}
                p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
                self.hermite = np.array([p1, p2, t1, t2, 0])
                pprint (self.hermite)
                self.basis = np.dot(self.HERMATRIX, self.hermite)
                pprint (self.basis)

            def value_at(self, frame):
                if frame == self.start_frame:
                    return self.hermite[0]

                # Get the 0 < T < 1 interval we will interpolate on
                t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

                # S[s_] = {s^4, s^3, s^2, s^1, s^0}
                multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])

                # P[s_] = S[s].h.CC
                interpolated_scalar = np.dot(self.basis, multipliers_vec)
                return interpolated_scalar

        class BezierSegment(LinearSegment):
            class Pt:
                def __init__(self, x, y, tanx, tany):
                    self.x = x
                    self.y = y
                    self.tanx = tanx
                    self.tany = tany
            
            def __init__(self, x1, x2, y1, y2, t1x, t1y, t2x, t2y):
                super().__init__(x1, x2, y1, y2)
                self.a = self.Pt(x1, y1, t1x, t1y)
                self.b = self.Pt(x2, y2, t2x, t2y)
                self._mode = 'bezier'

            def value_at(self, frame):
                if frame == self.start_frame:
                    return self.a.y
                
                t = self.approximate_t(frame, self.a.x, self.a.tanx, self.b.tanx, self.b.x)
                vy = self.bezier(t, self.a.y, self.a.tany, self.b.tany, self.b.y)
                return vy
            
            def bezier(self, t, a, b, c, d):
                return a + (a*(-3) + b*3)*(t) + (a*3 - b*6 + c*3)*(t**2) + (-a + b*3 - c*3 + d)*(t**3)
            
            def clamp(self, value):
                if value < 0:
                    return 0.0
                elif value > 1:
                    return 1.0
                else:
                    return value
            
            APPROXIMATION_EPSILON = 1.0e-09
            VERYSMALL = 1.0e-20
            MAXIMUM_ITERATIONS = 100
            
            def approximate_t(self, atX, p0x, c0x, c1x, p1x):
                if atX - p0x < self.VERYSMALL:
                    return 0.0
                elif p1x - atX < self.VERYSMALL:
                    return 1.0

                u, v = 0.0, 1.0
                
                for i in range(self.MAXIMUM_ITERATIONS):
                    a = (p0x + c0x) / 2.0
                    b = (c0x + c1x) / 2.0
                    c = (c1x + p1x) / 2.0
                    d = (a + b) / 2.0
                    e = (b + c) / 2.0
                    f = (d + e) / 2.0
                    
                    if abs(f - atX) < self.APPROXIMATION_EPSILON:
                        return self.clamp((u + v) * 0.5)
                    
                    if f < atX:
                        p0x = f
                        c0x = e
                        c1x = c
                        u = (u + v) / 2.0
                    else:
                        c0x = a
                        c1x = d
                        p1x = f
                        v = (u + v) / 2.0
                
                return self.clamp((u + v) / 2.0)


        class ConstantPrepolate(ConstantSegment):
            def __init__(self, to_frame, base_value):
                super().__init__(float('-inf'), to_frame, base_value)
                self._mode = 'ConstantPrepolate'

            def value_at(self, frame):
                return self.v1

        class ConstantExtrapolate(ConstantSegment):
            def __init__(self, from_frame, base_value):
                super().__init__(from_frame, float('inf'), base_value)
                self._mode = 'ConstantExtrapolate'

            def value_at(self, frame):
                return self.v1
            
        class LinearPrepolate(ConstantPrepolate):
            def __init__(self, to_frame, base_value, tangent):
                self.tangent = float(tangent)
                super().__init__(to_frame, base_value)
                self._mode = 'LinearPrepolate'

            def value_at(self, frame):
                frame_diff = (self.end_frame - frame)
                return self.v1 + (self.tangent * frame_diff)
            
        class LinearExtrapolate(ConstantExtrapolate):
            def __init__(self, from_frame, base_value, tangent):
                self.tangent = float(tangent)
                super().__init__(from_frame, base_value)
                self._mode = 'LinearExtrapolate'

            def value_at(self, frame):
                frame_diff = (frame - self.start_frame)
                return self.v1 + (self.tangent * frame_diff)

        class ConstantFunction(ConstantSegment):
            def __init__(self, value):
                super().__init__(float('-inf'), float('inf'), value)
                self._mode = 'ConstantFunction'

            def defines(self, frame):
                return True

            def value_at(self, frame):
                return self.v1


        def __init__(self, channel):
            self.segments = []
            self.extrap = channel.get('Extrap', 'constant')

            if channel.get('Size', 0) == 0:
                self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
            elif channel.get('Size') == 1 and self.extrap == 'constant':
                self.segments = [FlameChannellInterpolator.ConstantFunction(channel.get('Value', 0))]
            elif channel.get('Size') == 1 and self.extrap == 'linear':
                kframes = channel.get('KFrames')
                frame = list(kframes.keys())[0]
                base_value = kframes[frame].get('Value')
                left_tangent = kframes[frame].get('LHandle_dY') / kframes[frame].get('LHandle_dX') * -1
                right_tangent = kframes[frame].get('RHandle_dY') / kframes[frame].get('RHandle_dX')
                self.segments = [
                    FlameChannellInterpolator.LinearPrepolate(frame, base_value, left_tangent),
                    FlameChannellInterpolator.LinearExtrapolate(frame, base_value, right_tangent)
                ]
            else:
                self.segments = self.create_segments_from_channel(channel)

        def sample_at(self, frame):
            if self.extrap == 'cycle':
                return self.sample_from_segments(self.frame_number_in_cycle(frame))
            elif self.extrap == 'revcycle':
                return self.sample_from_segments(self.frame_number_in_revcycle(frame))
            else:
                return self.sample_from_segments(frame)

        def first_defined_frame(self):
            first_f = self.segments[0].end_frame
            if first_f == float('-inf'):
                return 1
            return first_f

        def last_defined_frame(self):
            last_f = self.segments[-1].start_frame
            if last_f == float('inf'):
                return 100
            return last_f

        def frame_number_in_revcycle(self, frame):
            animated_across = self.last_defined_frame() - self.first_defined_frame()
            offset = abs(frame - self.first_defined_frame())
            absolute_unit = offset % animated_across
            cycles = offset // animated_across
            if cycles % 2 == 0:
                return self.first_defined_frame() + absolute_unit
            else:
                return self.last_defined_frame() - absolute_unit

        def frame_number_in_cycle(self, frame):
            animated_across = self.last_defined_frame() - self.first_defined_frame()
            offset = frame - self.first_defined_frame()
            modulo = offset % animated_across
            return self.first_defined_frame() + modulo

        def create_segments_from_channel(self, channel):
            kframes = channel.get('KFrames')
            index_frames = list(kframes.keys())
            # First the prepolating segment
            segments = [self.pick_prepolation(channel.get('Extrap', 'constant'), kframes[index_frames[0]], kframes[index_frames[1]])]

            # Then all the intermediate segments, one segment between each pair of keys
            for index, key in enumerate(index_frames[:-1]):
                segments.append(self.key_pair_to_segment(kframes[key], kframes[index_frames[index + 1]]))

            # and the extrapolator
            segments.append(self.pick_extrapolation(channel.get('Extrap', 'constant'), kframes[index_frames[-2]], kframes[index_frames[-1]]))
            return segments

        def sample_from_segments(self, at_frame):
            for segment in self.segments:
                if segment.defines(at_frame):
                    return segment.value_at(at_frame)
            raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
        
        def segment_mode(self, at_frame):
            for segment in self.segments:
                if segment.defines(at_frame):
                    return segment.mode()
            raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')
        
        def get_segment(self, at_frame):
            for segment in self.segments:
                if segment.defines(at_frame):
                    return segment
            raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')

        def pick_prepolation(self, extrap_symbol, first_key, second_key):
            if extrap_symbol == 'linear' and second_key:
                if first_key.get('CurveMode') != 'linear':
                    first_key_left_slope = first_key.get('LHandle_dY') / first_key.get('LHandle_dX') * -1
                    return FlameChannellInterpolator.LinearPrepolate(
                        first_key.get('Frame'), 
                        first_key.get('Value'), 
                        first_key_left_slope)
                else:
                    # For linear keys the tangent actually does not do anything, so we need to look a frame
                    # ahead and compute the increment
                    increment = (second_key.get('Value') - first_key.get('Value')) / (second_key.get('Frame') - first_key.get('Frame'))
                    return FlameChannellInterpolator.LinearPrepolate(first_key.get('Frame'), first_key.get('Value'), increment)
            else:
                return FlameChannellInterpolator.ConstantPrepolate(first_key.get('Frame'), first_key.get('Value'))
        
        def pick_extrapolation(self, extrap_symbol, previous_key, last_key):
            if extrap_symbol != 'constant':
                if previous_key and (last_key.get('CurveMode')  == 'linear' or last_key.get('CurveOrder')  == 'linear'):
                    # For linear keys the tangent actually does not do anything, so we need to look a frame
                    # ahead and compute the increment
                    increment = (last_key.get('Value') - previous_key.get('Value')) / (last_key.get('Frame') - previous_key.get('Frame'))
                    return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), increment)
                else:
                    last_key_right_slope = last_key.get('LHandle_dY') / last_key.get('LHandle_dX')
                    return FlameChannellInterpolator.LinearExtrapolate(last_key.get('Frame'), last_key.get('Value'), last_key_right_slope)
            else:
                return FlameChannellInterpolator.ConstantExtrapolate(last_key.get('Frame'), last_key.get('Value'))

        def key_pair_to_segment(self, key, next_key):
            key_left_tangent = key.get('LHandle_dY') / key.get('LHandle_dX') * -1
            key_right_tangent = key.get('RHandle_dY') / key.get('RHandle_dX')
            next_key_left_tangent = next_key.get('LHandle_dY') / next_key.get('LHandle_dX') * -1
            next_key_right_tangent = next_key.get('RHandle_dY') / next_key.get('RHandle_dX')

            if key.get('CurveMode') == 'bezier':
                return FlameChannellInterpolator.BezierSegment(
                    key.get('Frame'), 
                    next_key.get('Frame'),
                    key.get('Value'), 
                    next_key.get('Value'),
                    float(key.get('Frame')) + float(key.get('RHandle_dX')), 
                    float(key.get('Value')) + float(key.get('RHandle_dY')),
                    float(next_key.get('Frame')) + float(next_key.get('LHandle_dX')),
                    float(next_key.get('Value')) + float(next_key.get('LHandle_dY'))
                    )
            
            elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'cubic'):
                return FlameChannellInterpolator.HermiteSegment(
                    key.get('Frame'), 
                    next_key.get('Frame'), 
                    key.get('Value'), 
                    next_key.get('Value'),
                    key_right_tangent, 
                    next_key_left_tangent
                    )
            elif (key.get('CurveMode') in ['natural', 'hermite']) and (key.get('CurveOrder') == 'quartic'):
                return FlameChannellInterpolator.HermiteSegmentQuartic5x5(
                    key.get('Frame'), 
                    next_key.get('Frame'), 
                    key.get('Value'), 
                    next_key.get('Value'),
                    key_right_tangent, 
                    next_key_left_tangent
                    )
            elif key.get('CurveMode') == 'constant':
                return FlameChannellInterpolator.ConstantSegment(
                    key.get('Frame'), 
                    next_key.get('Frame'), 
                    key.get('Value')
                    )
            else:  # Linear and safe
                return FlameChannellInterpolator.LinearSegment(
                    key.get('Frame'), 
                    next_key.get('Frame'), 
                    key.get('Value'), 
                    next_key.get('Value')
                    )


    tw_setup_xml = ET.fromstring(tw_setup_string)
    tw_setup = dictify(tw_setup_xml)
    # pprint (tw_setup)

    # start = int(tw_setup['Setup']['Base'][0]['Range'][0]['Start'])
    # end = int(tw_setup['Setup']['Base'][0]['Range'][0]['End'])
    # TW_Timing_size = int(tw_setup['Setup']['State'][0]['TW_Timing'][0]['Channel'][0]['Size'][0]['_text'])

    TW_SpeedTiming_size = tw_setup['Setup']['State'][0]['TW_SpeedTiming'][0]['Channel'][0]['Size']
    TW_RetimerMode = tw_setup['Setup']['State'][0]['TW_RetimerMode']

    frame_value_map = {}

    if TW_RetimerMode == 1:
        # 'Timing' channel is enough
        tw_channel = 'TW_Timing'
        channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
        if 'KFrames' in channel.keys():
            channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Index'])}
        interpolator = FlameChannellInterpolator(channel)
        for frame_number in range (start_frame, end_frame+1):
            frame_value_map[frame_number] = interpolator.sample_at(frame_number)
        return frame_value_map

    else:
        # speed - based timewarp somehow
        # solves frames in a different way
        # depending on a segment mode

        tw_channel = 'TW_Speed'
        channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
        if 'KFrames' in channel.keys():
            channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Index'])}
        speed_channel = dict(channel)
        tw_channel = 'TW_SpeedTiming'
        channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
        if 'KFrames' in channel.keys():
            channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Index'])}
        speed_timing_channel = dict(channel)

        # pprint (speed_timing_channel)
        # pprint (speed_channel)


        speed_interpolator = FlameChannellInterpolator(speed_channel)
        timing_interpolator = FlameChannellInterpolator(speed_timing_channel)

        for frame_number in range (start_frame, end_frame+1):
            frame_value_map[frame_number] = round(timing_interpolator.sample_at(frame_number), 2)
        
        '''
        for frame_number in range (start_frame, end_frame+1):
            if speed_interpolator.segment_mode(frame_number) in ['hermite', 'linear']:
                speed_segment = speed_interpolator.get_segment(frame_number)
                start_frame_value = timing_interpolator.sample_at(speed_segment.start_frame)
                start_speed_value = speed_interpolator.sample_at(speed_segment.start_frame)
                previous_frame_value = start_frame_value
                previous_speed_value = start_speed_value
                for segment_frame_number in range (
                    int(round(speed_segment.start_frame)) + 1, 
                    int(round(speed_segment.end_frame)) 
                    ):
                    speed_value = speed_interpolator.sample_at(segment_frame_number)
                    speed_delta = previous_speed_value - speed_value
                    
                    new_frame_value = previous_frame_value + (1 / (100 / speed_value)) + (1 / (100 / speed_delta)) / 2
                    if segment_frame_number == frame_number:
                        frame_value_map[frame_number] = round(new_frame_value, 2)
                        break
                    previous_frame_value = new_frame_value
                    previous_speed_value = speed_value
        '''
        # print ('frame: %s, speed mode: %s, timing_mode: %s' % (frame_number, speed_interpolator.segment_mode(frame_number), timing_interpolator.segment_mode(frame_number)))
        
        pprint (frame_value_map)
        # pprint (timing_map)

        sys.exit()


        timing_map = {}
        tw_channel = 'TW_SpeedTiming'
        channel = tw_setup['Setup']['State'][0][tw_channel][0]['Channel'][0]
        if 'KFrames' in channel.keys():
            channel['KFrames'] = {x['Frame']: x for x in sorted(channel['KFrames'][0]['Key'], key=lambda d: d['Index'])}
            kframes = channel.get('KFrames')
            index_frames = list(kframes.keys())
        else:
            kframes = {}
            index_frames = []
        
        if index_frames[0] > start_frame:
            # we need to extrapolate backwards from the first 
            # keyframe in SpeedTiming channel

            previous_frame_value = kframes[index_frames[0]]['Value']
            previous_frame_value = float(format(previous_frame_value,".2f"))
            timing_map[index_frames[0]] = previous_frame_value
            for frame_number in range(index_frames[0] - 1, start_frame - 1, -1):
                new_value = previous_frame_value - (1 / (100 / frame_value_map[frame_number]))
                new_value = float(format(new_value,".2f"))
                timing_map[frame_number] = new_value
                previous_frame_value = new_value
            
            pprint (timing_map)

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
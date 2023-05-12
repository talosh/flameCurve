from os.path import dirname, join

from .segments import ConstantFunction
from .segments import pick_prepolation, pick_extrapolation
from .segments import LinearPrepolate, LinearExtrapolate, ConstantPrepolate, ConstantExtrapolate
from .segments import BezierSegment

class Interpolator:
    NEG_INF = float('-inf')
    POS_INF = float('inf')

    def __init__(self, channel):
        self.segments = []
        self.extrap = channel.extrapolation

        if channel.length == 0:
            self.segments = [ConstantFunction(channel.base_value)]
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
        if first_f == self.NEG_INF:
            return 1
        return first_f

    def last_defined_frame(self):
        last_f = self.segments[-1].start_frame
        if last_f == self.POS_INF:
            return 100
        return last_f

    def create_segments_from_channel(self, channel):
        # First the prepolating segment
        segments = [pick_prepolation(channel.extrapolation, channel[0], channel[1])]

        # Then all the intermediate segments, one segment between each pair of keys
        for index, key in enumerate(channel[:-1]):
            segments.append(self.key_pair_to_segment(key, channel[index + 1]))

        # and the extrapolator
        segments.append(pick_extrapolation(channel.extrapolation, channel[-2], channel[-1]))
        return segments

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

    def sample_from_segments(self, at_frame):
        for segment in self.segments:
            if segment.defines(at_frame):
                return segment.value_at(at_frame)
        raise ValueError(f'No segment on this curve that can interpolate the value at {at_frame}')

    def key_pair_to_segment(key, next_key):
        if key.interpolation == 'bezier':
            return BezierSegment(key.frame, next_key.frame,
                                key.value, next_key.value,
                                key.r_handle_x, key.r_handle_y,
                                next_key.l_handle_x, next_key.l_handle_y)
        elif key.interpolation in ['natural', 'hermite']:
            print("We're in Natural:Hermite")
            print("key.frame: ", key.frame)
            print("next_key.frame: ", next_key.frame)
            print("key.value: ", key.value)
            print("next_key.value: ", next_key.value)
            print("key.right_slope: ", key.right_slope)
            print("next_key.left_slope: ", next_key.left_slope)
            return HermiteSegment(key.frame, next_key.frame, key.value, next_key.value,
                                key.right_slope, next_key.left_slope)
        elif key.interpolation == 'constant':
            return ConstantSegment(key.frame, next_key.frame, key.value)
        else:  # Linear and safe
            return LinearSegment(key.frame, next_key.frame, key.value, next_key.value)

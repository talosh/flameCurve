from typing import IO, List, Tuple
import re
from io import StringIO

class ChannelNotFoundError(RuntimeError):
    pass

class NoKeyframesError(RuntimeError):
    pass

class Extractor:
    DEFAULT_CHANNEL_TO_EXTRACT = "Timing/Timing"
    DEFAULT_START_FRAME = 1
    DEFAULTS = {
        'destination': StringIO(),
        'start_frame': None,
        'end_frame': None,
        'channel': DEFAULT_CHANNEL_TO_EXTRACT,
        'on_curve_limits': False
    }

    SETUP_END_FRAME_PATTERN = re.compile(r'(MaxFrames|Frames)\s+(\d+)')
    SETUP_START_FRAME_PATTERN = re.compile(r'MinFrame\s+(\d+)')

    # Pass the path to Flame setup here and you will get the animation curve on the object passed in
    # the :destionation option (defaults to STDOUT). The following options are accepted:
    #
    #  :destination - The object to write the output to, anything that responds to shovel (<<) will do
    #  :start_frame - From which frame the curve should be baked. Will default to the first keyframe of the curve
    #  :end_frame - Upto which frame to bake. Will default to the last keyframe of the curve
    #  :channel - Name of the channel to extract from the setup. Defaults to "Timing/Timing" (timewarp frame)
    #
    # Note that start_frame and end_frame will be converted to integers.
    # The output will look like this:
    #
    #   1  123.456
    #   2  124.567

    @classmethod
    def extract(cls, path: str, options=None):
        if options is None:
            options = {}
        extractor = cls()
        extractor._extract(path, options)
        return extractor.DEFAULTS['destination'].getvalue()

    def _extract(self, path: str, options):
        options = self.DEFAULTS.copy()
        options.update(options)
        with open(path) as f:
            # Then parse
            channels = FlameChannelParser.parse(f)
            selected_channel = self.find_channel_in(channels, options['channel'])
            interpolator = FlameChannelParser.Interpolator(selected_channel)

            # Configure the range
            self.configure_start_and_end_frame(f, options, interpolator)

            # And finally...
            self.write_channel(interpolator, options['destination'], options['start_frame'], options['end_frame'])

    def start_and_end_frame_from_curve_length(self, interp):
        s, e = interp.first_defined_frame, interp.last_defined_frame
        if (not s) or (not e):
            raise NoKeyframesError(
                "This channel probably has no animation so there is no way to automatically tell how many keyframes it has. Please set the start and end frame explicitly."
            )
        elif s == e:
            raise NoKeyframesError(
                f"This channel has only one keyframe at frame {s} and baking it makes no sense."
            )
        return int(s), int(e)

    def configure_start_and_end_frame(self, f: IO[str], options, interpolator):
        # If the settings specify last and first frame...
        if options['on_curve_limits']:
            options['start_frame'], options['end_frame'] = self.start_and_end_frame_from_curve_length(interpolator)
        else:  # Detect from the setup itself (the default)
            # First try to detect start and end frames from the known flags
            f.seek(0)
            detected_start, detected_end = self.detect_start_and_end_frame_in_io(f)

            options['start_frame'] = options['start_frame'] or detected_start or self.DEFAULT_START_FRAME
            options['end_frame'] = options['end_frame'] or detected_end

            # If the setup does not contain that information retry with curve limits
            if (not options['start_frame']) or (not options['end_frame']):
                options['on_curve_limits'] = True
                self.configure_start_and_end_frame(f, options, interpolator)

    def detect_start_and_end_frame_in_io(self, io: IO[str]) -> Tuple[int, int]:
        cur_offset, s, e = io.tell(), None, None
        io.seek(0)
        for line in io:
            if (elements := self.SETUP_START_FRAME_PATTERN.findall(line)):
                s = int(elements[-1])
            elif (elements := self.SETUP_END_FRAME_PATTERN.findall(line)):
                e = int(elements[-1])
                return s, e
        io.seek(cur_offset)

    def compose_channel_not_found_message(for_channel, other_channels):
        message = f"Channel #{for_channel} not found in this setup (set the channel with the :channel option). Found other channels though:\n"
        message += "\n".join(f"\t{c.path}" for c in other_channels)
        return message
        
    def find_channel_in(channels, channel_path):
        selected_channel = next((c for c in channels if channel_path == c.path), None)
        if not selected_channel:
            raise ChannelNotFoundError(compose_channel_not_found_message(channel_path, channels))
        return selected_channel
        
    def write_channel(interpolator, to_io, from_frame_i, to_frame_i):
        if (to_frame_i - from_frame_i) == 1:
            print("WARNING: You are extracting one animation frame. Check the length of your setup, or set the range manually", file=sys.stderr)
        
        for frame in range(from_frame_i, to_frame_i+1):
            write_frame(to_io, frame, interpolator.sample_at(frame))
            
    def write_frame(to_io, frame, value):
        line = f"{frame}\t{value:.5f}\n"
        to_io.write(line)
                
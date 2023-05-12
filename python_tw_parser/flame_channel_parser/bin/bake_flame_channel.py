#!/usr/bin/env python3

import sys
import argparse
from os.path import dirname, abspath, exists
from lib.flame_channel_parser import Extractor

def main():
    parser = argparse.ArgumentParser(
        description='Extract an animation channel from a Flame setup')
    parser.add_argument('setup_path', metavar='SETUP_PATH', 
                        help='path to the Flame setup file')
    parser.add_argument('-c', '--channel', dest='channel_name', 
                        required=True, metavar='CHANNEL_NAME',
                        help='select the channel to bake (for example in Timewarp setups the useful one is Timing/Timing)')
    parser.add_argument('-s', '--startframe', dest='start_frame', type=int, 
                        metavar='FRAME', help='bake the curve from this specific frame onwards (defaults to the first keyframe in the setup)')
    parser.add_argument('-e', '--endframe', dest='end_frame', type=int,
                        metavar='FRAME', help='bake the curve up to this specific frame (defaults to the last keyframe in the setup)')
    parser.add_argument('-k', '--keyframed-range-only', dest='on_curve_limits', 
                        action='store_true', help='bake the curve from the first keyframe to the last only (overrides --startframe and --endframe)')
    parser.add_argument('-f', '--to-file', dest='filename', metavar='FILENAME', 
                        help='write the curve to a file at this path instead of printing it to STDOUT')
    args = parser.parse_args()

    setup_path = args.setup_path
    channel_name = args.channel_name
    start_frame = args.start_frame
    end_frame = args.end_frame
    on_curve_limits = args.on_curve_limits
    filename = args.filename

    if not exists(setup_path):
        sys.exit("File does not exist.")
    
    destination = open(filename, "wb") if filename else sys.stdout
    options = {
        'channel': channel_name,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'on_curve_limits': on_curve_limits,
        'destination': destination
    }
    
    Extractor.extract(setup_path, options)
    destination.close() if filename else None

if __name__ == '__main__':
    main()

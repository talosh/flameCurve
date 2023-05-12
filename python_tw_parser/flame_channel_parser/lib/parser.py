import re

class Parser:
    
    # Here you can assign a logger proc or a lambda that will be call'ed with progress reports
    def __init__(self):
        self.logger_proc = None
    
    @property
    def logger_proc(self):
        return self._logger_proc
    
    @logger_proc.setter
    def logger_proc(self, value):
        self._logger_proc = value
    
    # Parses the setup passed in the IO. If a block is given to the method it will yield Channel
    # objects one by one instead of accumulating them into an array (useful for big setups)
    def parse(self, io):
        self.do_logs = callable(self.logger_proc)
        
        channels = []
        node_name, node_type = None, None
        
        while True:
            line = io.readline()
            if not line:
                break
            match = re.match(r'Node (\w+)', line)
            if match:
                node_type = match.group(1)
            else:
                match = re.match(r'Name (\w+)', line)
                if match:
                    node_name = match.group(1)
                else:
                    match = re.match(r'Channel (.+)\n', line)
                    if match and self.channel_is_useful(match.group(1)):
                        self.log(f"Parsing channel {match.group(1)}")
                        channel = self.parse_channel(io, match.group(1), node_type, node_name)
                        if callable(self.logger_proc):
                            yield channel
                        else:
                            channels.append(channel)
        
        if callable(self.logger_proc):
            return
        else:
            return channels
    
    # This method will be called internally with information on items being processed.
    # The implementation just calls the logger_proc instance variable
    def log(self, message):
        if self.do_logs and callable(self.logger_proc):
            self.logger_proc(message)
    
    # Override this method to skip some channels, this will speedup
    # your code alot
    def channel_is_useful(self, channel_name):
        return True
    
    # Defines a number of regular expression matchers applied to the file as it is being parsed
    def key_matchers(self):
        return [
            # Previously:
            ['frame', float,  r'Frame ([\-\d\.]+)'],
            ['value', float,  r'Value ([\-\d\.]+)'],
            ['left_slope', float, r'LeftSlope ([\-\d\.]+)'],
            ['right_slope', float, r'RightSlope ([\-\d\.]+)'],
            ['interpolation', str, r'Interpolation (\w+)'],
            ['break_slope', str, r'BreakSlope (\w+)'],
            
            # 2012 intoroduces:
            ['r_handle_x', float, r'RHandleX ([\-\d\.]+)'],
            ['l_handle_x', float, r'LHandleX ([\-\d\.]+)'],
            ['r_handle_y', float, r'RHandleY ([\-\d\.]+)'],
            ['l_handle_y', float, r'LHandleY ([\-\d\.]+)'],
            ['curve_mode', str, r'CurveMode (\w+)'],
            ['curve_order', str, r'CurveOrder (\w+)'],
        ]
    
    BASE_VALUE_MATCHER = r'Value ([\-\d\.]+)'
    KF_COUNT_MATCHER = r'Size (\d+)'
    EXTRAP_MATCHER = r'Extrapolation (\w+)'
    CHANNEL_MATCHER = r'Channel (.+)\n'
    NODE_TYPE_MATCHER = re.compile(r'Node (\w+)')
    NODE_NAME_MATCHER = re.compile(r'Name (\w+)')
    LITERALS = ['linear', 'constant', 'natural', 'hermite', 'cubic', 'be
   
import re

class Parser:
    def __init__(self):
        self.logger_proc = None
        self.do_logs = False

    def parse(self, io):
        self.do_logs = callable(self.logger_proc)
        channels = []
        node_name, node_type = None, None
        while not io.eof():
            line = io.gets()
            if re.search(NODE_TYPE_MATCHER, line):
                node_type = re.search(NODE_TYPE_MATCHER, line).group(1)
            elif re.search(NODE_NAME_MATCHER, line):
                node_name = re.search(NODE_NAME_MATCHER, line).group(1)
            elif re.search(CHANNEL_MATCHER, line) and channel_is_useful(re.search(CHANNEL_MATCHER, line).group(1)):
                self.log(f"Parsing channel {re.search(CHANNEL_MATCHER, line).group(1)}")
                channel = parse_channel(io, re.search(CHANNEL_MATCHER, line).group(1), node_type, node_name)
                if callable(self.logger_proc):
                    yield channel
                else:
                    channels.append(channel)
        return channels if not callable(self.logger_proc) else None

    def log(self, message):
        if self.do_logs:
            self.logger_proc(message)

    def channel_is_useful(self, channel_name):
        return True

    def key_matchers(self):
        return [
            # Previously:
            ('frame', float, re.compile(r'Frame ([\-\d\.]+)')),
            ('value', float, re.compile(r'Value ([\-\d\.]+)')),
            ('left_slope', float, re.compile(r'LeftSlope ([\-\d\.]+)')),
            ('right_slope', float, re.compile(r'RightSlope ([\-\d\.]+)')),
            ('interpolation', str, re.compile(r'Interpolation (\w+)')),
            ('break_slope', str, re.compile(r'BreakSlope (\w+)')),
            # 2012 introduces:
            ('r_handle_x', float, re.compile(r'RHandleX ([\-\d\.]+)')),
            ('l_handle_x', float, re.compile(r'LHandleX ([\-\d\.]+)')),
            ('r_handle_y', float, re.compile(r'RHandleY ([\-\d\.]+)')),
            ('l_handle_y', float, re.compile(r'LHandleY ([\-\d\.]+)')),
            ('curve_mode', str, re.compile(r'CurveMode (\w+)')),
            ('curve_order', str, re.compile(r'CurveOrder (\w+)')),
        ]

    base_value_matcher = re.compile(r'Value ([\-\d\.]+)')
    keyframe_count_matcher = re.compile(r'Size (\d+)')
    BASE_VALUE_MATCHER = re.compile(r'Value ([\-\d\.]+)')
    KF_COUNT_MATCHER = re.compile(r'Size (\d+)')
    EXTRAP_MATCHER = re.compile(r'Extrapolation (\w+)')
    CHANNEL_MATCHER = re.compile(r'Channel (.+)\n')
    NODE_TYPE_MATCHER = re.compile(r'Node (\w+)')
    NODE_NAME_MATCHER = re.compile(r'Name (\w+)')
    LITERALS = ['linear', 'constant', 'natural', 'hermite', 'cubic', 'bezier', 'cycle', 'revcycle']


    def parse_channel(io, channel_name, node_type, node_name):
        c = Channel(channel_name, node_type, node_name)
        indent, end_mark = None, "ENDMARK"
        
        while True:
            line = io.readline()
            
            if not indent:
                match = re.search(r"^(\s+)", line)
                if match:
                    indent = match.group(1)
                    end_mark = f"{indent}End"
            
            if re.search(KF_COUNT_MATCHER, line):
                num_keyframes = int(re.search(KF_COUNT_MATCHER, line).group(1))
                for idx in range(num_keyframes):
                    log(f"Extracting keyframe {idx + 1} of {num_keyframes}")
                    c.push(extract_key_from(io))
            
            elif re.search(BASE_VALUE_MATCHER, line):
                c.base_value = float(re.search(BASE_VALUE_MATCHER, line).group(1))
            
            elif re.search(EXTRAP_MATCHER, line):
                c.extrapolation = symbolize_literal(re.search(EXTRAP_MATCHER, line).group(1))
            
            elif line.strip() == end_mark:
                break
        
        return c


    def extract_key_from(io):
        frame = None
        end_matcher = re.compile(r"^End$")
        key = Key()
        
        while True:
            line = io.readline()
            
            if re.search(end_matcher, line):
                return key
            
            else:
                for property, cast_method, pattern in key_matchers:
                    match = re.search(pattern, line)
                    if match:
                        v = symbolize_literal(getattr(match.group(1), cast_method)())
                        setattr(key, property, v)
        
        raise Exception("Did not detect any keyframes!")


    def symbolize_literal(v):
        return v.to_sym() if v in LITERALS else v

'''
Note: I assumed that log() is a logging function that prints messages to the console. 
If it's not defined, you should define it accordingly. Also, I assumed that 
Channel, Key, key_matchers, and LITERALS are defined elsewhere in the code.
'''
'''
Note that this code assumes that Curve and Tuple are defined elsewhere in the program. 
Also, Python's regular expression syntax is slightly different than Ruby's, 
so the regular expression patterns have been updated accordingly.
'''

class Comment:
    def __init__(self, text):
        self.text = text.strip()

    def is_tuple(self):
        return False

    def is_comment(self):
        return True

    def __str__(self):
        return f"# {self.text}"

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)


class Parser:
    COMMENT_PATTERN = r"^#(.+)$"
    CORRELATION_RECORD_PATTERN = r"^([-]?\d+)\t([-]?(\d+(\.\d*)?)|\.\d+)([eE][+-]?[0-9]+)?$"

    def parse(self, path_or_io):
        if isinstance(path_or_io, str):
            with open(path_or_io, "r") as f:
                return self.parse(f)

        line_counter = 0
        elements = []
        for line in path_or_io:
            line_counter += 1
            line = line.strip()
            if not line:
                continue
            if re.match(self.COMMENT_PATTERN, line):
                item = self.extract_comment(line)
            elif re.match(self.CORRELATION_RECORD_PATTERN, line):
                item = self.extract_tuple(line)
            else:
                raise ValueError(f"Malformed line {line!r} at line {line_counter}")
            elements.append(item)

        curve = Curve(elements)
        if hasattr(path_or_io, "name") and path_or_io.name:
            curve.filename = os.path.basename(path_or_io.name)
        return curve

    def extract_comment(self, line):
        comment_txt = re.search(self.COMMENT_PATTERN, line).group(1).strip()
        return Comment(comment_txt)

    def extract_tuple(self, line):
        slots = re.search(self.CORRELATION_RECORD_PATTERN, line).groups()
        frame, value, exponent = int(slots[0]), float(slots[1]), slots[4]
        if exponent:
            value *= 10 ** int(exponent)
        return Tuple(frame, value)

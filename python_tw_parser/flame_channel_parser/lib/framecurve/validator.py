class Validator:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.performed = False

    def any_errors(self):
        return bool(self.errors)

    def any_warnings(self):
        return bool(self.warnings)

    def parse_and_validate(self, path_or_io):
        try:
            self.validate(Parser().parse(path_or_io))
        except Malformed as e:
            self.errors.append(e.message)

    def validate(self, curve):
        self.errors.clear()
        self.warnings.clear()
        for method_name in filter(lambda m: re.match('^(verify|recommend)', m), dir(self)):
            getattr(self, method_name)(curve)
        self.performed = True

    def ok(self):
        return self.performed and not self.any_errors() and not self.any_warnings()

    def verify_at_least_one_line(self, curve):
        if not curve:
            self.errors.append("The framecurve did not contain any lines at all")

    def verify_at_least_one_tuple(self, curve):
        first_tuple = next((e for e in curve if e.tuple()), None)
        if not first_tuple:
            self.errors.append("The framecurve did not contain any frame correlation records")

    def verify_proper_sequencing(self, curve):
        tuples = [e for e in curve if e.tuple()]
        frame_numbers = [t.at for t in tuples]
        proper_sequence = sorted(frame_numbers)
        if frame_numbers != proper_sequence:
            self.errors.append("The frame sequencing is out of order " +
                                "(expected {} but got {}).".format(proper_sequence, frame_numbers) +
                                " The framecurve spec mandates that frames are recorded sequentially")

    def verify_no_linebreaks_in_comments(self, curve):
        for i, r in enumerate(curve):
            if r.comment() and ("\r" in r.text() or "\n" in r.text()):
                self.errors.append("The comment at line {} contains a line break".format(i + 1))

    def verify_non_negative_source_and_destination_frames(self, curve):
        for i, t in enumerate(curve):
            if not t.tuple():
                continue
            line_no = i + 1
            if t.at < 1:
                self.errors.append("The line {} had it's at_frame value ({}) below 1. The spec mandates at_frame >= 1.".format(line_no, t.at))
            elif t.value < 0:
                self.errors.append("The line {} had a use_frame_of_source value ({:.5f}) below 0. The spec mandates use_frame_of_source >= 0.".format(line_no, t.value))

    def verify_file_naming(self, curve):
        if not hasattr(curve, 'filename') or not curve.filename:
            return
        if not re.match(r'\.framecurve\.txt$', curve.filename):
            self.errors.append("The framecurve file has to have the .framecurve.txt double extension, but had {}".format(os.path.splitext(curve.filename)[1]))

    def verify_no_duplicate_records(self, curve):
        detected_dupes = []
        for t in curve:
            if not t.tuple() or t.at in detected_dupes:
                continue
            elements = [e for e in curve if e.tuple() and e.at == t.at]
            if len(elements) > 1:
                detected_dupes.append(t.at)
                self.errors.append("The framecurve contains the same frame ({}) twice or more ({} times)".format(t.at, len(elements)))

    def recommend_proper_preamble(self, curve):
        if not curve or not curve[0

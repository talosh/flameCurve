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
        self.warnings = []
        self.errors = []
        
        for method_name in dir(self):
            if method_name.startswith(('verify', 'recommend')):
                method = getattr(self, method_name)
                method(curve)
        
        self.performed = True
    
    def ok(self):
        return self.performed and not self.any_errors() and not self.any_warnings()
    
    def verify_at_least_one_line(self, curve):
        if not curve:
            self.errors.append('The framecurve did not contain any lines at all')
    
    def verify_at_least_one_tuple(self, curve):
        if not any(element.tuple_() for element in curve):
            self.errors.append('The framecurve did not contain any frame correlation records')
    
    def verify_proper_sequencing(self, curve):
        tuples = [element for element in curve if element.tuple_()]
        frame_numbers = [tuple_.at for tuple_ in tuples]
        proper_sequence = sorted(frame_numbers)
        
        if frame_numbers != proper_sequence:
            self.errors.append(f"The frame sequencing is out of order (expected {proper_sequence}, but got {frame_numbers}). The framecurve spec mandates that frames are recorded sequentially")
    
    def verify_no_linebreaks_in_comments(self, curve):
        for i, element in enumerate(curve, start=1):
            if element.comment_() and any(char in element.text for char in ('\r', '\n')):
                self.errors.append(f"The comment at line {i} contains a line break")
    
    def verify_non_negative_source_and_destination_frames(self, curve):
        for i, tuple_ in enumerate(curve, start=1):
            if tuple_.tuple_():
                if tuple_.at < 1:
                    self.errors.append(f"The line {i} had its at_frame value ({tuple_.at}) below 1. The spec mandates at_frame >= 1.")
                elif tuple_.value < 0:
                    self.errors.append(f"The line {i} had a use_frame_of_source value ({tuple_.value:.5f}) below 0. The spec mandates use_frame_of_source >= 0.")
    
    def verify_file_naming(self, curve):
        if hasattr(curve, 'filename') and curve.filename and not curve.filename.endswith('.framecurve.txt'):
            self.errors.append(f"The framecurve file has to have the .framecurve.txt double extension, but had {os.path.splitext(curve.filename)[-1]}")
    
    def verify_no_duplicate_records(self, curve):
        duplicates = {t.at for t in curve if t.tuple_() and curve.count(t) > 1}
        for duplicate in duplicates:
            self.errors.append(f"The framecurve contains the same frame ({duplicate}) twice or more ({curve.count(Element(t, 0))} times)")
    
    def recommend_proper_preamble(self, curve):
        if not (curve and curve[0].comment_() and 'framecurve.org/specification' in curve[0].text):
            self.warnings.append("It is recommended that a framecurve starts with a comment with the specification URL")
    
    def recommend_proper_column_headers(self, curve):
        line_two = curve[1] if len(curve) > 1 else None
        if not (line_two and line_two.comment and "at_frame\tuse_frame_of_source" in line_two.text):
            self.warnings.append("It is recommended for the second comment to provide a column header")

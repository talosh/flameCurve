from typing import List, Union
import os

class Validator:
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.performed: bool = False
    
    def any_errors(self) -> bool:
        return bool(self.errors)
    
    def any_warnings(self) -> bool:
        return bool(self.warnings)
    
    def parse_and_validate(self, path_or_io: Union[str, bytes, os.PathLike, int, object]) -> None:
        try:
            curve = Parser().parse(path_or_io)
            self.validate(curve)
        except Malformed as e:
            self.errors.append(str(e))
    
    def validate(self, curve) -> None:
        self.warnings.clear()
        self.errors.clear()
        for method_name in self.methods_matching(r'^(verify|recommend)'):
            getattr(self, method_name)(curve)
        self.performed = True
    
    def ok(self) -> bool:
        return self.performed and not self.any_errors() and not self.any_warnings()
    
    def methods_matching(self, pattern: str) -> List[str]:
        return [m for m in dir(self) if callable(getattr(self, m)) and re.match(pattern, m)]
    
    def verify_at_least_one_line(self, curve) -> None:
        if not curve:
            self.errors.append("The framecurve did not contain any lines at all")
    
    def verify_at_least_one_tuple(self, curve) -> None:
        first_tuple = next((e for e in curve if e.tuple()), None)
        if not first_tuple:
            self.errors.append("The framecurve did not contain any frame correlation records")
    
    def verify_proper_sequencing(self, curve) -> None:
        tuples = [e for e in curve if e.tuple()]
        frame_numbers = [t.at for t in tuples]
        proper_sequence = sorted(frame_numbers)
        if frame_numbers != proper_sequence:
            self.errors.append(f"The frame sequencing is out of order (expected {proper_sequence} but got {frame_numbers}). The framecurve spec mandates that frames are recorded sequentially")
    
    def verify_no_linebreaks_in_comments(self, curve) -> None:
        for i, r in enumerate(curve):
            if r.comment() and ("\r" in r.text or "\n" in r.text):
                self.errors.append(f"The comment at line {i+1} contains a line break")
    
    def verify_non_negative_source_and_destination_frames(self, curve) -> None:
        for i, t in enumerate(curve):
            if not t.tuple():
                continue
            line_no = i + 1
            if t.at < 1:
                self.errors.append(f"The line {line_no} had its at_frame value ({t.at}) below 1. The spec mandates at_frame >= 1.")
            elif t.value < 0:
                self.errors.append(f"The line {line_no} had a use_frame_of_source value ({t.value:.5f}) below 0. The spec mandates use_frame_of_source >= 0.")
    
    def verify_file_naming(self, curve) -> None:
        if hasattr(curve, "filename") and curve.filename:
            if not curve.filename.endswith(".framecurve.txt"):
                self.errors.append(f"The framecurve file has to have the .framecurve.txt double extension, but had {os.path.splitext(curve.filename)[1]}")
    
    def verify_no_duplicate_records(self, curve) -> None:
        detected_dupes = []
        for t in curve:
            if not t.tuple() or t.at in detected

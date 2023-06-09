'''
Note that I removed the private keyword, 
as Python doesn't have true private methods. 
Instead, I made the methods that should be considered 
private as static methods using the @staticmethod decorator.
'''

class FramecurveSerializer:
    
    # Serialize the passed curve into io. Will use the materialized curve version.
    # Will write the file with CRLF linebreaks instead of LF.
    # Also, if the passed Curve object does not contain a preamble (URL and column headers)
    # they will be added automatically
    @staticmethod
    def serialize(io, curve):
        if not FramecurveSerializer.curve_has_preamble(curve):
            FramecurveSerializer.write_preamble(io)
        for record in curve:
            io.write(f"{record}\r\n")

    # Serialize the passed curve into io and raise an exception
    @staticmethod
    def validate_and_serialize(io, curve):
        v = FramecurveValidator()
        v.validate(curve)
        if v.any_errors():
            raise FramecurveMalformed(f"Will not serialize a malformed curve: {', '.join(v.errors)}")
        FramecurveSerializer.serialize(io, curve)

    @staticmethod
    def write_preamble(io):
        io.write("# http://framecurve.org/specification-v1\n")
        io.write("# at_frame\tuse_frame_of_source\n")

    @staticmethod
    def curve_has_preamble(curve):
        if not curve:
            return False
        first_comment, second_comment = curve[0], curve[-1]
        if not first_comment or not second_comment:
            return False
        if not (first_comment.comment() and second_comment.comment()):
            return False
        if "http://framecurve.org" not in first_comment.text or "at_frame\tuse_frame_of_source" not in second_comment.text:
            return False
        return True

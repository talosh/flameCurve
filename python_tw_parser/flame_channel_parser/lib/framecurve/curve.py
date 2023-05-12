# Represents a curve file with comments and frame correlation records
class Curve:
    def __init__(self, *elements):
        self.filename = None
        self.elements = []
        for e in elements:
            if isinstance(e, list):
                self.elements.extend(e)
            else:
                self.elements.append(e)
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, index):
        return self.elements[index]
    
    def __iter__(self):
        return iter(self.elements)
    
    def __contains__(self, item):
        return item in self.elements
    
    def empty(self):
        return not bool(self.elements)
    
    def comment(self, text):
        self.elements.append(Comment(text.strip()))
    
    def tuple(self, at, value):
        t = Tuple(int(at), float(value))
        # Validate for sequencing
        if self.any_tuples():
            last_frame = self.only_tuples()[-1].at
            if t.at <= last_frame:
                raise Malformed(f"Cannot add a frame that comes before or at the same frame as the previous one ({t.at} after {last_frame})")
        self.elements.append(t)
    
    def each(self, callback):
        for e in self.elements:
            callback(e)
    
    def each_comment(self, callback):
        for e in self.elements:
            if isinstance(e, Comment):
                callback(e)
    
    def each_tuple(self, callback):
        for e in self.elements:
            if isinstance(e, Tuple):
                callback(e)
    
    def only_tuples(self):
        return [e for e in self.elements if isinstance(e, Tuple)]
    
    def any_tuples(self):
        return any(isinstance(e, Tuple) for e in self.elements)
    
    def to_materialized_curve(self):
        c = Curve()
        c.comment("http://framecurve.org/specification-v1")
        c.comment("at_frame\tuse_frame_of_source")
        for t in self.each_defined_tuple():
            c.tuple(t.at, t.value)
        return c
    
    def each_defined_tuple(self):
        tuples = self.only_tuples()
        for i in range(len(tuples)):
            tuple_ = tuples[i]
            next_tuple = tuples[i + 1] if i + 1 < len(tuples) else None
            if next_tuple is None:
                yield tuple_
            else: # Apply linear interpolation
                dt = next_tuple.at - tuple_.at
                if dt == 1:
                    yield tuple_
                else:
                    dy = next_tuple.value - tuple_.value
                    delta = dy / dt
                    for increment in range(dt):
                        value_inc = delta * increment
                        yield Tuple(tuple_.at + increment, tuple_.value + value_inc)
    

class Comment:
    def __init__(self, text):
        self.text = text
    
    def __repr__(self):
        return f"Comment('{self.text}')"
    
    def comment(self):
        return self.text


class Tuple:
    def __init__(self, at, value):
        self.at = at
        self.value = value
    
    def __repr__(self):
        return f"Tuple({self.at}, {self.value})"
    
    def tuple(self):
        return (self.at, self.value)


class Malformed(Exception):
    pass

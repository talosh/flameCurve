'''Note that I have also included the Comment class definition, which was not 
included in the original code but seems to be used in the Curve class. 
I have also added a definition for the Malformed exception that is 
raised in the tuple method, since
'''

class Framecurve:
    
    class Tuple:
        def __init__(self, at, value):
            self.at = at
            self.value = value
        
        def tuple(self):
            return True
        
        def comment(self):
            return False
        
        def __str__(self):
            return "%d\t%.5f" % (self.at, self.value)
        
        def __lt__(self, other):
            return str(self) < str(other)
    
    class Comment:
        def __init__(self, text):
            self.text = text
        
        def tuple(self):
            return False
        
        def comment(self):
            return True
        
        def __str__(self):
            return "# %s" % self.text
    
    class Curve:
        def __init__(self, *elements):
            self.elements = []
            for e in elements:
                self.elements.extend(e)
        
        @property
        def filename(self):
            return self._filename
        
        @filename.setter
        def filename(self, filename):
            self._filename = filename
        
        def each_tuple(self):
            for e in self.elements:
                if e.tuple():
                    yield e
        
        def only_tuples(self):
            return [e for e in self.elements if e.tuple()]
        
        def __iter__(self):
            for e in self.elements:
                yield e
        
        def each_comment(self):
            for e in self.elements:
                if e.comment():
                    yield e
        
        def comment(self, text):
            self.elements.append(Framecurve.Comment(text.strip()))
        
        def tuple(self, at, value):
            t = Framecurve.Tuple(int(at), float(value))
            # Validate for sequencing
            if self.any_tuples():
                last_frame = self.only_tuples()[-1].at
                if t.at <= last_frame:
                    raise Framecurve.Malformed("Cannot add a frame that comes before or at the same frame as the previous one (%d after %d)" % (t.at, last_frame))
            self.elements.append(t)
        
        def __len__(self):
            return len(self.elements)
        
        def empty(self):
            return len(self.elements) == 0
        
        def __getitem__(self, index):
            return self.elements[index]
        
        def any_tuples(self):
            return any([e.tuple() for e in self.elements])
        
        def to_materialized_curve(self):
            c = Framecurve.Curve()
            c.comment("http://framecurve.org/specification-v1")
            c.comment("at_frame\tuse_frame_of_source")
            for t in self.each_defined_tuple():
                c.tuple(t.at, t.value)
            return c
        
        def each_defined_tuple(self):
            tuples = self.only_tuples()
            for i in range(len(tuples)):
                tuple_ = tuples[i]
                next_tuple = tuples[i+1] if i+1 < len(tuples) else None
                if next_tuple is None:
                    yield tuple_
                else:  # Apply linear interpolation
                    dt = next_tuple.at - tuple_.at
                    if dt == 1:
                        yield tuple_
                    else:
                        dy = next_tuple.value - tuple_.value
                        delta = dy / dt
                        for increment in range(dt):
                            value_inc = delta * increment
                            yield Framecurve.Tuple(tuple_.at + increment, tuple_.value + value_inc)

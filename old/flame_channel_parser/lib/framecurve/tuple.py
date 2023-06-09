class Tuple:
    def __init__(self, at, value):
        self.at = at
        self.value = value

    def __lt__(self, other):
        return self.to_s() < other.to_s()

    def __eq__(self, other):
        return self.to_s() == other.to_s()

    def tuple(self):
        return True

    def comment(self):
        return False

    def to_s(self):
        return f"{self.at}\t{self.value:.5f}"

'''
Note that Python does not have an explicit Comparable module as Ruby does, 
but we can implement the comparison functions __lt__ (less than) and __eq__ (equal to) 
to achieve the same functionality. 
Also, Python does not have separate methods to check if an object 
is a tuple or a comment, so we create instance methods tuple and comment 
to return True or False respectively.
'''
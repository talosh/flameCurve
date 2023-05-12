'''
Note that in Python, we don't need to explicitly include the Comparable module, 
since we can implement the comparison operators (__eq__, __lt__, __gt__, etc.) 
directly in the class definition. Also note that we've renamed the class 
to Comment instead of Framecurve::Comment, since Python 
doesn't use the :: syntax for namespace separation.
'''

class Comment:
    def __init__(self, text):
        self.text = text

    def tuple(self):
        return False

    def comment(self):
        return True

    def __str__(self):
        return "# " + str(self.text).replace('\r\n', '')

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

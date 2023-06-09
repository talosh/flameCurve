import numpy as np

class ConstantSegment:
    NEG_INF = (-1.0/0.0)
    POS_INF = (1.0/0.0)

    def __init__(self, from_frame, to_frame, value):
        self.start_frame = from_frame
        self.end_frame = to_frame
        self.v1 = value

    def defines(self, frame):
        return (frame < self.end_frame) and (frame >= self.start_frame)

    def value_at(self, frame):
        return self.v1

class LinearSegment(ConstantSegment):

    def __init__(self, from_frame, to_frame, value1, value2):
        self.vint = (value2 - value1)
        super().__init__(from_frame, to_frame, value1)

    def value_at(self, frame):
        on_t_interval = (frame - self.start_frame) / (self.end_frame - self.start_frame)
        return self.v1 + (on_t_interval * self.vint)

class HermiteSegment(LinearSegment):
    HERMATRIX = np.array([
        [2,  -2,  1,  1],
        [-3, 3,   -2, -1],
        [0,   0,  1,  0],
        [1,   0,  0,  0]
    ]).T

    def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
        self.start_frame, self.end_frame = from_frame, to_frame
        frame_interval = (self.end_frame - self.start_frame)

        # Default tangents in flame are 0, so when we do None.to_f this is what we will get
        # CC = {P1, P2, T1, T2}
        p1, p2, t1, t2 = value1, value2, tangent1.to_f * frame_interval, tangent2.to_f * frame_interval
        self.hermite = np.array([p1, p2, t1, t2])
        self.basis = np.dot(self.HERMATRIX, self.hermite)

    def value_at(self, frame):
        if frame == self.start_frame:
            return self.hermite[0]

        # Get the 0 < T < 1 interval we will interpolate on
        # Q[frame_] = P[ ( frame - 149 ) / (time_to - time_from)]
        t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

        # S[s_] = {s^3, s^2, s^1, s^0}
        multipliers_vec = np.array([t ** 3, t ** 2, t ** 1, t ** 0])

        # P[s_] = S[s].h.CC
        interpolated_scalar = np.dot(self.basis, multipliers_vec)
        return interpolated_scalar

'''
Note that Struct is not available in Python, 
so BezierSegment.Pt is implemented as a separate class. 
Also, the Ruby clamp method has been implemented as an 
instance method in the Python code.
'''

class BezierSegment(LinearSegment):
    class Pt:
        def __init__(self, x, y, tanx, tany):
            self.x = x
            self.y = y
            self.tanx = tanx
            self.tany = tany
    
    def __init__(self, x1, x2, y1, y2, t1x, t1y, t2x, t2y):
        super().__init__(x1, x2, y1, y2)
        self.a = self.Pt(x1, y1, t1x, t1y)
        self.b = self.Pt(x2, y2, t2x, t2y)

    def value_at(self, frame):
        if frame == self.start_frame:
            return self.a.y
        
        t = self.approximate_t(frame, self.a.x, self.a.tanx, self.b.tanx, self.b.x)
        vy = self.bezier(t, self.a.y, self.a.tany, self.b.tany, self.b.y)
        return vy
    
    def bezier(self, t, a, b, c, d):
        return a + (a*(-3) + b*3)*(t) + (a*3 - b*6 + c*3)*(t**2) + (-a + b*3 - c*3 + d)*(t**3)
    
    def clamp(self, value):
        if value < 0:
            return 0.0
        elif value > 1:
            return 1.0
        else:
            return value
    
    APPROXIMATION_EPSILON = 1.0e-09
    VERYSMALL = 1.0e-20
    MAXIMUM_ITERATIONS = 100
    
    def approximate_t(self, atX, p0x, c0x, c1x, p1x):
        if atX - p0x < self.VERYSMALL:
            return 0.0
        elif p1x - atX < self.VERYSMALL:
            return 1.0

        u, v = 0.0, 1.0
        
        for i in range(self.MAXIMUM_ITERATIONS):
            a = (p0x + c0x) / 2.0
            b = (c0x + c1x) / 2.0
            c = (c1x + p1x) / 2.0
            d = (a + b) / 2.0
            e = (b + c) / 2.0
            f = (d + e) / 2.0
            
            if abs(f - atX) < self.APPROXIMATION_EPSILON:
                return self.clamp((u + v) * 0.5)
            
            if f < atX:
                p0x = f
                c0x = e
                c1x = c
                u = (u + v) / 2.0
            else:
                c0x = a
                c1x = d
                p1x = f
                v = (u + v) / 2.0
        
        return self.clamp((u + v) / 2.0)
    
'''
Note that in Python, True and False are keywords, 
so the defines method in ConstantFunction should return True instead of true. 
Also, since Python does not have symbols like Ruby, we use constants 
defined in the ConstantSegment class instead. Finally, we use the super() 
function to call the constructor and methods of the parent class, 
instead of explicitly specifying the name of the parent class like in Ruby.
'''

class ConstantPrepolate(LinearSegment):
    def __init__(self, upto_frame, base_value):
        super().__init__(NEG_INF, upto_frame, base_value)

    def value_at(self, frame):
        return self.v1

class LinearPrepolate(LinearSegment):
    def __init__(self, upto_frame, base_value, tangent):
        self.tangent = float(tangent)
        super().__init__(NEG_INF, upto_frame, base_value)

    def value_at(self, frame):
        frame_diff = (frame - self.end_frame)
        return self.v1 + (self.tangent * frame_diff)

class ConstantExtrapolate(LinearSegment):
    def __init__(self, from_frame, base_value):
        super().__init__(from_frame, POS_INF, base_value)

    def value_at(self, frame):
        return self.v1

class LinearExtrapolate(ConstantExtrapolate):
    def __init__(self, from_frame, base_value, tangent):
        self.tangent = float(tangent)
        super().__init__(from_frame, base_value)

    def value_at(self, frame):
        frame_diff = (frame - self.start_frame)
        return self.v1 + (self.tangent * frame_diff)

class ConstantFunction(ConstantSegment):
    def __init__(self, value):
        super().__init__(NEG_INF, POS_INF, value)

    def defines(self, frame):
        return True

    def value_at(self, frame):
        return self.v1

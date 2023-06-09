class Key:
    def __init__(self):
        self.frame = None
        self.value = None
        self.interpolation = None
        self.curve_order = None
        self.curve_mode = None
        self.left_slope = None
        self.right_slope = None
        self.break_slope = False
        self.l_handle_x = None
        self.l_handle_y = None
        self.r_handle_x = None
        self.r_handle_y = None

    def value(self):
        return float(self.value) if self.value else 0.0

    def frame(self):
        return float(self.frame) if self.frame else 1.0

    def left_slope(self):
        if self.break_slope:
            if self.has_2012_tangents():
                dy = self.value - self.l_handle_y
                print("Left Slope DY:", dy)
                dx = float(self.l_handle_x) - self.frame
                return (dy / dx) * -1
            else:
                return float(self.left_slope) if self.left_slope else None
        else:
            return self.right_slope()

    def right_slope(self):
        if self.has_2012_tangents():
            print("@Frame:", self.frame)
            dy = self.value - self.r_handle_y
            print("Right Slope DY:", dy)
            dx = self.frame - float(self.r_handle_x)
            print("Right Slope DX:", dx)
            print("Right Slope Value:", dy / dx)
            return dy / dx
        else:
            return float(self.right_slope) if self.right_slope else None

    def has_2012_tangents(self):
        if hasattr(self, 'has_tangents'):
            return self.has_tangents
        else:
            self.has_tangents = bool(self.l_handle_x and self.l_handle_y)
            return self.has_tangents

    def interpolation(self):
        if not self.has_2012_tangents():
            return self.interpolation or 'constant'
        else:
            if str(self.curve_order) == 'constant':
                return 'constant'
            elif str(self.curve_order) == 'cubic':
                if str(self.curve_mode) == 'hermite' or str(self.curve_mode) == 'natural':
                    return 'hermite'
                elif str(self.curve_mode) == 'bezier':
                    return 'bezier'
            elif str(self.curve_order) == 'linear':
                return 'linear'
            raise Exception(f"Cannot determine interpolation for {self}")

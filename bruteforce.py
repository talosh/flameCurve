import sys
import numpy as np
import multiprocessing
from pprint import pprint

correct_values = {1: 1.0,
 2: 2,
 3: 3,
 4: 3.98,
 5: 4.96,
 6: 5.93,
 7: 6.88,
 8: 7.81,
 9: 8.72,
 10: 9.61,
 11: 10.48,
 12: 11.32,
 13: 12.14,
 14: 12.93,
 15: 13.69,
 16: 14.43,
 17: 15.14,
 18: 15.82,
 19: 16.48,
 20: 17.11,
 21: 17.72,
 22: 18.31,
 23: 18.88,
 24: 19.43,
 25: 19.96,
 26: 20.49,
 27: 21,
 28: 21.5}

class HermiteSegmentQuartic():
    def __init__(self, from_frame, to_frame, value1, value2, tangent1, tangent2):
        self.start_frame, self.end_frame = from_frame, to_frame
        frame_interval = (self.end_frame - self.start_frame)
        self._mode = 'hermite'

        '''
        self.HERMATRIX = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        '''

        # Default tangents in flame are 0, so when we do None.to_f this is what we will get
        # CC = {P1, P2, T1, T2}
        p1, p2, t1, t2 = value1, value2, tangent1 * frame_interval, tangent2 * frame_interval
        self.hermite = np.array([p1, p2, t1, t2, 0])

    def value_at(self, frame, HERMATRIX):
        if frame == self.start_frame:
            return self.hermite[0]

        # Get the 0 < T < 1 interval we will interpolate on
        t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

        # S[s_] = {s^4, s^3, s^2, s^1, s^0}
        multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])
        self.basis = np.dot(HERMATRIX, self.hermite)

        # P[s_] = S[s].h.CC
        interpolated_scalar = np.dot(self.basis, multipliers_vec)
        return interpolated_scalar

a = -3
b = 4

rows = []

for a1 in range (a, b):
    for a2 in range (a, b):
        for a3 in range (a, b):
            for a4 in range (a, b):
                for a5 in range (a, b):
                    R = [a1, a2, a3, a4, a5]
                    rows.append(R)


test_frames_one = [12]
test_frames_three = [8, 12, 18]
test_frames_five = [4, 8, 12, 18, 24]

correct_result_one = {}
for frame in test_frames_one:
    correct_result_one[frame] = round(correct_values[frame], 2)

correct_result_three = {}
for frame in test_frames_three:
    correct_result_three[frame] = round(correct_values[frame], 2)

correct_result_five = {}
for frame in test_frames_five:
    correct_result_five[frame] = round(correct_values[frame], 2)


interp = HermiteSegmentQuartic(
    1,
    28,
    1,
    21.4979591,
    1,
    -0.5,
)

def test_values(HERMATRIX):
    test_result = {}

    for test_frame in test_frames_one:
        test_result[test_frame] = round(interp.value_at(test_frame, HERMATRIX), 2)

    if list(test_result.values()) == list(correct_result_one.values()):
        test_result = {}
        for test_frame in test_frames_three:
            test_result[test_frame] = round(interp.value_at(test_frame, HERMATRIX), 2)
        
        if list(test_result.values()) == list(correct_result_three.values()):
            print ('*********')
            pprint (HERMATRIX)
            print ('*********')

            # test_result = {}
            # for test_frame in test_frames_five:
            #    test_result[test_frame] = round(interp.value_at(test_frame, HERMATRIX), 2)
            # if list(test_result.values()) == list(correct_result_five.values()):
    # del interp

if __name__ == '__main__':
    for row5 in rows:
        for row4 in rows:
            for row3 in rows:
                for row2 in rows:
                    print ("%s, %s, %s, %s, %s" % ('[*, *, *, *, *]', row2, row3, row4, row5), end="\r", flush=True)
                    MATRIXES = []
                    for row1 in rows:
                        MATRIXES.append(np.array([row1, 
                        row2, 
                        row3, 
                        row4, 
                        row5]))
                    for M in MATRIXES:
                        test_values(M)
                    # pool = multiprocessing.Pool(1)
                    # pool.map(test_values, MATRIXES)
                    # pool.close()
                    # pool.join()
                    # print ('row5 passed with %s values' % index)
                    # sys.exit()
                # print ('row4 passed')
            # print ('row3 passed')
        # print ('row2 passed')
    # print ('row1 passed: done')





                    

                    


                




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
        self.value1 = value1
        self.value2 = value2
        self.tangent1 = tangent1
        self.tangent2 = tangent2
        self.frame_interval = frame_interval


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
        self.hermite = np.array([p1, p2, t1, t2])

    def value_at(self, frame, alpha, beta):
        if frame == self.start_frame:
            return self.hermite[0]

        # Get the 0 < T < 1 interval we will interpolate on
        t = (frame - self.start_frame) / (self.end_frame - self.start_frame)

        # S[s_] = {s^4, s^3, s^2, s^1, s^0}
        multipliers_vec = np.array([t ** 4, t ** 3, t ** 2, t ** 1, t ** 0])
        # self.basis = np.dot(HERMATRIX, self.hermite)

        # P[s_] = S[s].h.CC
        # interpolated_scalar = np.dot(self.basis, multipliers_vec)
        # quatric functions

        aa0 = 1 + (alpha - 3)*(t ** 2) + 2 * (1 - alpha) * (t ** 3) + alpha * (t ** 4)
        aa1 = (3 - alpha) * (t **2) + 2 * (alpha - 1) * (t ** 3) - alpha * (t ** 4)
        bb0 = t + (beta - 2) * (t ** 2) + (1 - 2 * beta) * (t ** 3) + beta * (t ** 4)
        bb1 = -1 * (beta + 1) * (t ** 2) + (2 * beta + 1) * (t ** 3) - beta * (t ** 4)

        # P[s_] = S[s].h.CC
        # interpolated_scalar = np.dot(self.basis, multipliers_vec)
        p1, p2, t1, t2 = self.value1, self.value2, self.tangent1 * self.frame_interval, self.tangent2 * self.frame_interval
        # interpolated_scalar = a0*p1 + a1*p2 + b0*t1 + b1*t2
        interpolated_scalar = aa0*p1 + aa1*p2 + bb0*t1 + bb1*t2
        return interpolated_scalar

a = -4
b = 4

rows = []

for a1 in range (a, b+1):
    for a2 in range (a, b+1):
        for a3 in range (a, b+1):
            for a4 in range (a, b+1):
                R = [a1, a2, a3, a4]
                rows.append(R)


test_frames_one = [12, 18]
test_frames_three = [8, 12, 18]
test_frames_five = [4, 8, 12, 18, 24]

correct_result_one = {}
for frame in test_frames_one:
    correct_result_one[frame] = round(correct_values[frame], 2)

correct_result_three = {}
for frame in test_frames_three:
    correct_result_three[frame] = round(correct_values[frame], 1)

correct_result_five = {}
for frame in test_frames_five:
    correct_result_five[frame] = round(correct_values[frame], 1)


interp = HermiteSegmentQuartic(
    1,
    28,
    1,
    21.4979591,
    1,
    -0.5,
)

def test_values(alpha, beta):
    test_result = {}

    for test_frame in test_frames_one:
        test_result[test_frame] = round(interp.value_at(test_frame, alpha, beta), 2)
        print ("%s, %s" % (test_frame, test_result[test_frame]), end="\r", flush=True)


    if list(test_result.values()) == list(correct_result_one.values()):
        print ('*********')
        print ('alpha: %s, beta: %s' % (alpha, beta))
        print ('*********')

        test_result = {}
        for test_frame in test_frames_three:
            test_result[test_frame] = round(interp.value_at(test_frame, alpha, beta), 1)
        
        if list(test_result.values()) == list(correct_result_three.values()):
            print ('*********')
            print ('alpha: %s, beta: %s' % (alpha, beta))
            print ('*********')

            # test_result = {}
            # for test_frame in test_frames_five:
            #    test_result[test_frame] = round(interp.value_at(test_frame, HERMATRIX), 2)
            # if list(test_result.values()) == list(correct_result_five.values()):
    # del interp

if __name__ == '__main__':
    for alpha in range (-100000, 100000):
        for beta in range (-100000, 100000):
            test_values(alpha/1000, beta / 1000)
'''
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
'''





                    

                    


                




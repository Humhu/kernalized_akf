import numpy as np
from bisect import bisect_left

def get_nearest(keys, items, query):
    i = bisect_left(keys, query)
    
    # Far right (more) case
    if i == len(keys):
        return items[i-1]
    ti = keys[i]
    # Far left (less) case
    if i == 0:
        return items[i]
    # Middle cases
    j = i - 1
    tj = keys[j]
    # If ti is closer to t than tj
    if (query - tj) < (ti - query):
        return items[j]
    else:
        return items[i]

class TruthEstimator(object):

    def __init__(self, tq, Q, tr, R):
        self.Qlist = zip(tq, Q)
        self.Qlist.sort(key = lambda x : x[0])
        self.Qtimes = [q[0] for q in self.Qlist]
        self.Rlist = zip(tr, R)
        self.Rlist.sort(key = lambda x : x[0])
        self.Rtimes = [r[0] for r in self.Rlist]

    def update(self, t, pre, up, x=None):
        pass

    def update_params(self):
        pass

    def predict_q(self, t, x=None):
        return get_nearest(keys=self.Qtimes, items=self.Qlist, query=t)[1]

    def predict_r(self, t, x=None):
        return get_nearest(keys=self.Rtimes, items=self.Rlist, query=t)[1]
    
    def single_gradient(self):
        return None

    def remove_before(self, t=None, now=None):
        pass
        
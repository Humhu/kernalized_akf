# cython: profile=True

cimport numpy as np
import numpy as np
import math
from itertools import izip
cimport kernels

cdef class Regressor(object):

    cdef list x
    cdef list y
    cdef list prior_x
    cdef list prior_y
    cdef kernels.Kernel kfunc

    def __init__(self, kernels.Kernel kfunc):
        self.x = []
        self.y = []
        self.prior_x = []
        self.prior_y = []
        self.kfunc = kfunc

    cpdef void add(self, double t, np.ndarray y):
        self.x.append(t)
        self.y.append(y)

    cpdef void add_prior(self, double dt, np.ndarray y):
        self.prior_x.append(dt)
        self.prior_y.append(y)

    cpdef void clear(self):
        self.x = []
        self.y = []

    cpdef void clear_prior(self):
        self.prior_x = []
        self.prior_y = []

    cpdef void remove_before(self, double t):
        cdef int n_before = len(self.x)
        cdef int i = 0
        while i < len(self.x):
            if self.x[i] < t:
                del self.x[i]
                del self.y[i]
            else:
                i += 1

    def __len__(self):
        return len(self.x)

    cpdef np.ndarray[np.float_t, ndim=1] get_theta(self):
        return self.kfunc.get_theta()

    cpdef void set_theta(self, np.ndarray[np.float_t, ndim=1] th):
        self.kfunc.set_theta(th)

    cpdef np.ndarray regress(self, double t_query):
        cdef double t
        cdef np.ndarray y
        cdef double dt
        cdef double k
        cdef np.ndarray num
        cdef double den = 0
        cdef int init = False
        cdef int i

        if len(self.prior_x) == 0 and len(self.x) == 0:
            return None

        for i in range(len(self.prior_x)):
            dt = self.prior_x[i]
            y = self.prior_y[i]
            k = self.kfunc.evaluate(dt)

            if not init:
                num = k * y
                init = True
            else:
                num += k * y
            den += k

        for i in range(len(self.x)):
            t = self.x[i]
            y = self.y[i]
            dt = t_query - t
            k = self.kfunc.evaluate(dt)
            
            if not init:
                num = k * y
                init = True
            else:
                num += k * y
            den += k

        if den == 0:
            return None
        return num / den

    cpdef list deriv(self, double t_query):
        num = 0
        cdef double den = 0
        dnum = [0] * len(self.get_theta())
        dden = [0] * len(self.get_theta())
        cdef double t
        cdef double dt
        cdef double k
        cdef int i

        for i in range(len(self.x)):
            t = self.x[i]
            y = self.y[i]
            dt = t_query - t
            k = self.kfunc.evaluate(dt)
            dk = self.kfunc.deriv(dt)
            num += k * y
            den += k
            for i in range(len(dk)):
                dnum[i] += dk[i] * y
                dden[i] += dk[i]
        for i in range(len(self.prior_x)):
            dt = self.prior_x[i]
            y = self.prior_y[i]
            k = self.kfunc.evaluate(dt)
            dk = self.kfunc.deriv(dt)
            num += k * y
            den += k
            for i in range(len(dk)):
                dnum[i] += dk[i] * y
                dden[i] += dk[i]

        if den == 0:
            return None
        return [dnumi / den - num *ddeni / (den*den) for dnumi,ddeni in izip(dnum, dden)]
        # return dnum / den - num * dden / (den * den)





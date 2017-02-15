# cython: profile=True

cimport numpy as np
import numpy as np
import math

cdef class Kernel(object):

    cpdef double evaluate(self, double x):
        return 0

    cpdef np.ndarray[np.float_t, ndim=1] get_theta(self):
        return np.zeros(1)

    cpdef void set_theta(self, np.ndarray[np.float_t, ndim=1] th):
        pass

cdef class SigmoidKernel(Kernel):
    cdef double w
    cdef double h
    cdef double log_w

    def __init__(self, double w, double h):
        self.log_w = w
        self.w = math.exp(w)
        self.h = h

    cpdef double evaluate(self, double x):
        cdef double den = 1.0 + math.exp(-self.w * (self.h - x))
        return 1.0 / den

    cpdef np.ndarray[np.float_t, ndim=1] deriv(self, double x):
        cdef double f = self.evaluate(x)
        cdef double dx = self.h - x
        cdef double fbase = -(f**2) * -self.w * dx * math.exp(-self.w * dx)
        cdef ddw = fbase * self.w
        cdef ddh = fbase
        return np.array([ddw, ddh])

    cpdef np.ndarray[np.float_t, ndim=1] get_theta(self):
        return np.array([self.log_w, self.h])

    cpdef void set_theta(self, np.ndarray[np.float_t, ndim=1] th):
        self.log_w = th[0]
        self.w = math.exp(self.log_w)
        self.h = th[1]

class WeightKernel(object):
    def __init__(self, n, dt, min_x=0):
        self.log_weights = np.zeros(n)
        self.weights = np.exp(self.log_weights)
        self.dt = dt
        self.min_x = min_x

    def evaluate(self, x):
        i = math.floor(x / self.dt)
        if i >= len(self.weights) or i < 0:
            return 0
        return self.weights[i]

    def __str__(self):
        return 'Weight: [%s]' % np.array_str(self.weights, max_line_width=200, precision=5)

    def deriv(self, x):
        g = np.zeros(len(self.weights))
        if x <= self.min_x:
            return g
        i = math.floor(x / self.dt)
        if i < len(self.weights) and i >= 0:
            g[i] = self.weights[i]
        return g

    def get_theta(self):
        return self.log_weights

    def set_theta(self, th):
        self.log_weights = th
        self.weights = np.exp(self.log_weights)

cdef class BoxKernel(Kernel):

    cdef double w

    def __init__(self, w=1.0):
        self.w = w
    
    cpdef double evaluate(self, double x):
        if x <= 0 or x >= self.w:
            return 0
        return 1

    cpdef np.ndarray[np.float_t, ndim=1] deriv(self, double x):
        return np.zeros(1)

    cpdef np.ndarray[np.float_t, ndim=1] get_theta(self):
        return np.array([self.w])

    cpdef void set_theta(self, np.ndarray[np.float_t, ndim=1] th):
        self.w = th[0]

class ConstantKernel(object):

    def __init__(self, k=0.0):
        self.logw = k
        self.w = math.exp(self.logw)

    def evaluate(self, xa, xb=None):
        return self.w

    def __str__(self):
        return "Constant: %f" % self.w

    def deriv(self, xa, xb=None):
        return np.array([self(xa,xb)])

    def get_theta(self):
        return np.array([self.logw])

    def set_theta(self, th):
        self.logw = th[0]
        self.w = math.exp(self.logw)

class SumKernel(object):

    def __init__(self, ka, kb):
        self.ka = ka
        self.kb = kb

    def evaluate(self, xa, xb=None):
        if xb is None:
            xb = xa
        return self.ka(xa) + self.kb(xb)

    def __str__(self):
        return "Sum: (%s, %s)" % (str(self.ka), str(self.kb))

    def deriv(self, xa, xb=None):
        if xb is None:
            xb = xa
        da = self.ka.deriv(xa)
        db = self.kb.deriv(xb)
        return np.hstack((da,db))

    def get_theta(self):
        return np.hstack((self.ka.get_theta(), self.kb.get_theta()))

    def set_theta(self, th):
        la = len(self.ka.get_theta())
        lb = len(self.kb.get_theta())
        self.ka.set_theta(th[0:la])
        self.kb.set_theta(th[la:la+lb])

class ProductKernel(object):

    def __init__(self, ka, kb):
        self.ka = ka
        self.kb = kb

    def evaluate(self, xa, xb=None):
        if xb is None:
            xb = xa
        return self.ka(xa) * self.kb(xb)

    def __str__(self):
        return "Product: (%s, %s)" % (str(self.ka), str(self.kb))

    def deriv(self, xa, xb=None):
        if xb is None:
            xb = xa
        a = self.ka(xa)
        da = self.ka.deriv(xa)
        b = self.kb(xb)
        db = self.kb.deriv(xb)
        return np.hstack((b*da,a*db))

    def get_theta(self):
        return np.hstack((self.ka.get_theta(), self.kb.get_theta()))

    def set_theta(self, th):
        la = len(self.ka.get_theta())
        lb = len(self.kb.get_theta())
        self.ka.set_theta(th[0:la])
        self.kb.set_theta(th[la:la+lb])

class SEKernel(object):

    def __init__(self, w=0.0):
        self.logw = w
        self.w = math.exp(self.logw)

    def __str__(self):
        return "SquaredExp: exp(-x*x*%f)" % self.w

    def px(self, x):
        if np.iterable(x):
            x = np.asarray(x)
            return np.dot(x, x)
        else:
            return x*x

    def evaluate(self, x):
        return math.exp(-self.px(x)*self.w)

    def deriv(self, x):
        d = self(x) * -self.w*self.px(x)
        return np.array([d])

    def get_theta(self):
        return np.array([self.logw])

    def set_theta(self, th):
        self.logw = th[0]
        self.w = math.exp(self.logw)

class OneSidedSEKernel(object):

    def __init__(self, w=0.0):
        self.logw = w
        self.w = math.exp(self.logw)

    def evaluate(self, x):
        if x <= 0:
            return 0
        return math.exp(-x*x*self.w)

    def __str__(self):
        return "OneSidedSquaredExp: exp(-x*x*%f) for x > 0" % self.w        

    def deriv(self, x):
        if x <= 0:
            return np.array([0])
        # Since theta is log w, the kernel is actually a double exponential
        # e^-x*x*e^logw
        d = self(x) * -self.w*x*x
        return np.array([d])

    def get_theta(self):
        return np.array([self.logw])

    def set_theta(self, th):
        self.logw = th[0]
        self.w = math.exp(self.logw)

class OneSidedExpKernel(object):

    def __init__(self, w=0.0):
        self.logw = w
        self.w = math.exp(self.logw)

    def evaluate(self, x):
        if x <= 0:
            return 0
        return math.exp(-x*self.w)

    def __str__(self):
        return "OneSidedExp: exp(-|x|*%f) for x > 0" % self.w        

    def deriv(self, x):
        if x <= 0:
            return np.array([0])
        d = self(x) * -self.w*x
        return np.array([d])

    def get_theta(self):
        return np.array([self.logw])

    def set_theta(self, th):
        self.logw = th[0]
        self.w = math.exp(self.logw)
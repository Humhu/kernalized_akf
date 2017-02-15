cimport numpy as np

cdef class Kernel(object):
    cpdef double evaluate(self, double x)

    cpdef np.ndarray[np.float_t, ndim=1] get_theta(self)

    cpdef void set_theta(self, np.ndarray[np.float_t, ndim=1] th)

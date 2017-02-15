import numpy as np
from numpy.random import multivariate_normal as mvn
import math, random
from itertools import izip

def integrator_matrix(dt, dim, order, int_order=None):
    """Produces a discrete time dynamics integrator matrix.
    """
    N = dim*(order+1)
    A = np.identity(N)
    if int_order >= order:
        raise ValueError( 'int_order can be at most order' )
    if int_order is None:
        int_order = order
    
    acc = 1
    for i in range(int_order):
        iinds = range(0, dim*(order-i))
        jinds = range(dim*(i+1), A.shape[1])
        acc = acc * dt / (i+1)
        A[iinds, jinds] = acc
    return A

def deriv_control_matrix(dim, order, gain=1):
    N = dim * (order+1)
    B = np.zeros((N,dim))
    B[dim*order:,:] = np.identity(dim) * gain
    return B

def position_gain_matrix(dim, order, gain):
    N = dim * (order+1)
    K = np.zeros((dim, N))
    K[:,0:dim] = gain * np.identity(dim)
    return K

def closed_loop_matrix(A, B, K):
    return A - np.dot(B,K)

def rollout(dt, A, tf, x0, Q=None):
    N = A.shape[0]

    if Q is None:
        Q = 0.001 * dt * np.identity(N)
    
    t = np.linspace(start=0, stop=tf, num=math.ceil(tf/dt)+1)
    xs = [x0]
    Qs = [Q]
    for ti in t[:-1]:
        noise = mvn(np.zeros(N), Q)
        xi = np.dot(A, xs[-1]) + noise
        xs.append(xi)
        Qs.append(Q)
    return {'x' : np.array(xs), 't' : np.array(t), 'Q' : np.array(Qs)}
    
def ramps_trajectory(dt, dim, A, k=2.0, Q=None, vtar=None, iters=1):
    N = A.shape[0]
    x0 = np.zeros(N)

    x = [x0]
    if vtar is None:
        vtar = np.random.rand(dim)
    if Q is None:
        Q = 0.001 * dt * np.identity(N)
    def noise():
        return mvn(np.zeros(N), Q)

    ts = [0]
    t = 0
    Qs = [Q]
    for i in range(iters):
        # Stationary
        for i in np.arange(start=0, stop=2, step=dt):
            x_last = x[-1]
            x_last[-dim:] = np.zeros(dim)
            x_next = np.dot(A, x_last) + noise()
            x.append(x_next)
            t += dt
            ts.append(t)
            Qs.append(Q)

        # Ramp up
        for i in np.arange(start=0, stop=2, step=dt):
            x_last = x[-1]
            v_last = x_last[dim:2*dim]
            x_last[-dim:] = k*(vtar - v_last)
            x_next = np.dot(A, x_last) + noise()
            x.append(x_next)
            t += dt
            ts.append(t)
            Qs.append(Q)
            
        # Ramp down
        for i in np.arange(start=0, stop=2, step=dt):
            x_last = x[-1]
            x_last[-dim:] = np.zeros(dim)
            x_next = np.dot(A, x_last) + noise()
            x.append(x_next)
            t += dt
            ts.append(t)
            Qs.append(Q)
            
        # Stationary
        for i in np.arange(start=0, stop=2, step=dt):
            x_last = x[-1]
            v_last = x_last[dim:2*dim]
            x_last[-dim:] = -k*v_last
            x_next = np.dot(A, x_last) + noise()
            x.append(x_next)
            t += dt
            ts.append(t)
            Qs.append(Q)
            
    return {'x' : np.array(x), 't' : np.array(ts), 'Q' : np.array(Qs)}

def generate_observations(x, t, dim, C, base_cov=None):
    obs = []
    R = []
    errs = []
    ts = []
    mask = []
    obs_dim = C.shape[0]
    if base_cov is None:
        base_cov = 0.005 * np.identity(obs_dim)

    for ti, xi in izip(t,x):
        v = xi[dim:2*dim]
        ss = np.dot(v,v)
        noise_cov = base_cov * 2.0 * ss + 1E-6*np.identity(obs_dim)
        
        p_thresh = math.exp(-ss)
        ts.append(ti)
        if random.random() > p_thresh:
            obs.append(None)
            R.append(None)
            errs.append(None)
            mask.append(False)
        else:
            noise = np.random.multivariate_normal(np.zeros(obs_dim), noise_cov)
            obs.append(np.dot(C, xi) + noise)
            R.append(noise_cov)
            errs.append(noise)
            mask.append(True)

    return {'obs' : obs, 'R' : R, 'noise' : errs, 't' : ts, 'mask' : mask}

def pos_obs_matrix(dim, order):
    C = np.zeros((dim, dim*(order+1)))
    C[:,0:dim] = np.identity(dim)
    return C

def vel_obs_matrix(dim, order):
    C = np.zeros((dim, dim*(order+1)))
    C[:,dim:2*dim] = np.identity(dim)
    return C

def all_obs_matrix(dim, order):
    return np.identity(dim*(order+1))

def velacc_obs_matrix(dim, order):
    C = np.zeros((2*dim, dim*(order+1)))
    C[:,dim:3*dim] = np.identity(2*dim)
    return C

def posvel_obs_matrix(dim, order):
    C = np.zeros((2*dim, dim*(order+1)))
    C[:,0:2*dim] = np.identity(2*dim)
    return C
from regressor import *
import numpy as np
from scipy.stats import multivariate_normal
import scipy.optimize

def gll_deriv(x, cov):
    inv_cov = np.linalg.solve(cov, np.identity(cov.shape[0]))
    inv_samp = np.linalg.solve(cov, x)
    return -0.5 * (inv_cov - np.outer(inv_samp, inv_samp))

class AdaptiveEstimator(object):

    def __init__(self, qkernel, rkernel, use_diag=True, online_opt=False,
                 online_step=float(1E-3), online_ss=1, buff_len=float('Inf'),
                 Q_def=None, R_def=None):
       self.kq = qkernel
       self.qreg = Regressor(self.kq)
       self.kr = rkernel
       self.rreg = Regressor(self.kr)
       self.inno_buff = []
       self.use_diag = use_diag
       self.online_opt = online_opt
       self.online_step = online_step
       self.online_ss = online_ss
       self.online_counter = 0
       self.buff_len = buff_len
       self.Q_def = Q_def
       self.R_def = R_def

    def __len__(self):
        return len(self.inno_buff)

    def add_q_prior(self, dt, q, dx=None):
        self.qreg.add_prior(dt=dt, y=q)

    def add_r_prior(self, dt, r, dx=None):
        self.rreg.add_prior(dt=dt, y=r)

    def update(self, t, pre, up):
        # Version using residual
        # hpht = np.dot(np.dot(up.C, up.Ppost), up.C.T)
        # rest = np.outer(up.residual, up.residual) + hpht

        # Version using innovation
        rest = np.outer(up.inno, up.inno) - np.dot(np.dot(up.C, pre.Ppre), up.C.T)
        if self.use_diag:
            rest = np.diag(rest).copy()
            rest[rest<0] = 1E-6
            rest = np.diag(rest)
        self.rreg.add(t=t, y=rest)

        qest = np.dot(up.K, up.inno)
        qest = np.outer(qest, qest) + up.Ppost - np.dot(np.dot(pre.A, pre.lastPpost), pre.A.T)

        # Version assuming steady state
        # qest = np.dot(up.K, up.inno)
        # qest = np.outer(qest, qest)

        if self.use_diag:
            qdiag = np.diag(qest).copy()
            qdiag[qdiag<0] = 0
            qest = np.diag(qdiag)
        self.qreg.add(t=t, y=qest)

        self.inno_buff.append((t, pre, up))

    def update_params(self):
        def obj(th):
            th_old = self.get_theta()
            self.set_theta(th)
            res = self.log_likelihood()
            self.set_theta(th_old)
            return -res

        def jac(th):
            th_old = self.get_theta()
            self.set_theta(th)
            res = self.gradient()
            self.set_theta(th_old)
            return -res

        if not self.online_opt:
            return

        # result = scipy.optimize.minimize(obj, x0=self.get_theta(), method='L-BFGS-B', jac=True, bounds=[[-6,6]]*len(self.get_theta()))
        # self.set_theta(result['x'])

        grad = self.single_gradient()
        self.online_counter += 1
        if grad is not None and self.online_counter % self.online_ss == 0:
            # ll0 = self.log_likelihood()
            # alpha = scipy.optimize.line_search(obj, jac, self.get_theta(), grad)[0]
            # alpha = self.online_step
            # beta = 0.5
            # th = self.get_theta()
            # for i in range(10):
            #     self.set_theta(th + alpha * grad)
            #     ll = self.log_likelihood()
            #     if ll > ll0:
            #         break
            #     alpha *= beta
            self.set_theta(self.get_theta() + grad*self.online_step)
            # print 'LL before: %f after: %f' % (ll0, self.log_likelihood())

    def clear(self):
        self.qreg.clear()
        self.rreg.clear()
        self.inno_buff = []

    def predict_q(self, t, x=None):
        Q = self.qreg.regress(t)
        if Q is None:
            return self.Q_def
        else:
            return Q

    def predict_r(self, t, x=None):
        R = self.rreg.regress(t)
        if R is None:
            return self.R_def
        else:
            return R

    def remove_before(self, t=None, now=None):
        if t is None and now is None:
            raise ValueError('Must specify t or now')
        if now is not None:
            t = now - self.buff_len
        self.qreg.remove_before(t)
        self.rreg.remove_before(t)

    def get_theta(self):
        return np.hstack((self.qreg.get_theta(),
                          self.rreg.get_theta()))

    def set_theta(self, th):
        th = np.asarray(th)
        if len(th) != len(self.get_theta()):
            raise ValueError('Theta incorrect dimension.')
        qth = self.qreg.get_theta()
        self.qreg.set_theta(th[0:len(qth)])
        self.rreg.set_theta(th[len(qth):])

    def optimize(self):
        obj = lambda x : -self.log_likelihood(x)
        jac = lambda x : -self.gradient(x)
        res = scipy.optimize.minimize(fun=obj, jac=jac, x0=self.get_theta(),
                                      method='bfgs')
        return res

    def log_likelihood(self, theta=None):
        if theta is not None:
            theta_old = self.get_theta()
            self.set_theta(theta)

        acc = 0
        for dat in self.inno_buff:
            t, pre, up = dat
            Q = self.predict_q(t=t)
            R = self.predict_r(t=t)
            if Q is None or R is None:
                continue

            temp = np.dot(np.dot(pre.A, pre.lastPpost), pre.A.T)
            Cv = R + np.dot(np.dot(up.C, Q), up.C.T) + \
                 np.dot(np.dot(up.C, temp), up.C.T)
            acc += multivariate_normal.logpdf(x=up.inno, cov=Cv)

        if theta is not None:
            self.set_theta(theta_old)
        return acc

    def single_gradient(self, i=-1, theta=None):
        if theta is not None:
            theta_old = self.get_theta()
            self.set_theta(theta)

        dat = self.inno_buff[i]
        t, pre, up = dat
        Q = self.predict_q(t=t)
        R = self.predict_r(t=t)
        dQ = self.qreg.deriv(t)
        dR = self.rreg.deriv(t)
        if Q is None or R is None or dQ is None or dR is None:
            return np.zeros(len(self.get_theta()))

        temp = np.dot(np.dot(pre.A, pre.lastPpost), pre.A.T)
        Cv = R + np.dot(np.dot(up.C, Q), up.C.T) + \
                np.dot(np.dot(up.C, temp), up.C.T)
        dL = gll_deriv(x=up.inno, cov=Cv)

        dCvdQ = [np.dot(np.dot(up.C, dQi), up.C.T) for dQi in dQ]
        # dCvdR = [dRi for dRi in dR]
        # dCv = [np.dot(np.dot(up.C, dQ), up.C.T), dR]
        dCv = dCvdQ + dR
        dthi = [np.sum(dL*dCvi) for dCvi in dCv]

        if theta is not None:
            self.set_theta(theta_old)
        return np.hstack(dthi)

    def gradient(self, theta=None):
        if theta is not None:
            theta_old = self.get_theta()
            self.set_theta(theta)

        dth = np.zeros(len(self.get_theta()))
        for i in range(len(self.inno_buff)):
            g = self.single_gradient(i)
            if g is None:
                continue
            dth += g

        if theta is not None:
            self.set_theta(theta_old)
        return np.squeeze(np.array(dth))

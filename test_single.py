from regressor import *
from kernels import *
from dynamics import *
from kalman_filter import *
from adaptive_estimator import *
from evaluation import *
from ground_truth import *
from scipy.stats import multivariate_normal as mvn
from itertools import izip
import matplotlib.pyplot as plt

# Parameters
dt = 0.5
dim = 1
order = 1
N = dim*(order+1)
traj_time = 100
init_burn = 2
x_dim = dim*(order+1)

# Generate data
vtar = [1]

def Afunc(dt):
    A = integrator_matrix(dt=dt, dim=dim, order=order)
    B = deriv_control_matrix(dim=dim, order=order, gain=dt)
    K = position_gain_matrix(dim=dim, order=order, gain=0.1)
    return closed_loop_matrix(A=A, B=B, K=K)

C = pos_obs_matrix(dim=dim, order=order)
Q = 1E-6*np.identity(N)
R = 1E-6*np.identity(C.shape[0])

kf = KalmanFilter(Afunc=Afunc, C=C)

q_ose = OneSidedSEKernel()
q_con = ConstantKernel()
q_pse = SumKernel(q_ose, q_con)

r_ose = OneSidedSEKernel()
r_con = ConstantKernel()
r_pse = SumKernel(r_ose, r_con)

q_sig = SigmoidKernel(0,1)
r_sig = SigmoidKernel(0,1)

q_raw = WeightKernel(n=20, dt=0.1, min_x=dt)
r_raw = WeightKernel(n=20, dt=0.1, min_x=dt)

ae_akf = AdaptiveEstimator(qkernel=q_sig,
                           rkernel=r_sig,
                           use_diag=True,
                           online_opt=True,
                           online_step=1E-1,
                           online_ss=1,
                           buff_len=2)
# ae_ose.add_q_prior(dt=1.0, q=Q)
# ae_ose.add_r_prior(dt=1.0, r=R)

q_box = BoxKernel(1)
r_box = BoxKernel(1)
ae_box = AdaptiveEstimator(qkernel=q_box,
                           rkernel=r_box,
                           use_diag=True,
                           online_opt=False,
                           buff_len=2)
# ae_box.add_q_prior(dt=0.5, q=Q)
# ae_box.add_r_prior(dt=0.5, r=R)

print 'Generating trajectory...'
x0 = np.array([1,0])
P0 = 0.01*np.identity(x_dim)
traj = rollout(dt=dt, A=Afunc(dt), tf=traj_time, x0=x0, Q=Q)
obs = generate_observations(x=traj['x'], t=traj['t'], dim=dim, C=C)
te = TruthEstimator(tq=traj['t'], Q=traj['Q'], tr=traj['t'], R=obs['R'])

print 'Running box AKF...'
kf.initialize(x=x0, P=P0, t=0)
box_res = run_filter(t=obs['t'], obs=obs['obs'],
                     estimator=ae_box, kf=kf, init_burn=init_burn,
                     Q_def=Q, R_def=R)
print 'Running ideal KF...'
kf.initialize(x=x0, P=P0, t=0)
tru_res = run_filter(t=obs['t'], obs=obs['obs'],
                     estimator=te, kf=kf, init_burn=init_burn,
                     Q_def=Q, R_def=R)

print 'Running OKAKF...'
kf.initialize(x=x0, P=P0, t=0)
akf_res = run_filter(t=obs['t'], obs=obs['obs'],
                     estimator=ae_akf, kf=kf, init_burn=init_burn,
                     Q_def=Q, R_def=R)

plt.ion()
plt.figure()
plt.subplot(2,1,1)
plt.plot(box_res['inno_lls'], 'r-')
plt.plot(tru_res['inno_lls'], 'g-')
plt.plot(akf_res['inno_lls'], 'b-')
plt.ylabel('vLL')

print 'Box vLL: %f' % np.sum(box_res['inno_lls'])
print 'Ideal vLL: %f' % np.sum(tru_res['inno_lls'])
print 'AKF vLL: %f' % np.sum(akf_res['inno_lls'])

box_xll = [mvn.logpdf(x=xi, cov=P) for (xi,P) in izip(box_res['x'] - traj['x'], box_res['P'])]
tru_xll = [mvn.logpdf(x=xi, cov=P) for (xi,P) in izip(tru_res['x'] - traj['x'], tru_res['P'])]
akf_xll = [mvn.logpdf(x=xi, cov=P) for (xi,P) in izip(akf_res['x'] - traj['x'], akf_res['P'])]
plt.subplot(2,1,2)
plt.plot(box_xll, 'r-')
plt.plot(tru_xll, 'g-')
plt.plot(akf_xll, 'b-')
plt.ylabel('xLL')

print 'Box xLL: %f' % np.sum(box_xll)
print 'Ideal xLL: %f' % np.sum(tru_xll)
print 'AKF xLL: %f' % np.sum(akf_xll)

ot = [t for t,z in izip(obs['t'], obs['obs']) if z is not None]
on = [n for n,z in izip(obs['noise'], obs['obs']) if z is not None]

plt.figure()
plt.subplot(3,1,1)
plt.plot(traj['t'], traj['x'][:,0], 'g-')
plt.ylabel('x')

plt.subplot(3,1,2)
plt.plot(ot[1:], np.abs(akf_res['dx']), 'b.')
plt.plot(ot[1:], np.abs(box_res['dx']), 'r+')
plt.plot(ot[1:], np.abs(tru_res['dx']), 'gx')
plt.plot(akf_res['t'][1:], np.sqrt(akf_res['Q'][:,0,0]), 'b-')
plt.plot(box_res['t'][1:], np.sqrt(box_res['Q'][:,0,0]), 'r-')
plt.plot(tru_res['t'][1:], np.sqrt(tru_res['Q'][:,0,0]), 'g-')
plt.ylabel('State dx')

plt.subplot(3,1,3)
plt.plot(ot[1:], np.abs(akf_res['inno']), 'b.')
plt.plot(ot[1:], np.abs(on[1:]), 'gx')
plt.plot(ot[1:], np.sqrt(akf_res['R'][:,0,0]), 'b-')
plt.plot(ot[1:], np.sqrt(box_res['R'][:,0,0]), 'r-')
plt.plot(ot[1:], np.sqrt(tru_res['R'][:,0,0]), 'g-')
plt.ylabel('Obs errs')

from regressor import *
from dynamics import *
from kalman_filter import *
from adaptive_estimator import *
from evaluation import *
import matplotlib.pyplot as plt

# Parameters
dt = 0.01
dim = 2
order = 2
N = dim*(order+1)
alpha = 0.1
lag = 4.0

# Generate data
vtar = [1,-1]
traj = ramps_trajectory(dt=dt, dim=dim, order=order, vtar=vtar, iters=2)
obs, R_true = vel_observations(traj=traj, dim=dim)
A = integrator_matrix(dt=dt, dim=dim, order=order)
C = vel_obs_matrix(dim=dim, order=order)
Q = float(1E-1)*np.identity(N)
R = float(1E-1)*np.identity(dim)

ae_ose = AdaptiveEstimator(qkernel=OneSidedSEKernel(),
                           rkernel=OneSidedSEKernel())
x_ose, Q_ose, R_ose = run_filter(x_init=traj[0], obs=obs, estimator=ae_ose,
                                 sys_A=A, sys_C=C, dt=dt, lag=lag, Q_def=Q, R_def=R)

ae_box = AdaptiveEstimator(qkernel=BoxKernel(0.5), 
                           rkernel=BoxKernel(0.5))
x_box, Q_box, R_box = run_filter(x_init=traj[0], obs=obs, estimator=ae_box,
                                 sys_A=A, sys_C=C, dt=dt, lag=lag, Q_def=Q, R_def=R)

err_box = x_box - traj[1:]
verr_box = err_box[:,2:4]
vrms_box = [np.linalg.norm(err) for err in verr_box]
print 'OSE RMS: %f' % np.mean(vrms_box)

err_box = x_box - traj[1:]
verr_box = err_box[:,2:4]
vrms_box = [np.linalg.norm(err) for err in verr_box]
print 'Box RMS: %f' % np.mean(vrms_box)

plt.figure()
for i in range(dim):
    plt.subplot(dim,1,i+1)
    plt.plot(traj[1:,dim+i], 'r-')
    plt.plot(x_box[:,dim+i], 'g--')
    plt.plot(x_box[:,dim+i], 'b-')
    plt.plot(obs[1:,i], 'kx')

plt.figure()
for i in range(dim):
    plt.subplot(dim,1,i+1)
    plt.plot(R_true[:,i,i], 'r-')
    plt.plot(R_ose[:,i,i], 'g--')
    plt.plot(R_box[:,i,i], 'b-')

plt.show()
raw_input()

from regressor import *
from dynamics import *
from kalman_filter import *
from adaptive_estimator import *
from evaluation import *
from scipy.stats import multivariate_normal
from itertools import izip
import matplotlib.pyplot as plt

# Parameters
dt = 0.1
dim = 2
order = 2
N = dim*(order+1)
alpha = 0.1
lag = 3.0
grad_tol = 0
traj_iters = 6
init_burn = 0.0
grad_trim = 0.0
plot_skip = 6/dt
plot_stds = 2
x_dim = dim * (order+1)

# Generate data
vtar = [1,-1]
A = integrator_matrix(dt=dt, dim=dim, order=order)
A_sys = integrator_matrix(dt=dt, dim=dim, order=2)
C = pos_obs_matrix(dim=dim, order=order)
Q = np.identity(N)
R = np.identity(C.shape[0])
P_init = np.identity(x_dim)

online_opt=True

q_ose = OneSidedSEKernel()
q_sek = SEKernel()
q_con = ConstantKernel(-3)
q_pse = SumKernel(q_ose, q_con)
q_mix = SumKernel(q_ose, q_sek)

r_ose = OneSidedSEKernel()
r_sek = SEKernel()
r_con = ConstantKernel(-3)
r_pse = SumKernel(r_ose, r_con)
r_mix = SumKernel(r_ose, r_sek)

ae_ose = AdaptiveEstimator(qkernel=q_pse,
                           rkernel=r_pse,
                           use_diag=True,
                           online_opt=online_opt,
                           online_step=alpha)
ae_ose.add_q_prior(dt=1.0, q=Q)
ae_ose.add_r_prior(dt=1.0, r=R)

fig = plt.figure()
ll_ax = plt.subplot(3,1,1)
plt.ylabel('Obs LL')
ll_line = plt.plot(0,0,'bx-')[0]

rms_ax = plt.subplot(3,1,2)
plt.ylabel('Average RMS')
rms_line = plt.plot(0,0,'rx-')[0]

xll_ax = plt.subplot(3,1,3)
plt.ylabel('Pred LL')
plt.xlabel('iteration')
xll_line = plt.plot(0,0,'gx-')[0]

fig = plt.figure()
Q_ax = plt.subplot(3,1,1)
plt.ylabel('Q1')
Q_line = plt.plot(0,0,'g-')[0]
dx_scatter = plt.plot(0, 0, 'k.')[0]

R_ax = plt.subplot(3,1,2)
plt.ylabel('R1')
R_line = plt.plot(0,0,'r-')[0]
Rt_line = plt.plot(0, 0, 'b-')[0]
inno_scatter = plt.plot(0, 0, 'k.')[0]

P_ax = plt.subplot(3,1,3)
P_line = plt.plot(0,0,'b-')[0]
err_scatter = plt.plot(0, 0, 'k.')[0]
plt.ylabel('P1')
plt.xlabel('Timestep')

z_lls = []
x_lls = []
rms_avgs = []
gradients = []
plt.ion()
plt.show(False)

def update_ax(line, ax, y, x=None):
    if x is None:
        x = range(len(y))
    line.set_xdata(x)
    line.set_ydata(y)

    ax.set_xlim((min(x), max(x)))
    ax.set_ylim((min(y), max(y)))

def x_likelihood(x, P):
    lls = [multivariate_normal.logpdf(x=xi, cov=Pi) for xi,Pi in izip(x,P)]
    return np.sum(lls)

while True:
    traj = ramps_trajectory(dt=dt, dim=dim, A=A_sys, vtar=vtar, iters=traj_iters)
    obs, R_true = generate_observations(traj=traj, dim=dim, C=C)
    results = run_filter(x_init=traj[0], P_init=P_init, obs=obs,
                         estimator=ae_ose, dt=dt, lag=lag,
                         sys_A=A, sys_C=C, init_burn=init_burn,
                         Q_def=Q, R_def=R)
    ae_ose.remove_before(grad_trim) # Trim off initial effects
    z_lls.append(ae_ose.log_likelihood())
    grad = ae_ose.gradient()
    print grad
    if not online_opt:
        ae_ose.set_theta(ae_ose.get_theta() + alpha*grad)
    print 'Theta: ' + np.array_str(ae_ose.get_theta())
    print 'Qreg: ' + str(ae_ose.qreg.kfunc)
    print 'Rreg: ' + str(ae_ose.rreg.kfunc)
    ae_ose.clear()

    x_err = results['x'] - traj
    rms_err = [np.linalg.norm(err[0:2]) for err in x_err]
    rms_avgs.append(np.mean(rms_err))

    x_lls.append(x_likelihood(x_err[plot_skip:], results['P'][plot_skip:]))

    # Plot progress so far
    update_ax(line=ll_line, ax=ll_ax, y=z_lls)
    update_ax(line=rms_line, ax=rms_ax, y=rms_avgs)
    update_ax(line=xll_line, ax=xll_ax, y=x_lls)

    update_ax(line=Q_line, ax=Q_ax, y=plot_stds * np.sqrt(results['Q'][plot_skip:,0,0]))
    update_ax(line=dx_scatter, ax=Q_ax, y=np.abs(results['dx'][plot_skip:,0]))
    update_ax(line=R_line, ax=R_ax, y=plot_stds * np.sqrt(results['R'][plot_skip:,0,0]))
    update_ax(line=Rt_line, ax=R_ax, y=plot_stds * np.sqrt(R_true[plot_skip:,0,0]))
    update_ax(line=inno_scatter, ax=R_ax, y=np.abs(results['inno'][plot_skip:,0]))
    update_ax(line=P_line, ax=P_ax, y=plot_stds * np.sqrt(results['P'][plot_skip:,0,0]))
    update_ax(line=err_scatter, ax=P_ax, y=np.abs(x_err[plot_skip:,0]))

    plt.draw()
    plt.pause(0.01)

    if np.linalg.norm(grad) < grad_tol:
        print 'Gradient tolerance achieved. Terminating...'
        break

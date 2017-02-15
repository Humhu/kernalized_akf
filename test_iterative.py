from regressor import *
from kernels import *
from dynamics import *
from kalman_filter import *
from adaptive_estimator import *
from evaluation import *
from ground_truth import *
from scipy.stats import multivariate_normal
from itertools import izip
import matplotlib.pyplot as plt

# Parameters
dt = 0.1
dim = 2
order = 1
N = dim*(order+1)
alpha = 1E-1
alpha_ss = 1
lag = 2
grad_tol = 0
traj_iters = 5
traj_time = 100
init_burn = 2
plot_skip = 0/dt
plot_stds = 2
x_dim = dim*(order+1)

# Generate data
vtar = [1]
A = integrator_matrix(dt=dt, dim=dim, order=order)
B = deriv_control_matrix(dim=dim, order=order, gain=dt)
K = position_gain_matrix(dim=dim, order=order, gain=0.5)
Ak = closed_loop_matrix(A=A, B=B, K=K)
C = pos_obs_matrix(dim=dim, order=order)
Q = 0.01*np.identity(N)
R = 0.01*np.identity(C.shape[0])
P_init = 0.01*np.identity(x_dim)

online_opt = True

q_ose = OneSidedSEKernel()
q_sek = SEKernel()
q_con = ConstantKernel()
q_pse = SumKernel(q_ose, q_con)
q_mix = SumKernel(q_ose, q_sek)

r_ose = OneSidedSEKernel()
r_sek = SEKernel()
r_con = ConstantKernel()
r_pse = SumKernel(r_ose, r_con)
r_mix = SumKernel(r_ose, r_sek)

q_raw = WeightKernel(n=10, dt=0.1)
r_raw = WeightKernel(n=10, dt=0.1)
ae_ose = AdaptiveEstimator(qkernel=q_ose,
                           rkernel=r_ose,
                           use_diag=True,
                           online_opt=online_opt,
                           online_step=alpha,
                           online_ss=alpha_ss)
# ae_ose.add_q_prior(dt=1.0, q=Q)
# ae_ose.add_r_prior(dt=1.0, r=R)

q_box = BoxKernel(1)
r_box = BoxKernel(1)
ae_box = AdaptiveEstimator(qkernel=q_box,
                           rkernel=r_box,
                           use_diag=True,
                           online_opt=False)
# ae_box.add_q_prior(dt=0.5, q=Q)
# ae_box.add_r_prior(dt=0.5, r=R)

fig = plt.figure()
ll_ax = plt.subplot(3,1,1)
plt.ylabel('Obs LL')
ll_line = plt.plot(0,0,'bx-')[0]
boxll_line = plt.plot(0,0,'rx-')[0]
trull_line = plt.plot(0,0,'gx-')[0]

rms_ax = plt.subplot(3,1,2)
plt.ylabel('Average RMS')
rms_line = plt.plot(0,0,'bx-')[0]
boxrms_line = plt.plot(0,0,'rx-')[0]
trurms_line = plt.plot(0,0,'gx-')[0]

xll_ax = plt.subplot(3,1,3)
plt.ylabel('Pred LL')
plt.xlabel('iteration')
xll_line = plt.plot(0,0,'bx-')[0]
boxxll_line = plt.plot(0,0,'rx-')[0]
truxll_line = plt.plot(0,0,'gx-')[0]

fig = plt.figure()
Q_ax = plt.subplot(3,1,1)
plt.ylabel('Q1')
Q_line = plt.plot(0,0,'b-')[0]
boxQ_line = plt.plot(0, 0, 'r-')[0]
truQ_line = plt.plot(0,0,'g-')[0]
dx_scatter = plt.plot(0, 0, 'b.')[0]
boxdx_scatter = plt.plot(0,0,'rx')[0]
trudx_scatter = plt.plot(0,0,'g+')[0]

R_ax = plt.subplot(3,1,2)
plt.ylabel('R1')
R_line = plt.plot(0,0,'b-')[0]
Cv_line = plt.plot(0,0,'b--')[0]
boxR_line = plt.plot(0, 0, 'r-')[0]
boxCv_line = plt.plot(0, 0, 'r--')[0]
truR_line = plt.plot(0, 0, 'g-')[0]
inno_scatter = plt.plot(0, 0, 'b.')[0]
boxinno_scatter = plt.plot(0,0,'rx')[0]
truinno_scatter = plt.plot(0,0,'g+')[0]

P_ax = plt.subplot(3,1,3)
P_line = plt.plot(0,0,'b-')[0]
boxP_line = plt.plot(0, 0, 'r-')[0]
truP_line = plt.plot(0,0,'g-')[0]
err_scatter = plt.plot(0, 0, 'b.')[0]
boxerr_scatter = plt.plot(0,0,'rx')[0]
truerr_scatter = plt.plot(0,0,'g+')[0]

plt.ylabel('P1')
plt.xlabel('Timestep')

z_lls = []
boxz_lls = []
truz_lls = []
x_lls = []
boxx_lls = []
trux_lls = []
rms_avgs = []
boxrms_avgs = []
trurms_avgs = []
gradients = []
plt.ion()
plt.show(False)

def update_ax(ax, line, y, x=None):
    if not hasattr(line, '__iter__'):
        line = [line]
        y = [y]
    
    if x is None:
        x = [range(len(yi)) for yi in y]

    if len(line) != len(y) or len(y) != len(x):
        raise ValueError('Number of lines and data series must match.')
    
    xmin = float('inf')
    xmax = float('-inf')
    ymin = float('inf')
    ymax = float('-inf')
    for (l,xi,yi) in izip(line,x,y):
        l.set_xdata(xi)
        l.set_ydata(yi)
        xmin = min(min(xi),xmin)
        xmax = max(max(xi),xmax)
        ymin = min(min(yi),ymin)
        ymax = max(max(yi),ymax)        

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

def x_likelihood(x, P):
    lls = [multivariate_normal.logpdf(x=xi, cov=Pi) for xi,Pi in izip(x,P)]
    return np.sum(lls)

x0 = np.array([1,-1,0,0])

traj = rollout(dt=dt, A=Ak, tf=traj_time, x0=x0, Q=Q)
obs = generate_observations(traj=traj['x'], dim=dim, C=C)
te = TruthEstimator(tq=traj['t'], Q=traj['Q'], tr=traj['t'], R=obs['R'])

box_res = run_filter(x_init=traj['x'][0], P_init=P_init, obs=obs['obs'],
                        estimator=ae_box, dt=dt, lag=lag,
                        sys_A=Ak, sys_C=C, init_burn=init_burn,
                        Q_def=Q, R_def=R)
tru_res = run_filter(x_init=traj['x'][0], P_init=P_init, obs=obs['obs'],
                        estimator=te, dt=dt, lag=lag,
                        sys_A=Ak, sys_C=C, init_burn=init_burn,
                        Q_def=Q, R_def=R)

def obj(th):
    ae_ose.online_opt = False
    ae_ose.set_theta(th)
    results = run_filter(x_init=traj['x'][0], P_init=P_init, obs=obs['obs'],
                        estimator=ae_ose, dt=dt, lag=lag,
                        sys_A=Ak, sys_C=C, init_burn=init_burn,
                        Q_def=Q, R_def=R)
    g = ae_ose.gradient()
    ae_ose.clear()
    return -np.sum(results['inno_lls'])#, -g

while True:
    results = run_filter(x_init=traj['x'][0], P_init=P_init, obs=obs['obs'],
                         estimator=ae_ose, dt=dt, lag=lag,
                         sys_A=Ak, sys_C=C, init_burn=init_burn,
                         Q_def=Q, R_def=R)

    z_lls.append(np.sum(results['inno_lls']))
    boxz_lls.append(np.sum(box_res['inno_lls']))
    truz_lls.append(np.sum(tru_res['inno_lls']))

    grad = np.mean(results['grads'], axis=0)
    print grad
    if not online_opt:
        ae_ose.set_theta(ae_ose.get_theta() + alpha*grad)
    print 'Theta: ' + np.array_str(ae_ose.get_theta())
    print 'Qreg: ' + str(ae_ose.qreg.kfunc)
    print 'Rreg: ' + str(ae_ose.rreg.kfunc)
    print 'vLL: %f' % z_lls[-1]
    
    ae_ose.clear()
    ae_box.clear()

    x_err = results['x'] - traj['x']
    rms_err = [np.linalg.norm(err[0:2]) for err in x_err]
    rms_avgs.append(np.mean(rms_err))

    boxx_err = box_res['x'] - traj['x']
    rms_err = [np.linalg.norm(err[0:2]) for err in boxx_err]
    boxrms_avgs.append(np.mean(rms_err))

    trux_err = tru_res['x'] - traj['x']
    rms_err = [np.linalg.norm(err[0:2]) for err in trux_err]
    trurms_avgs.append(np.mean(rms_err))

    x_ll = x_likelihood(x_err[plot_skip:], results['P'][plot_skip:])
    boxx_ll = x_likelihood(boxx_err[plot_skip:], box_res['P'][plot_skip:])
    x_lls.append(x_ll)
    boxx_lls.append(boxx_ll)
    trux_lls.append(x_likelihood(trux_err[plot_skip:], tru_res['P'][plot_skip:]))

    # Plot progress so far
    update_ax(ax=ll_ax, line=(ll_line, boxll_line, trull_line), 
              y=(z_lls, boxz_lls, truz_lls))
    update_ax(ax=rms_ax, line=(rms_line, boxrms_line, trurms_line), 
              y=(rms_avgs, boxrms_avgs, trurms_avgs))
    update_ax(ax=xll_ax, line=(xll_line, boxxll_line, truxll_line), 
              y=(x_lls, boxx_lls, trux_lls))

    update_ax(ax=Q_ax, line=(dx_scatter, Q_line, boxQ_line, truQ_line),
              y=(np.abs(results['dx'][plot_skip:,0]),
                 plot_stds*np.sqrt(results['Q'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(box_res['Q'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(tru_res['Q'][plot_skip:,0,0])))
    update_ax(ax=R_ax, line=(inno_scatter, R_line, Cv_line, 
                             boxinno_scatter, boxR_line, boxCv_line, 
                             truinno_scatter, truR_line),
              y=(np.abs(results['inno'][plot_skip:,0]),
                 plot_stds*np.sqrt(results['R'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(results['Cv'][plot_skip:,0,0]),
                 np.abs(box_res['inno'][plot_skip:,0]),
                 plot_stds*np.sqrt(box_res['R'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(box_res['Cv'][plot_skip:,0,0]),
                 np.abs(tru_res['inno'][plot_skip:,0]),
                 plot_stds*np.sqrt(tru_res['R'][plot_skip:,0,0])))
    update_ax(ax=P_ax, line=(err_scatter, P_line, boxP_line, truP_line),
              y=(np.abs(x_err[plot_skip:,0]),
                 plot_stds*np.sqrt(results['P'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(box_res['P'][plot_skip:,0,0]),
                 plot_stds*np.sqrt(tru_res['P'][plot_skip:,0,0])))

    plt.draw()
    plt.pause(0.01)

    if np.linalg.norm(grad) < grad_tol:
        print 'Gradient tolerance achieved. Terminating...'
        break

from regressor import *
from kernels import *
from dynamics import *
from kalman_filter import *
from adaptive_estimator import *
from execution import *
from ground_truth import *
from scipy.stats import multivariate_normal as mvn
from itertools import izip
import matplotlib.pyplot as plt

# Parameters
dim = 2
order = 1
N = dim*(order+1)
traj_steps = 1000
init_burn = 0
x_dim = dim*(order+1)

def logpdf(x, cov):
    try:
        return mvn.logpdf(x=x, cov=cov)
    except np.linalg.LinAlgError:
        return float('nan')

def evaluate_dt(dt):

    print 'Testing dt %f' % dt

    # Generate data
    vtar = [1]
    traj_time = traj_steps * dt

    def Afunc(dt):
        A = integrator_matrix(dt=dt, dim=dim, order=order)
        B = deriv_control_matrix(dim=dim, order=order, gain=dt)
        K = position_gain_matrix(dim=dim, order=order, gain=0.1)
        return closed_loop_matrix(A=A, B=B, K=K)

    C = pos_obs_matrix(dim=dim, order=order)
    Q = 1E-4*np.identity(N)
    R = 1E-4*np.identity(C.shape[0])

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
                            buff_len=2,
                            Q_def=Q,
                            R_def=R)
    ae_akf.add_q_prior(dt=2.0, q=Q)
    ae_akf.add_r_prior(dt=2.0, r=R)

    q_box = BoxKernel(1)
    r_box = BoxKernel(1)
    ae_box = AdaptiveEstimator(qkernel=q_box,
                            rkernel=r_box,
                            use_diag=True,
                            online_opt=False,
                            buff_len=2,
                            Q_def=Q,
                            R_def=R)
    # ae_box.add_q_prior(dt=0.5, q=Q)
    # ae_box.add_r_prior(dt=0.5, r=R)

    print 'Generating trajectory...'
    x0 = np.array([1,-1,0,1])
    P0 = 0.01*np.identity(x_dim)
    traj = rollout(dt=dt, A=Afunc(dt), tf=traj_time, x0=x0, Q=Q)
    obs = generate_observations(x=traj['x'], t=traj['t'], dim=dim, C=C)
    traj = zip(obs['t'], obs['obs'])
    te = TruthEstimator(tq=traj['t'], Q=traj['Q'], tr=traj['t'], R=obs['R'])

    print 'Running box AKF...'
    kf.initialize(x=x0, P=P0, t=0)
    box_res = run_filter(traj=traj, estimator=ae_box, kf=kf)
    print 'Running ideal KF...'
    kf.initialize(x=x0, P=P0, t=0)
    tru_res = run_filter(traj=traj, estimator=te, kf=kf)

    print 'Running OKAKF...'
    kf.initialize(x=x0, P=P0, t=0)
    akf_res = run_filter(traj=traj, estimator=ae_akf, kf=kf)

    box_xll = [logpdf(x=xi, cov=P) for (xi,P) in izip(box_res['up_x'][:-1] - traj['x'], box_res['est_P'])]
    tru_xll = [logpdf(x=xi, cov=P) for (xi,P) in izip(tru_res['up_x'][:-1] - traj['x'], tru_res['est_P'])]
    akf_xll = [logpdf(x=xi, cov=P) for (xi,P) in izip(akf_res['up_x'][:-1] - traj['x'], akf_res['est_P'])]

    box_vll = [logpdf(x=v.inno, cov=v.Cv) for v in box_res['up_raw']]
    tru_vll = [logpdf(x=v.inno, cov=v.Cv) for v in tru_res['up_raw']]
    akf_vll = [logpdf(x=v.inno, cov=v.Cv) for v in akf_res['up_raw']]

    return {'akf_vll' : np.sum(akf_vll),
            'box_vll' : np.sum(box_vll),
            'tru_vll' : np.sum(tru_vll),
            'akf_xll' : np.sum(akf_xll),
            'box_xll' : np.sum(box_xll),
            'tru_xll' : np.sum(tru_xll),
            'theta' : ae_akf.get_theta()}

n_trials = 1
dts = np.logspace(start=-2, stop=-1, num=6)
res = np.array([[evaluate_dt(dt) for i in range(n_trials)] for dt in dts])

plt.ion()
plt.figure()
all_thetas = np.array([[ri['theta'] for ri in r] for r in res])
thetas = np.mean(all_thetas, axis=1)
cm = plt.get_cmap('gist_rainbow')
get_color = lambda i : cm( i * 1.0 / (thetas.shape[1]-1) )

for i in range(thetas.shape[1]):
    plt.semilogx(dts, thetas[:,i], '-x', color=get_color(i), label='theta %d' % i)
plt.xlabel('dt')
plt.ylabel('theta')
plt.legend(loc='best')

akf_vll = np.array([[ri['akf_vll'] for ri in r] for r in res])
box_vll = np.array([[ri['box_vll'] for ri in r] for r in res])
tru_vll = np.array([[ri['tru_vll'] for ri in r] for r in res])
akf_xll = np.array([[ri['akf_xll'] for ri in r] for r in res])
box_xll = np.array([[ri['box_xll'] for ri in r] for r in res])
tru_xll = np.array([[ri['tru_xll'] for ri in r] for r in res])
plt.figure()
plt.subplot(2,1,1)
plt.semilogx(dts, np.mean(akf_vll, 1), 'b-.')
plt.semilogx(dts, np.mean(box_vll, 1), 'r-x')
# plt.semilogx(dts, np.mean(tru_vll, 1), 'g-+')
plt.ylabel('vll')
plt.subplot(2,1,2)
plt.semilogx(dts, np.mean(akf_xll, 1), 'b-.', label='okakf')
plt.semilogx(dts, np.mean(box_xll, 1), 'r-x', label='box')
# plt.semilogx(dts, np.mean(tru_xll, 1), 'g-+', label='ideal')
plt.ylabel('xll')
plt.xlabel('dt')

from kalman_filter import KalmanFilter
import numpy as np
from itertools import izip
from scipy.stats import multivariate_normal as mvn

def run_filter(estimator, kf, traj, est_times=None):
    """Runs a kalman filter on a trajectory with the specified covariance
    estimator object.

    Parameters
    ----------
    estimator : Covariance estimator
    kf : Initialized Kalman filter
    traj : List of (time,value) observation tuples
    est_times: Additional times at which to generate estimates

    Returns
    -------
    init_t : initial time
    init_x : initial state
    init_P : initial covariance

    est_t : estimated times - does not exist if est_times is None
    est_x : estimated states - does not exist if est_times is None
    est_P : estimated state covariances - does not exist if est_times is None

    pre_t : prediction step times
    pre_x : prediction step states
    pre_P : prediction step state covariances
    pre_Q : predicted transition covariances
    pre_raw : Kalman filter predict named tuples

    up_t : update times
    up_x : update step states
    up_P : update step state covariances
    up_R : predicted update covariances
    up_raw : Kalman filter update named tuples
    # grads : single-time gradients
    """
    
    # Merge observation and estimate times
    if est_times is not None:
        pred_data = [(d, None) for d in est_times]
        traj += pred_data
    traj.sort(key=lambda x: x[0])

    # Store initial values to report later
    init_x = kf.x
    init_P = kf.P
    init_t = kf.t

    if est_times is not None:
        est_t = []
        est_x = []
        est_P = []

    pre_t = []
    pre_x = []
    pre_P = []
    pre_Q = []
    pre_raw = []

    up_t = []
    up_x = []
    up_P = []
    up_R = []
    up_raw = []
    grads = []

    for t, obs in traj:

        # Clean out old data
        estimator.remove_before(now=t)

        # Filter predict
        Q_est = estimator.predict_q(t=t)
        pre = kf.predict(t=t, Q=Q_est)

        pre_t.append(t)
        pre_x.append(kf.x)
        pre_P.append(kf.P)
        pre_Q.append(Q_est)
        pre_raw.append(pre)

        if obs is None:
            est_x.append(kf.x)
            est_P.append(kf.P)
            est_t.append(t)
            continue

        # Filter update
        R_est = estimator.predict_r(t=t)
        up = kf.update(t=t, y=obs, R=R_est)

        up_t.append(t)
        up_x.append(kf.x)
        up_P.append(kf.P)
        up_R.append(R_est)
        up_raw.append(up)

        # Estimator update
        estimator.update(t=t, pre=pre, up=up)
        estimator.update_params()
        # g = estimator.single_gradient()
        # if g is not None:
            # grads.append(g)

    out = {'pre_x' : np.array(pre_x), 'pre_P' : np.array(pre_P), 'pre_t' : np.array(pre_t),
           'pre_Q' : np.array(pre_Q), 'pre_raw' : pre_raw,
           'up_x' : np.array(up_x), 'up_P' : np.array(up_P), 'up_t' : np.array(up_t),
           'up_R' : np.array(up_R), 'up_raw' : up_raw,
           'grads' : np.array(grads), 'init_x' : init_x, 'init_P' : init_P, 'init_t' : init_t}
    if est_times is not None:
        out['est_x'] = np.array(est_x)
        out['est_t'] = np.array(est_t)
        out['est_P'] = np.array(est_P)
    return out
    
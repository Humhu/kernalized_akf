import numpy as np
import execution
from scipy.stats import multivariate_normal as mvn
from itertools import izip

def compute_metrics(trace, key_func=None, truth=None):
    """Computes performance metrics from a results dict.

    Parameters
    ----------
    trace : Execution trace dict produced by run_filter
    key_func : Optional function to extract state vector/matrices
    truth : Optional ground truth state values

    Return Keys
    -----------
    """
    out = {}

    out['innovation_ll'] = [mvn.logpdf(x=d.inno, cov=d.Cv) for d in trace['up_raw']]

    if truth is not None:
        if key_func is None:
            key_func = lambda x: x

        x_err = [key_func(est_x) - true_x for est_x, true_x in izip(trace['est_x'], truth)]
        x_err = np.asarray(x_err)
        out['rms'] = np.sqrt(np.mean(x_err*x_err, axis=1))
        sub_P = [key_func(est_P) for est_P in trace['est_P']]
        out['state_ll'] = [mvn.logpdf(x=err, cov=P) for err, P in izip(x_err, sub_P)]
    return out

def evaluate_over_data(estimator, kf, trajs, init_x=None, init_P=None, init_t=None,
                       params=None, key_func=None):
    """Run an adaptive estimator on a set of data.

    Parameters
    ----------
    estimator  : An AdaptiveEstimator to train, assumed to be initialized already
    kf         : A KalmanFilter to use for training
    trajs      : A list of trajectory dicts
    init_x     : Initialization state(s) (default kf.x)
    init_P     : Initialization covariance(s) (default kf.P)
    init_t     : Initialization time(s) (default kf.t)
    params     : Optional parameters to set
    key_func   : Function to extract state vector/matrices

    Training Data Keys
    ------------------
    obs_times  : An iterable of scalars
    obs_vels   : An iterable of 1D numpy arrays

    Returns
    -------
    results : List of evaluation results
    """
    def check_init_size(inval, defval, depth, size):
        """Check array argument sizes
        """
        if inval is None:
            inval = defval
        inval = np.asarray(inval)
        if len(inval.shape) < depth:
            reps = np.ones(depth)
            reps[0] = size
            inval = np.tile(inval, reps)
        return inval

    num_trajs = len(trajs)
    init_x = check_init_size(inval=init_x, defval=kf.x, depth=2, size=num_trajs)
    init_P = check_init_size(inval=init_P, defval=kf.P, depth=3, size=num_trajs)
    init_t = check_init_size(inval=init_t, defval=kf.t, depth=1, size=num_trajs)
    if params is None:
        params = [None] * num_trajs
    else:
        params = np.asarray(params)
        if len(params.shape) < 2:
            params = np.tile(params, (1,num_trajs))

    results = []
    for traj, x0, P0, t0, par in izip(trajs, init_x, init_P, init_t, params):
        print 'Evaluating on %s...' % traj['description']
        estimator.clear()
        if par is not None:
            print 'Setting parameters to %s' % np.array_str(par, max_line_width=300)
            estimator.set_theta(par)

        kf.initialize(x=x0, P=P0, t=t0)
        obs_traj = zip(traj['obs_times'], traj['obs_values'])
        trace = execution.run_filter(estimator=estimator, kf=kf, traj=obs_traj,
                                     est_times=traj['true_times'])
        metrics = compute_metrics(trace=trace, key_func=key_func, truth=traj['true_values'])
        results.append((trace, metrics))
    return results

def hold_one_out(train_func, test_func, trajs):
    """Perform hold-one-out evaluation on a set of trials.

    Parameters
    ----------
    train_func : Function that takes a list of trajectories and produces an estimator
    test_func  : Function that takes an estimator and list of trajectories and returns a result
    trajs      : List of (obs,truth) trajectory tuples
    """
    estimators = []
    evaluations = []
    for i in range(len(trajs)):
        # Train estimator
        subset = trajs
        del subset[i]
        est = train_func(obs_trajs=subset)
        estimators.append(est)

        # Evaluate estimator
        evaluation = test_func(estimator=est, obs_trajs=[subset[i][0]], truth_trajs=[subset[i][1]])
        evaluations.append(evaluation)
    return estimators, evaluations

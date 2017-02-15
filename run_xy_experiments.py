import numpy as np
import pickle
import sys
import math
from itertools import izip, product
import matplotlib.pyplot as plt

from akf_lib.evaluation import evaluate_over_data
from akf_lib.estimator_factories import BoxEstimator, SigmoidEstimator
from experiments import ExperimentXY

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Please specify output file and files to test'
        sys.exit(-1)

    experiment = ExperimentXY(order=2)
    kf = experiment.create_filter()

    out_path = sys.argv[1]
    out_file = open(out_path, 'wb')
    files = sys.argv[2:]
    raws = [pickle.load(open(path, 'rb')) for path in files]
    trials = [experiment.process_trial(r) for r in raws]
    trial_t0s = [t['obs_times'][0] for t in trials]


    Q = 1E-4*np.identity(experiment.filter_dim)
    R = 1E-4*np.identity(experiment.obs_dim)

    # Assuming starting at zero velocity/acc with high certainty
    x0 = np.zeros(experiment.filter_dim)
    P0 = 1E-4 * np.identity(experiment.filter_dim)

    # 1. First test performance for box
    ae_box = BoxEstimator(use_diag=True,
                          buff_len=2.0,
                          Q_def=Q,
                          R_def=R)
    base_box_range = np.logspace(start=math.log10(0.05), stop=math.log10(2.0), num=2)
    box_params = np.array(list(product(base_box_range, base_box_range)))
    # ae_box.add_q_prior(dt=0.5, q=Q)
    # ae_box.add_r_prior(dt=0.5, r=R)

    box_train_results = []
    for params in box_params:
        print 'Testing parameters %s...' % np.array_str(params, max_line_width=300)
        ae_box.set_theta(params)
        results = evaluate_over_data(estimator=ae_box, kf=kf, trajs=trials,
                                     init_x=x0, init_P=P0, init_t=trial_t0s,
                                     key_func=experiment.extract_state)
        box_train_results.append(results)

    # 1.1 Find best parameters for box over all data
    box_param_metrics = [[tr[1] for tr in pset] for pset in box_train_results]
    box_param_vll = np.array([[tr['innovation_ll'] for tr in pset] for pset in box_param_metrics])
    box_param_rms = np.array([[tr['rms'] for tr in pset] for pset in box_param_metrics])
    box_param_xll = np.array([[tr['state_ll'] for tr in pset] for pset in box_param_metrics])

    avg_param_vll = np.mean(box_param_vll, axis=-1)
    avg_param_rms = np.mean(np.mean(box_param_rms, axis=-1), axis=-1)
    avg_param_xll = np.mean(box_param_xll, axis=-1)

    rankings = zip(params, avg_param_vll, avg_param_rms, avg_param_xll)
    rankings.sort(key=lambda x: x[1])
    best_vll_all = rankings[-1]
    rankings.sort(key=lambda x: x[2])
    best_rms_all = rankings[-1]
    rankings.sort(key=lambda x: x[3])
    best_xll_all = rankings[-1]
    print 'Box best VLL: %f RMS: %f XLL: %f' % (best_vll_all[1], best_rms_all[2], best_xll_all[3])

    # 2. Test performance of online AKF
    ae_akf = SigmoidEstimator(use_diag=True,
                              online_opt=True,
                              online_step=1E-1,
                              online_ss=1,
                              buff_len=2,
                              Q_def=Q,
                              R_def=R)
    ae_akf.add_q_prior(dt=2.0, q=Q)
    ae_akf.add_r_prior(dt=2.0, r=R)
    init_akf_params = ae_akf.get_theta()

    akf_results = evaluate_over_data(estimator=ae_akf, kf=kf, trajs=trials,
                                     init_x=x0, init_P=P0, init_t=trial_t0s,
                                     params=init_akf_params,
                                     key_func=experiment.extract_state)
    akf_vll = [tr[1]['innovation_ll'] for tr in akf_results]
    akf_rms = [tr[1]['rms'] for tr in akf_results]
    akf_xll = [tr[1]['state_ll'] for tr in akf_results]
    avg_akf_vll = np.mean(akf_vll)
    avg_akf_rms = np.mean(akf_rms)
    avg_akf_xll = np.mean(akf_xll)
    print 'AKF avg VLL: %f RMS: %f XLL: %f' % (avg_akf_vll, avg_akf_rms, avg_akf_xll)

    res = {'box_results' : box_train_results, 'akf_results' : akf_results,
           'akf_theta' : ae_akf.get_theta()}
    pickle.dump(res, out_file)
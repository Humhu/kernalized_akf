import pickle
import numpy as np
import akf_lib.dynamics
from akf_lib.kalman_filter import KalmanFilter

class ExperimentBase:

    def __init__(self):
        pass

    @staticmethod
    def process_trial(trial):
        """Process a trial to conform to experiment standards.
        """
        # Determine an appropriate time subset so that the first
        # observation precedes the first ground truth
        time_mask = np.logical_and(trial['true_times'] > trial['obs_times'][0],
                                   trial['true_times'] < trial['obs_times'][-1])
        trial['true_times'] = trial['true_times'][time_mask]
        trial['true_values'] = trial['true_values'][time_mask]
        return trial

class ExperimentXY(ExperimentBase):

    def __init__(self, order=2):
        ExperimentBase.__init__(self)
        self.order = order

    @property
    def state_dim(self):
        return 2

    @property
    def obs_dim(self):
        return self.state_dim

    def extract_state(self, x):
        x = np.asarray(x)
        inds = [1, 2]
        if len(x.shape) == 1:
            return x[inds]
        elif len(x.shape) == 2:
            tinds = np.atleast_2d(inds).T
            return x[tinds, inds]
        else:
            raise ValueError('Can only extract state from 1D or 2D')

    @property
    def filter_dim(self):
        return self.state_dim * (self.order+1)

    def compute_transition_matrix(self, dt):
        return akf_lib.dynamics.integrator_matrix(dt=dt, dim=self.state_dim, order=self.order)

    @property
    def observation_matrix(self):
        C = np.zeros((self.state_dim, self.filter_dim))
        C[:, 0:self.state_dim] = np.identity(self.state_dim)
        return C

    def create_filter(self):
        return KalmanFilter(Afunc=self.compute_transition_matrix, C=self.observation_matrix)

    @staticmethod
    def process_trial(trial):
        trial = ExperimentBase.process_trial(trial)
        # Extract just the linear XY components
        trial['obs_values'] = trial['obs_values'][:, 1:3]
        trial['true_values'] = trial['true_values'][:, 1:3]
        return trial
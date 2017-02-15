import math
import execution

class OnlineTrainer(object):

    def __init__(self, estimator, kf):
        self.estimator = estimator
        self.init_params = self.estimator.get_theta()

    def train(self, trajs):

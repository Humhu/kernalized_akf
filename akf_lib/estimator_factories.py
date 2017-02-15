import math
import kernels
from adaptive_estimator import AdaptiveEstimator

def BoxEstimator(q_bw=1.0, r_bw=1.0, **kwargs):
    q_box = kernels.BoxKernel(q_bw)
    r_box = kernels.BoxKernel(r_bw)
    est = AdaptiveEstimator(qkernel=q_box,
                            rkernel=r_box,
                            online_opt=False,
                            **kwargs)
    return est

def SigmoidEstimator(q_k=1.0, q_bw=1.0, r_k=1.0, r_bw=1.0, **kwargs):
    q_sig = kernels.SigmoidKernel(w=math.log(q_k), h=q_bw)
    r_sig = kernels.SigmoidKernel(w=math.log(r_k), h=r_bw)
    est = AdaptiveEstimator(qkernel=q_sig, rkernel=r_sig, **kwargs)
    return est

def ExponentialEstimator(q_k=1.0, r_k=1.0, **kwargs):
    q_exp = kernels.OneSidedSEKernel(w=math.log(q_k))
    r_exp = kernels.OneSidedSEKernel(w=math.log(r_k))
    est = AdaptiveEstimator(qkernel=q_exp, rkernel=r_exp, **kwargs)
    return est
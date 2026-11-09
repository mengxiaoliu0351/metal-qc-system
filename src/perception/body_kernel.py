"""Portable RBF binary decision head; scores are not calibrated probabilities."""
import numpy as np


def export_kernel(pipeline):
    scaler, model = pipeline.named_steps['sc'], pipeline.named_steps['clf']
    if model.classes_.tolist() != [0, 1]:
        raise ValueError('Expected normal=0, defect=1')
    return {'mean': scaler.mean_, 'scale': scaler.scale_,
            'support': model.support_vectors_, 'dual': model.dual_coef_[0],
            'intercept': model.intercept_, 'gamma': np.asarray(model._gamma)}


def kernel_margin(vectors, params):
    x = np.atleast_2d(np.asarray(vectors, dtype=float))
    x = (x - params['mean']) / params['scale']
    support = params['support']
    distance = np.maximum(0., (x*x).sum(1)[:, None]
                          + (support*support).sum(1)[None, :] - 2*x@support.T)
    return np.exp(-float(params['gamma'])*distance) @ params['dual'] + float(params['intercept'][0])


def kernel_scores(vectors, params):
    return 1/(1+np.exp(-np.clip(kernel_margin(vectors, params), -700, 700)))

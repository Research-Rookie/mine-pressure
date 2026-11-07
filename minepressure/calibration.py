"""CAL-array probability fitting and per-parameter F1/F2 threshold selection."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from .alarm_threshold import f_cut


def fit_probability_link(score, positive, known, positive_weight, *, regularization, solver_options):
    x = np.asarray(score)[known].astype(np.float64)
    y = np.asarray(positive)[known].astype(np.float64)
    assert len(x) and np.isfinite(x).all()
    prior = np.array([1., -np.log(positive_weight)])

    def objective(v):
        z = v[0]*x+v[1]
        e, d = expit(z)-y, v-prior
        loss = (np.logaddexp(0, z)-y*z).sum()+.5*regularization*(d*d).sum()
        gradient = np.array([(e*x).sum(), e.sum()])+regularization*d
        return float(loss), gradient

    opt = minimize(objective, prior, jac=True, method='L-BFGS-B',
                   bounds=[(1e-6, None), (None, None)],
                   options=solver_options)
    assert opt.success, opt.message
    return opt.x


def fit_alarm_policies(score, positive, known):
    """Arrays B,8,4; CAL only. No positive class gives an undefined cut."""
    return {
        policy: [f_cut(score[..., j], positive[..., j], known[..., j], beta)
                 if positive[..., j][known[..., j]].any() else None
                 for j in range(4)]
        for policy, beta in [('F1', 1), ('F2', 2)]
    }

# Loglinear (aka maximum entropy, maxent) string models
# implemented by Wfsts with features on arcs.
#
# References:
# * Eisner, J. (2002). Parameter estimation for probabilistic
# finite-state transducers. In Proceedings of the 40th Annual
# Meeting of the Association for Computational Linguistics (pp. 1-8).
# * Wu, K., Allauzen, C., Hall, K. B., Riley, M., & Roark, B. (2014).
# Encoding linear models as weighted finite-state transducers. In
# INTERSPEECH (pp. 1258-1262).
# * Markus Dreyer's fstrain (https://github.com/markusdr/fstrain).
# todo: stable mapping from Arcs to violation vectors

import numpy as np
from collections import ChainMap
from scipy import sparse

from pynini import Weight

from wynini import (config, Wfst, shortestdistance)


def arc_features(wfst):
    """
    Collect all features on arcs of wfst.
    """
    ftrs = set()
    fst = wfst.fst
    for q in fst.states():
        for t in fst.arcs(q):
            # Feature vector.
            phi_t = wfst.features(q, t)
            ftrs |= phi_t.keys()
    return ftrs


def violation_matrix(wfst, ftrs):
    """
    Construct sparse violation matrix.
    arg ftrs: list of loglinear features
    """
    fst = wfst.fst
    ftrs = list(ftrs)
    ftr2index = {ftr: i for i, ftr in enumerate(ftrs)}

    # CSR format [checkme].
    arc_ids = []
    ftr_ids = []
    vals = []
    arc_id = 0
    for q in fst.states():
        for t in fst.arcs(q):
            phi_t = wfst.features(q, t)
            for (ftr, val) in phi_t.items():
                arc_ids.append(arc_id)
                ftr_ids.append(ftr2index[ftr])
                vals.append(val)
            arc_id += 1

    V = sparse.csr_array( \
        (vals, (arc_ids, ftr_ids)),
        shape=(wfst.num_arcs(), len(ftrs)))

    return V, ftrs, ftr2index


def assign_weights(wfst, w):
    """
    Assign unnormalized -logprob weight to each arc t in wfst
    according to its Harmony: H(t) = -sum_k (w_k · ftr_k(t)).
    phi: arc t -> dictionary of feature values ('violations')
    {ftr_0: v_0, ftr_1: v_1, ...} (possibly empty or None).
    arg w: dictionary of feature weights {ftr_0:w_0, ftr_1:w_1, ...},
    (where missing weights are treated as 0).
    All feature values and weights should be non-negative.
    note: name clash with Wfst.assign_weights()
    todo: make identity (one) weight implicit?
    """
    wfst.map_weights('to_log')
    fst = wfst.fst
    one = Weight('log', 0.0)
    for q in fst.states():
        q_arcs = fst.mutable_arcs(q)
        for t in q_arcs:  # note: unstable arc reference
            phi_t = wfst.features(q, t)
            t.weight = Weight('log', dot_product(phi_t, w)) \
                if phi_t is not None else one
            q_arcs.set_value(t)
    return wfst


def dot_product(phi_t, w):
    """
    Dot product of features and weights represented 
    with dictionaries of non-negative values.
    """
    ret = 0.0
    if not w:  # (None or empty dict.)
        return ret
    for ftr, val in phi_t.items():
        w_ftr = w.get(ftr, 0.0)
        ret += w_ftr * val
    return ret


def assign_weights_vec(wfst, V, w):
    """
    Assign unnormalized -logprob weight to each arc t in wfst
    using violation matrix and weight vector.
    arg V: (sparse) violation matrix [narc x nftr]
    arg w: weight vector [nftr]
    """
    wfst.map_weights('to_log')
    fst = wfst.fst
    x = V @ w
    arc_id = 0
    for q in fst.states():
        q_arcs = fst.mutable_arcs(q)
        for t in q_arcs:  # note: unstable arc reference
            t.weight = Weight('log', x[arc_id])
            q_arcs.set_value(t)
            arc_id += 1
    return wfst


def expected(wfst, w=None, N=1, verbose=False):
    """
    Expected per-string violation counts of features/constraints
    given feature weights w -or- with weights already applied.
    Optionally scale expected values by data size N.
    All feature values and weights should be non-negative.
    """
    # Set arc weights equal to Harmonies
    # (sum of weighted feature violations).
    if w:
        assign_weights(wfst, w)
    if verbose:
        print(wfst.info())

    # Forward potentials (for each state q,
    # sum over all paths from initial state to q).
    alpha = shortestdistance(wfst, reverse=False)
    alpha = [float(w) for w in alpha]
    if verbose:
        print(f'alpha: {alpha}')

    # Backward potentials (for each state q,
    # sum over all paths from q to finals).
    beta = shortestdistance(wfst, reverse=True)
    beta = [float(w) for w in beta]
    if verbose:
        print(f'beta: {beta}')

    # Log partition function
    # (sum over all paths through machine).
    q0 = wfst.start_id()
    logZ = -beta[q0]
    if verbose:
        print(f'logZ: {logZ}')

    # Accumulate expected violations across arcs.
    expect = {}
    fst = wfst.fst
    for q in fst.states():
        for t in fst.arcs(q):
            # Feature vector.
            phi_t = wfst.features(q, t)
            if phi_t is None:
                continue
            # Unnormalized -logprob of all paths through t.
            plog = alpha[q] + float(t.weight) + beta[t.nextstate]
            # Convert to globally normalized probability of t.
            prob = np.exp(-plog - logZ)
            # Accumulate prob[t] * violations[t] into ftr expectation.
            for ftr, val in phi_t.items():
                expect[ftr] = expect.get(ftr, 0.0) + (prob * val)

    # Scale by data size.
    if N > 1:
        for ftr in expect:
            expect[ftr] *= N

    return expect, logZ


def logZ(wfst, w=None, verbose=False):
    """
    Log partition function given feature weights w -or-
    with weights already applied.
    All feature values and weights should be non-negative.
    """
    if w:
        assign_weights(wfst, w)

    # Backward potentials (for each state q,
    # sum over all paths from q to finals).
    beta = shortestdistance(wfst, reverse=True)
    beta = [float(w) for w in beta]
    if verbose:
        print(f'beta: {beta}')

    # Log partition function
    # (sum over all paths through machine).
    q0 = wfst.start_id()
    logZ = -beta[q0]
    if verbose:
        print(f'logZ: {logZ}')

    return logZ


def gradient(O, E, N=1, grad=None):
    """
    Negative gradient of feature weights computed from 
    dictionaries of 'observed' (aka clamped) and 
    'expected' (aka unclamped) feature counts.
    Optionally scale E by data size N to match O.
    """
    if grad is None:
        grad = {}
    # checkme: does E always cover observe?
    for ftr in ChainMap(O, E):
        O_ftr = O.get(ftr, 0.0)
        E_ftr = N * E.get(ftr, 0.0)
        grad[ftr] = grad.get(ftr, 0.0) - (O_ftr - E_ftr)
    return grad


def update(w, grad, alpha=1.0, w_min=1e-4):
    """
    Update weights in-place with neg gradient,
    learning rate alpha, and minimum weight w_min.
    todo: regularizer(s); interface to adagrad, etc.
    note: features without entries in grad are 
    not updated (except for regularization).
    """
    for ftr, g in grad.items():
        w_ftr = w.get(ftr, 0.0) + alpha * g
        w[ftr] = max(w_ftr, w_min)
    return w


def update_vec(w, grad, ftr2index, alpha=1.0, w_min=1e-4):
    """
    Vector-backed implementation of update().
    """
    for ftr, g in grad.items():
        ftr_id = ftr2index[ftr]
        w[ftr_id] += alpha * g
        w[ftr_id] = max(w[ftr_id], w_min)
    return w


def pprint_vals(vals, ftrs=None, N=1):
    """
    Print observed / expected / gradient values,
    optionally scaled by corpus size N.
    Use optional ftrs arg for to control
    feature subset or order.
    """
    ret = []
    if ftrs is None:
        ftrs = vals.keys()
    for ftr in ftrs:
        val = vals.get(ftr, 0.0)
        val = np.round(N * val, 2)
        ret.append(f'{ftr}: {val:.2f}')
    ret = '{' + ', '.join(ret) + '}'
    return ret


def print_vals(vals, ftrs=None, N=1):
    print(pprint_vals(vals, ftrs, N))

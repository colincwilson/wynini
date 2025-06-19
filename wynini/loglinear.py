import numpy as np
from collections import ChainMap
from scipy import sparse

from pynini import Weight

from wynini import config, Wfst, shortestdistance

# References:
# * Eisner, J. (2002). Parameter estimation for probabilistic
# finite-state transducers. In Proceedings of the 40th Annual
# Meeting of the Association for Computational Linguistics (pp. 1-8).
# * Wu, K., Allauzen, C., Hall, K. B., Riley, M., & Roark, B. (2014).
# Encoding linear models as weighted finite-state transducers. In
# INTERSPEECH (pp. 1258-1262).
# * Markus Dreyer's fstrain (https://github.com/markusdr/fstrain).
# todo: stable mapping from Arcs to violation vectors


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

    # CSR format. xxx check
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
    according to its Harmony: $- sum_k (w_k \cdot \phi_k(t))$.
    phi: arc t -> dictionary of feature values ('violations') {\phi_0:v_0, \phi_1:v_1, ...}
    arg w: dictionary of feature weights {\phi_0:w_0, \phi_1:w_1, ...}
    All feature values and weights should be non-negative.
    """
    wfst.map_weights('to_log')
    fst = wfst.fst
    One = Weight('log', 0.0)
    for q in fst.states():
        q_arcs = fst.mutable_arcs(q)
        for t in q_arcs:  # note: unstable arc reference
            phi_t = wfst.features(q, t)
            t.weight = Weight('log', dot_product(phi_t, w)) \
                if phi_t is not None else One
            q_arcs.set_value(t)
    return wfst


def dot_product(phi_t, w):
    """
    Dot product of features and weights represented 
    with dictionaries of non-negative values.
    """
    ret = 0.0
    for ftr, violn in phi_t.items():
        if ftr in w:
            ret += w[ftr] * violn
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


def expected(wfst, w=None):
    """
    Expected violation counts of features/constraints given 
    feature weights w -or- with weights already applied.
    All feature values and weights should be non-negative.
    """
    # Set arc weights equal to Harmonies
    # (sum of weighted feature violations).
    if w:
        assign_weights(wfst, w)

    # Forward potentials (for each state q,
    # sum over all paths from initial to q).
    alpha = shortestdistance(wfst, reverse=False)
    alpha = [float(w) for w in alpha]
    #print(alpha)

    # Backward potentials (for each state q,
    # sum over all paths from q to finals).
    beta = shortestdistance(wfst, reverse=True)
    beta = [float(w) for w in beta]
    #print(beta)

    # Partition function.
    # (sum over all paths through machine)
    logZ = -beta[0]

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
            # Accumulate prob[t] * violations[t].
            for ftr, violn in phi_t.items():
                expect[ftr] = \
                    expect.get(ftr, 0.0) + (prob * violn)

    return expect


def gradient(O_counts, E_counts, grad=None):
    """
    Neg gradient of feature weights computed from 
    dictionaries of 'observed' (aka clamped) and 
    'expected' (aka unclamped) feature counts.
    """
    if grad is None:
        grad = {}
    # checkme: does E_counts always cover O_counts?
    for ftr in ChainMap(O_counts, E_counts):
        O = O_counts.get(ftr, 0.0)
        E = E_counts.get(ftr, 0.0)
        grad[ftr] = grad.get(ftr, 0.0) - (O - E)
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
        w_ftr = w[ftr] + alpha * g
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

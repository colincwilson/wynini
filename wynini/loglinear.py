import numpy as np

from pynini import Weight

from wynini import config, Wfst, shortestdistance

# Reference:
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
            phi_t = wfst.get_features(q, t)
            ftrs |= phi_t.keys()
    return ftrs


def assign_weights(wfst, w):
    """
    Assign unnormalized -logprob weight to each arc t in wfst M 
    according to its Harmony: $-\sum_k (w_k \cdot \phi_k(t))$.
    phi: arc t -> dictionary of feature values ('violations') {\phi_0:v_0, \phi_1:v_1, ...}
    w: dictionary of feature weights {\phi_0:w_0, \phi_1:w_1, ...}
    All feature values and weights should be non-negative.
    """
    wfst.map_weights('to_log')
    fst = wfst.fst
    one = Weight('log', 0.0)
    for src in fst.states():
        q_arcs = fst.mutable_arcs(src)
        for t in q_arcs:
            phi_t = wfst.get_features(src, t)
            if phi_t:
                t.weight = Weight('log', dot_product(phi_t, w))
            else:
                t.weight = one
            q_arcs.set_value(t)
    return wfst


def expected(wfst, w=None):
    """
    Expected violation counts of features/constraints given 
    feature weights w -or- with weights already applied.
    All feature values and weights should be non-negative.
    """
    # Set arc weights equal to Harmonies
    # (sum of weighted feature violations).
    if w is not None:
        assign_weights(wfst, w)

    # Forward potentials.
    # (sum over all paths from initial to q)
    alpha = shortestdistance(wfst, reverse=False)
    alpha = [float(w) for w in alpha]
    #print(alpha)

    # Backward potentials.
    # (sum over all paths from q to finals)
    beta = shortestdistance(wfst, reverse=True)
    beta = [float(w) for w in beta]
    #print(beta)

    # Partition function.
    # (sum over all paths through machine)
    Z = np.exp(-beta[0])

    # Accumulate expected violations across arcs.
    expect = {}
    fst = wfst.fst
    for q in fst.states():
        for t in fst.arcs(q):
            # Feature vector.
            phi_t = wfst.get_features(q, t)
            if not phi_t:
                continue
            # Unnormalized -logprob of all paths through t.
            plog = alpha[q] + float(t.weight) + beta[t.nextstate]
            # Convert to globally normalized probability of t.
            prob = np.exp(-plog) / Z
            # Accumulate prob[t] * violations[t].
            for ftr, violn in phi_t.items():
                expect[ftr] = \
                    expect.get(ftr, 0.0) + (prob * violn)

    return expect


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


def gradient(O_counts, E_counts, grad=None):
    """
    Neg gradient of feature weights computed from 
    dictionaries of 'observed' (aka clamped) and 
    'expected' (aka unclamped) feature counts.
    """
    if grad is None:
        grad = {}
    for ftr, E in E_counts.items():
        grad[ftr] = grad.get(ftr, 0.0) \
                    - (O_counts.get(ftr, 0.0) - E)
    return grad


def update(w, grad, alpha=1.0, wmin=1e-3):
    """
    Update weights in-place with neg gradient,
    learning rate alpha, and minimum weight wmin.
    todo: regularizer(s)
    """
    for ftr, g in grad.items():
        w_ftr = w[ftr] + alpha * g
        w[ftr] = max(w_ftr, wmin)
    return w

import numpy as np

from pynini import Weight
from . import config
from .wfst import Wfst, shortestdistance, get_features

# todo: stable mapping from Arcs to violation vectors
# Reference:
# * Eisner, J. (2002). Parameter estimation for probabilistic
# finite-state transducers. In Proceedings of the 40th Annual
# Meeting of the Association for Computational Linguistics (pp. 1-8).
# * Wu, K., Allauzen, C., Hall, K. B., Riley, M., & Roark, B. (2014).
# Encoding linear models as weighted finite-state transducers. In
# INTERSPEECH (pp. 1258-1262).


def assign_weights(wfst, phi, w):
    """
    Assign unnormalized plog weight to each arc t in wfst M 
    according to its Harmony: $-\sum_k (w_k \cdot \phi_k(t))$.
    phi: 
    phi: arc t -> [\phi_0(t), \phi_1(t), ...] (violation vector)
    w: [w_0, w_1, ...] (weight vector)
    """
    fst = wfst.fst
    one = Weight('log', 0)
    for src in fst.states():
        q_arcs = fst.mutable_arcs(src)
        for t in q_arcs:
            phi_t = get_features(phi, src, t)
            #print((src, t.ilabel, t.olabel, t.nextstate), phi_t)
            if phi_t is None:
                t.weight = one
            else:
                t.weight = Weight('log', dot_product(phi_t, w))
            q_arcs.set_value(t)
    return wfst


def expected(wfst, phi, w):
    """
    Expected violation counts of features/constraints
    in phi given weights w.
    """
    # Set arc weights equal to Harmonies
    # (sum of weighted feature violations).
    assign_weights(wfst, phi, w)

    # Forward potentials
    # (sum over all paths from initial to q).
    alpha = shortestdistance(wfst, reverse=False)
    alpha = [float(w) for w in alpha]
    #print(alpha)

    # Backward potentials
    # (sum over all paths from q to finals).
    beta = shortestdistance(wfst, reverse=True)
    beta = [float(w) for w in beta]
    #print(beta)

    # Accumulate expected violations across arcs.
    expect = {}
    fst = wfst.fst
    for q in fst.states():
        for t in fst.arcs(q):
            # Feature vector.
            phi_t = get_features(phi, q, t)
            if phi_t is None or len(phi_t) == 0:
                continue
            # Unnormalized plog of all paths through t.
            plog = alpha[q] + float(t.weight) + beta[t.nextstate]
            # Accumulate pstar[t] * violations[t].
            pstar = np.exp(-plog)
            for ftr, violn in phi_t.items():
                expect[ftr] = expect.get(ftr, 0.0) + (pstar * violn)

    # Divide by partitition function (sum over all paths)
    Z = np.exp(-beta[0])
    for ftr in expect:
        expect[ftr] /= Z
    return expect


def dot_product(phi_t, w):
    """
    Dot product of features and weights represented 
    with dictionaries of non-zero values.
    """
    ret = 0.0
    for ftr, violn in phi_t.items():
        if ftr in w:
            ret += w[ftr] * violn
    return ret

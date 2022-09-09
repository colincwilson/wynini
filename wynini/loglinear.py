# -*- coding: utf-8 -*-

import numpy as np

from pynini import Weight, shortestdistance
from . import config
from .wfst import Wfst

# todo: stable mapping from Arcs to violation vectors


def loglinear_weights(wfst, phi, w):
    """
    Assign unnormalized plog weight to each arc t in wfst 
    according to its Harmony: $-\sum_k (w_k \cdot \phi_k(t))$.
    phi: arc t -> [\phi_0(t), \phi_1(t), ...] (violation vector)
    w: [w_0, w_1, ...] (weight vector)
    """
    fst = wfst.fst
    one = Weight('log', 0)
    for q in fst.states():
        q_arcs = fst.mutable_arcs(q)
        for t in q_arcs:
            _t = (q, t.ilabel, t.olabel, t.nextstate)
            if _t not in phi:
                t.weight = one
            else:
                phi_t = phi.get(_t)
                t.weight = Weight('log', np.dot(phi_t, w))
            q_arcs.set_value(t)
    return wfst


def loglinear_expected(wfst, phi, w):
    """
    Expected violation counts of features/constraints in phi 
    given weights w.
    """
    # Compute arc weights from Harmonies
    wfst = loglinear_weights(wfst, phi, w)
    fst = wfst.fst

    # Forward potentials (sum over all paths from initial to q)
    alpha = shortestdistance(fst, reverse=False)
    alpha = [float(w) for w in alpha]
    #print(alpha)

    # Backward potentials (sum over all paths from q to finals)
    beta = shortestdistance(fst, reverse=True)
    beta = [float(w) for w in beta]
    #print(beta)

    # Accumulate expected violations across arcs
    n = w.shape[0]
    expect = np.zeros(n)
    for q in fst.states():
        for t in fst.arcs(q):
            _t = (q, t.ilabel, t.olabel, t.nextstate)
            if _t not in phi:  # all-zero violation vector
                continue
            phi_t = phi.get(_t)  # violation vector
            # plog of all paths through t
            plog = alpha[q] + float(t.weight) + beta[t.nextstate]
            expect += np.exp(-plog) * phi_t

    # Divide by partitition function (sum over all paths)
    expect /= np.exp(-beta[0])
    return expect

# -*- coding: utf-8 -*-

import sys

sys.path.append('..')
from wynini.wfst import *
from wynini.loglinear import *

# Simple acceptor
M = trellis_acceptor(max_len=2)
print(M.print())

# Map from arc to violation vector
phi = {}
for q in M.fst.states():
    for t in M.fst.arcs(q):
        _t = (q, t.ilabel, t.olabel, t.nextstate)
        if M.ilabel(t) == 'a':
            phi[_t] = np.array([1.0, 0.0])  # *a, 0
        if M.ilabel(t) == 'b':
            phi[_t] = np.array([0.0, 1.0])  # 0, *b
print('phi:', phi)

# Constraint weights (non-negative)
w = np.array([1.0, 2.0])  # *a, *b
print('w:', w)

# Loglinear arc weights
loglinear_weights(M, phi, w)
print(M.print(acceptor=True, show_weight_one=True))

# Expected constraint violations
expect = loglinear_expected(M, phi, w)
print('E:', expect)
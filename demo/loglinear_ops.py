# Demo loglinear / maxent computations.
import sys
import numpy as np

from wynini import (config, loglinear)
from wynini.wywrapfst import *

config.init()

syms = ['a', 'b']

# Acceptor for observed strings.
D = acceps(['a b', 'b a'], arc_type='standard')
D.print(acceptor=True, show_weight_one=True)
D.draw('fig/D.dot', fig='png', show_weight_one=True)

# Acceptor for possible strings.
M = trellis(length=2,
            isymbols=syms,
            trellis=False,
            add_delim=False,
            arc_type='log')
M.print(acceptor=True, show_weight_one=True)
M.draw('fig/M.dot', fig='png')


# Map from arc to violation vector.
def phi_func(wfst, q, t):
    if wfst.ilabel(t) == 'a':
        return {'*a': 1.}
    if wfst.ilabel(t) == 'b':
        return {'*b': 1.}
    return None


D.assign_features(phi_func)
M.assign_features(phi_func)

# Observed constraint violations.
observe = loglinear.expected(D, w=None, verbose=True)
print('Observed: ', end='')
loglinear.print_expected(observe, N=1)

print('Arc violation vectors:')
for _t in M.phi:
    print('\t', _t, '->', M.phi[_t])

# Constraint weights (non-negative).
w = {'*a': 1.0, '*b': 1.0}
print('Constraint weights:', w)

# Loglinear arc weights.
loglinear.assign_weights(M, w)
M.print(acceptor=True, show_weight_one=True)

# Expected per-string constraint violations.
expect = loglinear.expected(M, w, verbose=True)
print('Expected: ', end='')
loglinear.print_expected(expect, N=1)

# Gradient.
grad = loglinear.gradient(observe, expect, grad=None)
print(f'Gradient: {grad}')

sys.exit(0)

# # # # # # # # # #

# Composition of two transducers with features.
M1 = trellis(length=2, isymbols=['a', 'b'], arc_type='log')
print('M1:')
M1.print(show_weight_one=True)  # acceptor = True
M1.draw('fig/M1.dot')

organize_arcs(M1, side='output')
#sys.exit(0)


def phi1_func(wfst, q, t):
    if wfst.ilabel(t) == 'a':
        return {'*a': 1.0}
    if wfst.ilabel(t) == 'b':
        return {'*b': 1.0}
    return None


M1.assign_features(phi1_func)
print('M1.phi:', M1.phi)
print()

isymbols2, _ = config.make_symtable(['a', 'b'])
osymbols2, _ = config.make_symtable(['A', 'B', 'C'])
M2 = Wfst(isymbols2, osymbols2, arc_type='log')
M2.add_state(0)
M2.add_state(1)
M2.add_state(2)
M2.add_arc(0, config.bos, config.bos, None, 1)
for x in ['a', 'b']:
    for y in ['A', 'B', 'C']:
        M2.add_arc(1, x, y, None, 1)
M2.add_arc(1, config.eos, config.eos, None, 2)
M2.set_initial(0)
M2.set_final(2)
print('M2:')
M2.print()
M2.draw('fig/M2.dot')


def phi2_func(wfst, q, t):
    ret = {}
    if wfst.ilabel(t) != wfst.olabel(t):
        ret['Ident'] = 1.0
    if wfst.olabel(t) == 'B':
        ret['*B'] = 1.0
    if len(ret) == 0:
        return None
    return ret


M2.assign_features(phi2_func)
print('M2.phi:', M2.phi)
print()

M = compose(M1, M2, verbose=True)
print('M:\n')
M.print()
print(M.phi)
M.draw('fig/M.dot')

print('M.phi:')
for t, v in M.phi.items():
    print(t, v)
print()

w = {'*a': 1.0, '*b': 2.0, '*Ident': 5.0, '*B': 6.0}
M = loglinear.assign_weights(M, w)
M.print(show_weight_one=True)
E = loglinear.expected(M, w)
print(E)

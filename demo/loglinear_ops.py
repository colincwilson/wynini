import sys
import numpy as np

sys.path.append('..')
from wynini.wfst import *
from wynini import loglinear

# Simple acceptor.
M = trellis(length=2, arc_type='log')
print(M.print(acceptor=True, show_weight_one=True))


# Map from arc to violation vector.
def phi_func(M, q, t):
    ret = {}
    if M.ilabel(t) == 'a':
        ret['*a'] = 1.0
    elif M.ilabel(t) == 'b':
        ret['*b'] = 1.0
    return ret


phi = assign_features(M, phi_func)

print('Arc violation vectors:')
for _t in phi:
    print('\t', _t, '->', phi[_t])

# Constraint weights (non-negative).
w = {'*a': 1.0, '*b': 2.0}
print('Constraint weights:', w)

# Loglinear arc weights.
loglinear.assign_weights(M, phi, w)
print(M.print(acceptor=True, show_weight_one=True))

# Expected constraint violations.
expect = loglinear.expected(M, phi, w)
print('Expected:', expect)
print()

# # # # # # # # # #

# Composition of two transducers with features.
M1 = trellis(length=2, sigma=['a', 'b'], arc_type='log')
print(M1.print(acceptor=True, show_weight_one=True))


def phi1_func(M, q, t):
    ret = {}
    if M.ilabel(t) == 'a':
        ret['*a'] = 1.0
    elif M.ilabel(t) == 'b':
        ret['*b'] = 1.0
    return ret


phi1 = assign_features(M1, phi1_func)
print('phi1:', phi1)
#print(phi1[(1, 3, 3, 2)])
print()

isymbols2, _ = config.make_symtable(['a', 'b'])
osymbols2, _ = config.make_symtable(['a', 'b', 'c'])
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
print(M2.print())


def phi2_func(M, q, t):
    ret = {}
    if M.ilabel(t) != M.olabel(t):
        ret['Ident'] = 1.0
    if M.olabel(t) == 'B':
        ret['*B'] = 1.0
    return ret


phi2 = assign_features(M2, phi2_func)
print('phi2:', phi2)
print()

M, phi = compose(M1, M2, phi1=phi1, phi2=phi2)
print('phi:')
for t, v in phi.items():
    print(t, v)

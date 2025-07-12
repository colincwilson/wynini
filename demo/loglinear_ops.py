# Demo loglinear / maxent computations.
import sys
import numpy as np

import wynini
from wynini import (config, acceps, trellis)
from wynini import loglinear
#from wynini.wywrapfst import *

config.init()

syms = ['p', 't', 'k', 'a']

# Acceptor for observed strings.
dat = [ \
    'p a', 't a', 'k a',
    'p a t', 'p a k',
    't a p', 't a k',
    'k a p', 'k a t' ]
N = len(dat)
D = acceps(dat, add_delim=True, arc_type='standard')
print(dat)
D.print(acceptor=True, show_weight_one=True)
D.draw('fig/D.dot', fig='png', show_weight_one=True)

# Acceptor for possible strings.
M = trellis(length=3,
            isymbols=syms,
            trellis=True,
            add_delim=True,
            arc_type='log')
M.print(acceptor=True, show_weight_one=True)
M.draw('fig/M.dot', fig='png')

# Map from arc to violation vector.
ftrs = ['*p', '*t', '*a']


def phi_func(wfst, q, t):
    if wfst.ilabel(t) == 'p':
        return {'*p': 1.}
    if wfst.ilabel(t) == 't':
        return {'*t': 1.}
    if wfst.ilabel(t) == 'a':
        return {'*a': 1.}
    return None


D.assign_features(phi_func)
M.assign_features(phi_func)

# Observed constraint violations.
#D.map_weights('to_log')
#D.assign_weights(lambda D, q, t: 0.)
observe, logZ = loglinear.expected(D, w=None, verbose=False)
print('Observed: ', end='')
loglinear.print_ftrs(observe, ftrs)
print(f'logZ = {logZ}, Z = {np.exp(logZ)}')

print('Arc violation vectors:')
for _t in M.phi:
    print(f'\t{_t} -> {M.phi[_t]}')

# Initial constraint weights (non-negative).
w = {'*p': 1., '*t': 1., '*a': 1.}
print('Constraint weights:', w)

nstep = 5  # Number of gradient updates.
alpha = 0.5  # Learning rate.
for _ in range(nstep):
    # Loglinear arc weights.
    loglinear.assign_weights(M, w)

    # Observed constraint violations.
    print('Observed: ', end='')
    loglinear.print_ftrs(observe, ftrs)

    # Expected per-string constraint violations scaled by N.
    expect, _ = loglinear.expected(M, w, N, verbose=False)
    print('Expected: ', end='')
    loglinear.print_ftrs(expect, ftrs)

    # Gradient.
    grad = loglinear.gradient(observe, expect, grad=None)
    print(f'Gradient: ', end='')
    loglinear.print_ftrs(grad, ftrs)

    # Constraint weight update.
    loglinear.update(w, grad, alpha)
    print(f'w: ', end='')
    loglinear.print_ftrs(w, ftrs)

    print()

# # # # # # # # # #

# N-gram features with tiers.

M_local = wynini.ngram(isymbols=syms, arc_type='log')
M_local.draw('fig/M_local.dot', fig='png')

M_cons = wynini.ngram(isymbols=syms, tier=['p', 't', 'k'], arc_type='log')
M_cons.draw('fig/M_cons.dot', fig='png')

M = wynini.compose(M_local, M_cons)
M.draw('fig/M.dot', fig='pdf')

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

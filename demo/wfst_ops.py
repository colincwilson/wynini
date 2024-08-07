import sys

sys.path.append('..')
from pynini import Arc, Weight
from wynini import config as wfst_config
from wynini.wfst import *

# # # # # # # # # #
# Alphabet (+ epsilon, bos, eos)
config = {'sigma': ['a', 'b', 'c'], 'special_syms': ['λ']}
wfst_config.init(config)

# Weighted finite-state machine
# acyclic, accepts ⋊(a|b)⋉
M = Wfst(wfst_config.symtable)
for q in [0, 1, 2, 3, 4, 5]:
    M.add_state(f'q{q}')
M.set_start('q0')
M.set_final('q4')
M.add_arc(src='q0', ilabel=wfst_config.bos, dest='q1')
M.add_arc(src='q1', ilabel='a', dest='q2')
M.add_arc(src='q1', ilabel='b', dest='q3')
M.add_arc(src='q2', ilabel=wfst_config.eos, dest='q4')
M.add_arc(src='q3', ilabel=wfst_config.eos, dest='q4')
M.add_arc(src='q1', ilabel='c', dest='q5')  # dead arc
M = M.connect()

print(M.print(acceptor=True, show_weight_one=True))
M.draw('M.dot')
# dot -Tpdf M.dot > M.pdf

# States
print('states:', list(M.states()))  # all states
print('finals:', list(M.finals()))  # final states
print('state id to label:', M._state2label)
for id in [0, 1]:
    print(f'is q{id} start?', M.is_start(f'q{id}'))
for id in [4, 0]:
    print(f'final weight of q{id}:', M.final(f'q{id}'))

# Arcs
print('number of arcs:', M.num_arcs())
print('arc type:', M.arc_type())
print('weight type:', M.weight_type())

# Paths
print('input strings:', list(M.istrings()))

# Transduce
inpt = 'a'
outpt = M.transduce(inpt)
#print(O.print(acceptor=True, show_weight_one=True))
print(list(outpt))

# Transduce (alternative method)
inpt = '⋊ a ⋉'
wfst_in = accep(inpt, add_delim=False)
print(wfst_in.print(acceptor=True))
wfst_out = compose(wfst_in, M)
print(list(wfst_out.ostrings()))

# Copy
M2 = M.copy()
print(M2.print(acceptor=True, show_weight_one=True))

# # # # # # # # # #
print('Braid acceptor')
config = {'sigma': ['a', 'b', 'c', 'd']}
wfst_config.init(config)
B = braid(length=2, tier=set(['a', 'b']))
print(B.print(acceptor=True))
B.draw('B.dot')
print()

# # # # # # # # # #
print('Trellis acceptor')
config = {'sigma': ['a', 'b', 'c', 'd']}
wfst_config.init(config)
T = trellis(length=2, tier=set(['a', 'b']))
print(T.print(acceptor=True))
T.draw('T.dot')
print()

# # # # # # # # # #
# Assign arc weights with arbitrary function


def wfunc(wfst, src, arc):
    #w_good = Weight('log', 1.0)  # -log2(0.5)
    #w_bad = Weight('log', 2.0)  # -log2(0.25)
    w_good = 1.0
    w_bad = 2.0
    if wfst.olabel(arc) == 'a':
        return w_good
    return w_bad


print('Trellis with assigned weights')
T_weight = T.map_weights('to_log')
T_weight.assign_weights(wfunc)
print(T_weight.print(acceptor=True))
T_weight.draw('T_weight.dot')
print()

# # # # # # # # # #
print('Ngram machines')
config = {'sigma': ['a', 'b'], 'special_syms': ['λ']}
wfst_config.init(config)
L = ngram(context='left', length=2)
L.draw('L.dot')
R = ngram(context='right', length=2)
R.draw('R.dot')
LR = ngram(context='both', length=(2, 1))
LR.draw('LR.dot')

print('Accepted strings (up to maximum length)')
print(list(L.accepted_strings(side='input', weights=True, max_len=4)))
print()

# # # # # # # # # #
# Transduction / composition
config = {'sigma': ['a', 'b']}
wfst_config.init(config)

# Machine that accepts [ab]*
M1 = Wfst(wfst_config.symtable)
q = 0
M1.add_state(q)
M1.set_start(q)
M1.set_final(q)
for x in wfst_config.sigma:
    M1.add_arc(src=q, ilabel=x, dest=q)

# Machine that accepts ab*
M2 = Wfst(wfst_config.symtable)
for q in [0, 1]:
    M2.add_state(q)
M2.set_start(0)
M2.set_final(1)
M2.add_arc(src=0, ilabel='a', dest=1)
M2.add_arc(src=1, ilabel='b', dest=1)

# Compose / intersect + trim
M12 = compose(M1, M2)
M12.draw('M12.dot')
print()

# # # # # # # # # #
print('Weighted transduction / composition')
I = accep('a b', arc_type='log')
I.assign_weights(lambda wfst, q, t: Weight('log', 2))
print(I.print(show_weight_one=True))
I.draw('I.dot')

M = ngram(context='left', length=1, arc_type='log')
M.assign_weights(lambda wfst, q, t: Weight('log', 3))
print(M.print(show_weight_one=True))
M.draw('M.dot')

O = compose(I, M)
print(O.print(show_weight_one=True))
O.draw('O.dot')
print()

# # # # # # # # # #
# Shortest distance / shortest paths.
M = Wfst(wfst_config.symtable, arc_type='log')
q0 = M.add_state(initial=True)
q1 = M.add_state()
qf = M.add_state(final=True)
M.add_arc(q0, 'a', 'a', Weight('log', 0.1), q1)
M.add_arc(q0, 'b', 'b', Weight('log', 0.1), qf)
print(M.print())

dist = shortestdistance(M, reverse=True)
print('Shortest distances:')
for q in M.fst.states():
    print(q, dist[q])

print('\nShortest paths:')
M.map_weights('to_tropical')
S = shortestpath_(M)
print(S.print())

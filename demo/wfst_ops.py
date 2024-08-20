import sys

sys.path.append('..')
from pynini import Arc, Weight
from wynini import config as wyconfig
from wynini.wywrapfst import *

# # # # # # # # # #
# Alphabet (+ epsilon, bos, eos)
config = {'sigma': ['a', 'b', 'c'], 'special_syms': ['λ']}
wyconfig.init(config)

# Weighted finite-state machine
# acyclic, accepts ⋊(a|b)⋉
M = Wfst(wyconfig.symtable)
for q in [0, 1, 2, 3, 4, 5]:
    M.add_state(f'q{q}')
M.set_start('q0')
M.set_final('q4')
M.add_arc(src='q0', ilabel=wyconfig.bos, dest='q1')
M.add_arc(src='q1', ilabel='a', dest='q2')
M.add_arc(src='q1', ilabel='b', dest='q3')
M.add_arc(src='q2', ilabel=wyconfig.eos, dest='q4')
M.add_arc(src='q3', ilabel=wyconfig.eos, dest='q4')
M.add_arc(src='q1', ilabel='c', dest='q5')  # dead arc
M = M.connect()

print(M.print(acceptor=True, show_weight_one=True))
M.draw('fig/M.dot')
# dot -Tpdf fig/M.dot > fig/M.pdf

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
wyconfig.init(config)
B = braid(length=2, tier=set(['a', 'b']))
print(B.print(acceptor=True))
B.draw('fig/B.dot')
print()

# # # # # # # # # #
print('Trellis acceptor')
config = {'sigma': ['a', 'b', 'c', 'd']}
wyconfig.init(config)
T = trellis(length=2, tier=set(['a', 'b']))
print(T.print(acceptor=True))
T.draw('fig/T.dot')
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
T_weight.draw('fig/T_weight.dot')
print()

# # # # # # # # # #
print('Ngram machines')
config = {'sigma': ['a', 'b'], 'special_syms': ['λ']}
wyconfig.init(config)
L = ngram(context='left', length=2)
L.draw('fig/L.dot')
R = ngram(context='right', length=2)
R.draw('fig/R.dot')
LR = ngram(context='both', length=(2, 1))
LR.draw('fig/LR.dot')

print('Accepted strings (up to maximum length)')
print(list(L.accepted_strings(side='input', weights=True, max_len=4)))
print()

# # # # # # # # # #
# Transduction / composition
config = {'sigma': ['a', 'b']}
wyconfig.init(config)

# Machine that accepts (a|b)*
M1 = Wfst(wyconfig.symtable)
q = 0
M1.add_state(q)
M1.set_start(q)
M1.set_final(q)
for x in wyconfig.sigma:
    M1.add_arc(src=q, ilabel=x, dest=q)

# Machine that accepts ab*
M2 = Wfst(wyconfig.symtable)
for q in [0, 1]:
    M2.add_state(q)
M2.set_start(0)
M2.set_final(1)
M2.add_arc(src=0, ilabel='a', dest=1)
M2.add_arc(src=1, ilabel='b', dest=1)

# Compose / intersect + trim
M12 = compose(M1, M2)
M12.draw('fig/M12.dot')
print()

# # # # # # # # # #
print('Weighted transduction / composition')
I = accep('a b', arc_type='log')
I.assign_weights(lambda wfst, q, t: Weight('log', 2))
print(I.print(show_weight_one=True))
I.draw('fig/I.dot')

M = ngram(context='left', length=1, arc_type='log')
M.assign_weights(lambda wfst, q, t: Weight('log', 3))
print(M.print(show_weight_one=True))
M.draw('fig/M.dot')

O = compose(I, M)
print(O.print(show_weight_one=True))
O.draw('fig/O.dot')
print()

# # # # # # # # # #
print('Weighted transduction / composition with pre-organized arcs')
M_arcs = organize_arcs(M, side='input')
O = compose(I, M, wfst2_arcs=M_arcs)
print(O.print(show_weight_one=True))
print()
sys.exit(0)

# # # # # # # # # #
# Shortest distance / shortest paths.
M = Wfst(wyconfig.symtable, arc_type='log')
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
S = shortestpath(M, ret_type='ostrings')
#print(S.print())
print(S)
S = shortestpath_(M)
print(S.print())

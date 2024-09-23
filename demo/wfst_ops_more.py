import sys

sys.path.append('..')
from pynini import Arc, Weight
from wynini import config as wyconfig
from wynini.wywrapfst import *

# # # # # # # # # #
# Alphabet (+ epsilon, bos, eos)
config = { \
    'sigma': ['a', 'b', 'c', 'd', 'e']}
symtable, syms = wyconfig.init(config)
epsilon = wyconfig.epsilon
print(syms)

print('\nWeighted composition with sorted arcs')
M1 = Wfst(symtable)
M1.add_state(0, initial=True)
M1.add_state(1)
M1.add_state(2)
M1.add_state(3, final=True)
M1.add_arc(0, 'b', 'b', None, 2)
M1.add_arc(0, 'a', 'a', None, 1)
M1.add_arc(0, 'a', 'a', None, 2)
M1.add_arc(1, 'c', 'c', None, 3)
M1.add_arc(2, 'c', 'c', None, 3)
M1.print()

M2 = Wfst(symtable)
M2.add_state(0, initial=True, final=True)
for x in ['a', 'b', 'c', 'd', 'e']:
    M2.add_arc(0, x, x, None, 0)
M2.print()

M1 = M1.arcsort('olabel')
M1.print()
for q in M1.state_ids():
    for t in M1.arcs(q):
        print(M1.print_arc(q, t))
print()

M2 = M2.arcsort('ilabel')
M2.print()
for q in M2.state_ids():
    for t in M2.arcs(q):
        print(M1.print_arc(q, t))
print()

M = compose_sorted(M1, M2)
print(M.info())
M.print()

print('\nTest epsilon-matching filter.')
# Mohri, Pereira, & Riley (2004), Fig. 8.
M1 = Wfst(symtable)
q0 = M1.add_state(initial=True)
q1 = M1.add_state()
q2 = M1.add_state()
q3 = M1.add_state()
q4 = M1.add_state(final=True)
M1.add_arc(q0, 'a', 'a', None, q1)
M1.add_arc(q1, 'b', epsilon, None, q2)
M1.add_arc(q2, 'c', epsilon, None, q3)
M1.add_arc(q3, 'd', 'd', None, q4)

M2 = Wfst(symtable)
q0 = M2.add_state(initial=True)
q1 = M2.add_state()
q2 = M2.add_state()
q3 = M2.add_state(final=True)
M2.add_arc(q0, 'a', 'd', None, q1)
M2.add_arc(q1, epsilon, 'e', None, q2)
M2.add_arc(q2, 'd', 'a', None, q3)

M1 = M1.arcsort(sort_type='olabel')
M2 = M2.arcsort(sort_type='ilabel')
M = compose_sorted(M1, M2)
print(M.info())
print(M)

import sys

sys.path.append('..')
from pynini import Arc, Weight
from wynini import config as wyconfig
from wynini.wywrapfst import *

# # # # # # # # # #
# Alphabet (+ epsilon, bos, eos)
config = { \
    'sigma': ['a', 'b', 'c', 'd']}
symtable, syms = wyconfig.init(config)
print(syms)

print("Weighted composition with sorted arcs")
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
for x in ['a', 'b', 'c', 'd']:
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

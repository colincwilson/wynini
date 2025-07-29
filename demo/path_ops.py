# Determinize acceptors and transducers.
import sys

#sys.path.append('..')
import wynini
from wynini import Wfst
from wynini import config as wyconfig

wyconfig.init()
M = Wfst(isymbols=['a', 'b', 'c', 'd'])
q0 = M.add_state(initial=True)
q1 = M.add_state()
q2 = M.add_state(final=True)
M.add_arc(q0, 'a', 'a', None, q1)
M.add_arc(q1, 'b', 'b', None, q2)

for x in M.istrings():
    print(x)

for x in M.ostrings():
    print(x)

print(list(M.istrings()))

print(list(M.path_items()))

# path_iter = M.paths().istrings()

# #print(path_iter)
# for x in path_iter:
#     print(x)

# for i in range(5):
#     if x is None:
#         break
#     print(i, x)
#     path_iter.next()

#print(list(path_iter.items()))

#print(M.istrings())

# Determinize acceptors and transducers.
import sys

sys.path.append('..')
import wynini
from wynini import Wfst
from wynini import config as wyconfig

wyconfig.init()
M = Wfst(isymbols=['a', 'b', 'c', 'd'])
M.add_state(0, initial=True, final=False)
for q in range(1, 5):
    M.add_state(q, final=False)
M.add_state(5, final=True)
M.add_arc(0, 'a', None, None, 1)
M.add_arc(1, 'b', None, None, 5)
M.add_arc(0, 'a', None, None, 2)
M.add_arc(2, 'b', None, None, 5)
M.add_arc(2, 'c', None, None, 5)
M.add_arc(0, 'a', None, None, 3)
M.add_arc(3, wyconfig.epsilon, None, None, 4)
M.add_arc(4, 'c', None, None, 5)
M.add_arc(4, 'd', None, None, 5)
print(M)

M_det = M.determinize()
print(M_det)

M_encode, _ = M.encode_labels()
print(M_encode)

M_det = M.determinize(acceptor=False)
print(M_det)

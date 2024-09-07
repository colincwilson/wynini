import sys

sys.path.append('..')
from wynini.simple_fst import *

fst = SimpleFst()
#fst.Q.add(0)
#fst.q0 = 0
#fst.F.add(0)
fst.set_start(0)
fst.set_final(0)
fst.set_final(1)
t = SimpleArc(0, 'a', 'b', 0)
fst.add_arc(t)
t = SimpleArc(0, 'c', 'd', 1)
fst.add_arc(t)
fst.print()

fst2 = fst.to_wfst()
fst2.print(acceptor=True, show_weight_one=True)

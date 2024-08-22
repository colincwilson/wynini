# Globally normalize acyclic machine over log semiring
# and generate random samples

import sys
import numpy as np
from pynini import Weight

#sys.path.append('..')
from wynini import config as wyconfig
from wynini.wywrapfst import *

# Alphabet.
config = {'sigma': ['a', 'b', 'c', 'd', 'e']}
wyconfig.init(config)

# Weighted acceptor.
M = Wfst(wyconfig.symtable, arc_type='log')
for q in [0, 1, 2, 3, 4, 5]:
    M.add_state(q)
M.set_start(0)
M.set_final(5, Weight('log', 1.0))

w = -np.log(0.5)  # plog(1/2)
M.add_arc(src=0, ilabel=wyconfig.bos, weight=Weight('log', w), dest=1)
M.add_arc(src=1, ilabel='a', weight=Weight('log', w), dest=2)
M.add_arc(src=1, ilabel='b', weight=Weight('log', w), dest=3)
M.add_arc(src=2, ilabel='c', weight=Weight('log', w), dest=4)
M.add_arc(src=2, ilabel='d', weight=Weight('log', w), dest=4)
M.add_arc(src=3, ilabel='e', weight=Weight('log', w), dest=4)
M.add_arc(src=4, ilabel=wyconfig.eos, weight=Weight('log', 1.0), dest=5)
M.print(acceptor=True, show_weight_one=True)
M.draw('fig/M.dot')

# Push weights toward initial state.
M_push = M.copy().push_weights()
M_push.print(acceptor=True, show_weight_one=True)
M_push.draw('fig/M_push.dot')

# Generate random sample of accepted strings.
samp = M_push.randgen(npath=100, select='log_prob')
print(list(samp))

# Push weights toward initial state.
dist = shortestdistance(M, reverse=True)
print(dist)
M_push = M.copy().reweight(dist)
M_push.print(acceptor=True, show_weight_one=True)

import sys
import numpy as np
from pathlib import Path
from pynini import Weight

sys.path.append(str(Path.home() / 'Code/Python/fst_util'))
from fst_util import config as wfst_config
from fst_util.wfst import *


def test():
    # Weighted acceptor
    config = {'sigma': ['a', 'b', 'c', 'd', 'e']}
    wfst_config.init(config)
    M = Wfst(wfst_config.symtable, arc_type='log')
    for q in [0, 1, 2, 3, 4, 5]:
        M.add_state(q)
    M.set_start(0)
    M.set_final(5, Weight('log', 1.0))
    w = -np.log(0.5)  # plog(1/2)
    M.add_arc(src=0, ilabel=wfst_config.bos, weight=Weight('log', w), dest=1)
    M.add_arc(src=1, ilabel='a', weight=Weight('log', w), dest=2)
    M.add_arc(src=1, ilabel='b', weight=Weight('log', w), dest=3)
    M.add_arc(src=2, ilabel='c', weight=Weight('log', w), dest=4)
    M.add_arc(src=2, ilabel='d', weight=Weight('log', w), dest=4)
    M.add_arc(src=3, ilabel='e', weight=Weight('log', w), dest=4)
    M.add_arc(src=4, ilabel=wfst_config.eos, weight=Weight('log', 1.0), dest=5)
    print(M.weight_type())
    print(M.print(acceptor=True, show_weight_one=True))
    M.draw('M.dot')

    # Push weights toward initial state
    M_push = M.push_weights()
    print(M_push.weight_type())
    print(M_push.print(acceptor=True, show_weight_one=True))
    M_push.draw('M_push.dot')

    # Random sample
    M_sample = M_push.sample(100, select='log_prob')
    print(list(M_sample.istrings()))


if __name__ == '__main__':
    test()

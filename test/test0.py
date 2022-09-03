import sys
from pathlib import Path

sys.path.append(str(Path.home() / 'Code/Python/fst_util'))
from fst_util import config as wfst_config
from fst_util.wfst import *


def test():
    # State labels
    config = {'sigma': [], 'special_syms': ['λ']}
    wfst_config.init(config)
    M = Wfst(wfst_config.symtable)
    for sym in ['λ', '⋊', 'A', 'B']:
        M.add_state(sym)
    M.set_start('λ')
    for sym in ['A', 'B']:
        M.set_final(sym)
    M.add_arc(src='λ', ilabel='⋊', dest='⋊')
    M.add_arc(src='⋊', ilabel='a', dest='A')
    M.add_arc(src='⋊', ilabel='b', dest='B')
    M.add_arc(src='A', ilabel='a', olabel='a', dest='A')
    M.add_arc(src='A', ilabel='a', olabel='b', dest='B')
    print(M.print(acceptor=True, show_weight_one=True))
    M.draw('tmp.dot')

    M2 = M.copy()
    print(M2.print(acceptor=True, show_weight_one=True))

    # Left- and right- context acceptors
    config = {'sigma': ['a', 'b'], 'special_syms': ['λ']}
    wfst_config.init(config)
    L = left_context_acceptor(context_length=2)
    L.draw('L.dot')
    R = right_context_acceptor(context_length=2)
    R.draw('R.dot')

    # Accepted strings
    print(accepted_strings(L, 'input', 4))

    # Connect (state labels preserved)
    C = Wfst(wfst_config.symtable)
    qf = C.add_state('0')
    q = C.add_state('1')
    q0 = C.add_state('2')
    C.set_start(q0)
    C.set_final(qf)
    #C.add_arc(src=q0, ilabel='a', dest=q)
    C.add_arc(src=q0, ilabel='a', dest=qf)
    C.add_arc(src=q, ilabel='b', dest=qf)
    print(C._state2label)
    C_trim = C.connect()
    print(C_trim._state2label)
    C_trim.draw('C_trim.dot')

    # Composition
    config = {'sigma': ['a', 'b']}
    wfst_config.init(config)
    M1 = Wfst(wfst_config.symtable)  # a*b*
    for q in [0, 1]:
        M1.add_state(q)
    M1.set_start(0)
    M1.set_final(1)
    for x in wfst_config.sigma:
        M1.add_arc(src=0, ilabel=x, dest=0)
        M1.add_arc(src=0, ilabel=x, dest=1)
        M1.add_arc(src=1, ilabel=x, dest=1)

    M2 = Wfst(wfst_config.symtable)  # ab*
    for q in [0, 1]:
        M2.add_state(q)
    M2.set_start(0)
    M2.set_final(1)
    for x in wfst_config.sigma:
        M2.add_arc(src=0, ilabel=x, dest=0)
        M2.add_arc(src=0, ilabel=x, dest=1)
    M = compose(M1, M2)
    M.draw('M.dot')

    # Arc deletion
    config = {'sigma': ['a', 'b']}
    wfst_config.init(config)
    M = Wfst(wfst_config.symtable)
    for q in [0, 1]:
        M.add_state(q)
    M.set_start(0)
    M.set_final(1)
    M.add_arc(src=0, ilabel='a', dest=1)
    M.add_arc(src=0, ilabel='b', dest=1)
    print(M.print())


if __name__ == '__main__':
    test()
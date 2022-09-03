import sys
from pathlib import Path

sys.path.append(str(Path.home() / 'Code/Python/fst_util'))
from fst_util import config as wfst_config
from fst_util.simple_fst import *


def test():
    fst = SimpleFst()
    #fst.Q.add(0)
    #fst.q0 = 0
    #fst.F.add(0)
    fst.set_start(0)
    fst.set_final(0)
    t = SimpleArc(0, 'a', 'b', 0)
    fst.add_arc(t)
    t = SimpleArc(0, 'c', 'd', 1)
    fst.add_arc(t)
    print(fst.print())

    fst2 = fst.to_wfst()
    print(fst2.print())


if __name__ == '__main__':
    test()
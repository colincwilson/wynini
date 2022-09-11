import sys

sys.path.append('..')
from pynini import Weight
from wynini.wfst import *

# Make a machine that represents each segment in its immediately preceding context:
M = ngram(context='left', length=1)

#  Place the arc weights of the machine in the neg log prob semiring (called "log" by pynini/OpenFst)
M.map_weights(map_type='to_log')

#  Assign arc weights with an arbitrary function; for example, disprefer aa and bb


def wfunc(wfst, src, arc):
    if (wfst.ilabel(arc) == wfst.state_label(src)[0]):
        return Weight('log', 3.0)  # violation
    return Weight('log', 0.0)  # no violation


M.assign_weights(wfunc)

#  Make a machine that represents all strings of length n
n = 4
A = braid(n)
A.map_weights(map_type='to_log')

#  Compose A o M, normalize, and sample strings of length n
S = compose(A, M)
S = S.push_weights()
S.draw('S.dot')

samp = S.randgen(npath=10)
print(list(samp))
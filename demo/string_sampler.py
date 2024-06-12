import sys

sys.path.append('..')
from pynini import Weight
from wynini import config
from wynini.wfst import *

config.init()

# 'Ngram' machine that represents each segment in its immediately
# preceding context, placing arc weights in the neg log prob
# semiring (called "log" by pynini/OpenFst)
M = ngram(context='left', length=1, arc_type='log')

# Assign arc weights with an arbitrary function; for example,
# disprefer sequences of identicals (aa and bb)


def wfunc(wfst, src, arc):
    if (wfst.ilabel(arc) == wfst.state_label(src)[0]):
        return Weight('log', 5.0)  # violation
    return Weight('log', 0.0)  # no violation


M.assign_weights(wfunc)

#  Machine that represents all strings of length n
n = 4
A = braid(n, arc_type='log')

#  Compose A o M, normalize, and sample strings of length n
S = compose(A, M)
S = S.push_weights()
S.draw('S.dot')

samp = S.randgen(npath=10)
print(list(samp))

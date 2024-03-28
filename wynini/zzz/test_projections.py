#! /usr/bin/python

import itertools
import string
import sys

import FSMGlobal
import fsm

#FSMGlobal.Sigma = ['a', 'b', 'c']
#projections = [FSMGlobal.Sigma, ['a','b']]
FSMGlobal.Sigma = list(string.ascii_lowercase)
projections = [FSMGlobal.Sigma, ['i', 'u', 'e', 'o', 'a']]

M0 = fsm.create_context_fsm(projections[0])
print M0.left_context
print M0.right_context
print len(M0.Q), len(M0.T)

M1 = fsm.create_context_fsm(projections[1])
print M1.left_context
print M1.right_context
print len(M1.Q), len(M1.T)

A = fsm.trellis(2)
print len(A.Q), len(A.T)
M = fsm.intersect(M0, A)

M = fsm.intersect(M, M1)
print len(M.Q), len(M.T)

sys.exit(0)

M = fsm.flatten(M)
Q, T, q0, qf, _, _ = M
#print Q
print len(M[0]), len(M[1])

fsm.dot(M, '/Users/colin/Desktop/tmp.dot')

# left context and right context maps for each projection xxx here only default projection
proj_id = 0
left_context = {}
right_context = {}
for x in projections[proj_id]:
    # all states with the meaning 'just seen x'
    left_context[x] = [q for q in M.Q if q[proj_id] in M0.left_context[x]]
    # all states with the meaning 'about to see x'
    right_context[x] = [q for q in M.Q if q[proj_id] in M0.right_context[x]]
print left_context
print right_context

sys.exit(0)


# find all states and transitions corresponding to given context on a projection
left, right = 'a', 'b'
proj_id = 0
Q_left = [q for q in M.Q if q[proj_id] in M0.left_context[left]]; #print Q_left
T_left = [t for t in M.T if t.dest in Q_left]; #print T_left
labels_left = set([t.label for t in T_left]); print labels_left

Q_right = [q for q in M.Q if q[proj_id] in M0.right_context[right]]; #print Q_right
T_right= [t for t in M.T if t.src in Q_right]; #print T_right
labels_right = set([t.label for t in T_right]); print labels_right

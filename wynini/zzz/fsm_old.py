from collections import namedtuple
import sys

import FSMGlobal
verbosity = 0

FSM = namedtuple('FSM', ['Q', 'T', 'q0', 'qf', 'left_context', 'right_context'])
Transition = namedtuple('Transition', ['src', 'label', 'dest'])

# # # # # # # # # #
# create FSM with states that encode one-elt left and right contexts
# - each state q corresponds to a (left_context, right_context) pair,
#       where left_context labels all incoming transitions
#       and right_context labels all outgoing transitions
def create_context_fsm(projection=None):
    word_begin = FSMGlobal.word_begin
    word_end = FSMGlobal.word_end
    Sigma = FSMGlobal.Sigma
    if projection is None:
        projection = Sigma

    # Part I
    # states and transitions labeled with segments on projection
    q0 = (None, word_begin)
    Q = set([q0])
    T = set([])

    Qnew = set(Q)
    while len(Qnew)!=0:
        Qold = set(Qnew); Qnew.clear()
        for q in Qold:
            x = q[-1]    # label of all transitions out of q
            for y in projection:    # one segment look-ahead
                r = (x,y)
                T.add(Transition(q, x, r))
                if r in Q or r in Qnew: continue
                Q.add(r); Qnew.add(r)
                if verbosity>0: print (q, x, r)

    # transitions to unique final state
    qf = (word_end, None)
    for q in set(Q):
        x = q[-1]
        r = (x, word_end)
        Q.add(r)
        T.add(Transition(q, x, r))
        T.add(Transition(r, word_end, qf))
        if verbosity>0: print (q, x, r); print (r, word_end, qf)
    Q.add(qf)

    # self-transitions labeled with segments *not* on projection
    for q in Q:
        if q[-1]==word_begin: continue
        for x in [x for x in Sigma if x not in projection]:
            T.add(Transition(q, x, q))
            if verbosity>0: print (q, x, q)

    # Part II   xxx move outside?
    # relabel states with integer ids, creating maps from left and
    # right contexts to lists of states instantiating the contexts
    Qlist = list(Q)
    Qindex = { Qlist[i]:i for i in xrange(len(Qlist)) }

    left_context = {x:[] for x in projection + [word_begin, word_end]}
    right_context = {x:[] for x in projection + [word_begin, word_end]}
    for q in Q:
        (left, right) = q
        if left is not None: left_context[left].append(Qindex[q])
        if right is not None: right_context[right].append(Qindex[q])

    Q = set(Qindex.values())
    q0 = Qindex[q0]
    qf = Qindex[qf]

    # relabel transitions
    T_new = set([])
    for t in T:
        T_new.add(Transition(Qindex[t.src], t.label, Qindex[t.dest]))
    T = T_new

    return FSM(Q, T, q0, qf, left_context, right_context)


# # # # # # # # # #
# create trellis for strings of length 0 to max_len (ignoring boundaries)
def trellis(max_len):
    word_begin = FSMGlobal.word_begin
    word_end = FSMGlobal.word_end
    Sigma = FSMGlobal.Sigma

    Q, T = set([]), set([])
    q0 = 0; Q.add(q0)
    q1 = 1; Q.add(q1)
    T.add(Transition(q0, word_begin, q1))

    qe = max_len+1; Q.add(qe)
    qf = max_len+2; Q.add(qf)
    T.add(Transition(qe, word_end, qf))

    for i in xrange(max_len):
        q = i + 1
        r = i + 2
        for x in Sigma:
            Q.add(r)
            T.add(Transition(q, x, r))
        T.add(Transition(q, word_end, qf))
    return FSM(Q, T, q0, qf, None, None)


# # # # # # # # # #
# intersect two machines
def intersect(M1, M2):
    print 'intersecting ...'
    Q_1, T_1, q0_1, qf_1 = M1.Q, M1.T, M1.q0, M1.qf
    Q_2, T_2, q0_2, qf_2 = M2.Q, M2.T, M2.q0, M2.qf

    Q = set([])
    T = set([])
    q0 = (q0_1, q0_2)
    qf = (qf_1, qf_2)
    Q.add(q0); Q.add(qf)

    Qnew = set([q0])
    while len(Qnew)!=0:
        Qold = set(Qnew); Qnew.clear()
        print Qold
        for q in Qold:
            q1, q2 = q
            if verbosity>0: print q, '-->', q1, 'and', q2
            T_q1 = [t for t in T_1 if t.src==q1]
            if verbosity>0: print T_q1
            for t1 in T_q1:
                x = t1.label
                T_q2 = [t for t in T_2 if t.src==q2 and t.label==x]
                if verbosity>0: print T_q2
                for t2 in T_q2:
                    r = (t1.dest, t2.dest)
                    T.add(Transition(q, x, r))
                    Qnew.add(r)
        Q.update(Qnew)
    print len(Q), len(T)
    M = trim(FSM(Q, T, q0, qf, None, None))
    return M


# # # # # # # # # #
# make FSM trim (apply after intersection)
def trim(M):
    print 'trimming ...'
    # forward pass
    Qdiscov = set([M.q0])
    Qnew = set([M.q0])
    while len(Qnew)!=0:
        print len(Qnew),
        Qold = set(Qnew); Qnew.clear()
        for q in Qold:
            q_T = [t for t in M.T if t.src==q]
            for t in q_T:
                r = t.dest
                if r in Qdiscov or r in Qnew: continue
                Qdiscov.add(r)
                Qnew.add(r)

    Q = set(Qdiscov)
    T = set([t for t in M.T if t.src in Qdiscov and t.dest in Qdiscov])

    # backward pass
    Qdiscov = set([M.qf])
    Qnew = set([M.qf])
    while len(Qnew)!=0:
        Qold = set(Qnew); Qnew.clear()
        for r in Qold:
            r_T = [t for t in T if t.dest==r]
            for t in r_T:
                q = t.src
                if q in Qdiscov or q in Qnew: continue
                Qdiscov.add(q)
                Qnew.add(q)

    Q &= Qdiscov
    T = set([t for t in T if t.src in Qdiscov and t.dest in Qdiscov])

    M = FSM(Q, T, q0, qf, M.left_context, M.right_context)
    return M


# # # # # # # # # #
# flatten states of FSM (apply after intersection), see http://stackoverflow.com/questions/3204245/how-do-i-convert-a-tuple-of-tuples-to-a-one-dimensional-list-using-list-comprehe
def flatten_state(q): return sum(q[0:-1], ()) + (q[-1],)
def flatten(M):
    Q, T, q0, qf = M.Q, M.T, M.q0, M.qf
    Q = set([flatten_state(q) for q in Q])
    T = set([Transition(flatten_state(t.src), t.label, flatten_state(t.dest)) for t in T])
    q0 = flatten_state(q0)
    qf = flatten_state(qf)
    return FSM(Q, T, q0, qf, M.left_context, M.right_context)


# # # # # # # # # #
# print FSM in dot format
def dot(M, fname):
    Q, T, q0, qf = M.Q, M.T, M.q0, M.qf
    Q_list = list(Q)
    stateid = {Q_list[i]:i for i in xrange(len(Q_list))}
    f = open(fname, 'w')
    f.write('digraph G {\n')
    f.write('rankdir=LR;\n')
    f.write('node [shape=circle]\n')
    f.write(str(stateid[q0]) +' [style=bold]\n')
    f.write(str(stateid[qf]) +' [shape=doublecircle]\n')
    for t in T:
        f.write(str(stateid[t.src]) +' -> '+ str(stateid[t.dest]) +' [label=\"' + str(t.label) + '\"]\n')
    f.write('}\n')

if False:
    verbosity = 1
    FSMGlobal.Sigma = ['a', 'b', 'c', 'd']
    #projections = [Sigma, ['a','b']]    # segmental projection must be first!
    projections = [Sigma, ['a','b']]
    [Q, T, q0, qf] = FSM(Sigma, projections[1])

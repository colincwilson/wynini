# -*- coding: utf-8 -*-

import itertools, re, string, sys
from collections import namedtuple
import fst_config as config

verbosity = 0

FST = namedtuple('FST', ['Q', 'T', 'q0', 'qf'])
# State set, transition set, initial state, final state set


class Transition():

    def __init__(self,
                 src=None,
                 ilabel=None,
                 olabel=None,
                 weight=None,
                 dest=None):
        self.src = src
        self.ilabel = ilabel if ilabel is not None else olabel
        self.olabel = olabel if olabel is not None else ilabel
        self.weight = weight
        self.dest = dest

    def __str__(self):
        return '('+ str(self.src) +','+ str(self.ilabel) \
                +','+ str(self.olabel) +','+ str(self.dest) +')'

    def __repr__(self):
        return self.__str__()


def intersect(M1, M2):
    """
    Intersect two FSTs, retaining contextual info from  
    the original machines by explicitly representing 
    each state q in the intersection as a pair (q1,q2)
    todo: matcher options as in OpenFST
    """
    Q1, T1, q0_1, qf_1 = \
        M1.Q, M1.T, M1.q0, M1.qf
    Q2, T2, q0_2, qf_2 = \
        M2.Q, M2.T, M2.q0, M2.qf

    # Index transitions by source state in each machine
    transition_map1 = {q1: [] for q1 in Q1}
    transition_map2 = {q2: [] for q2 in Q2}
    for t1 in T1:
        transition_map1[t1.src].append(t1)
    for t2 in T2:
        transition_map2[t2.src].append(t2)

    # States, transitions, initial state, final states
    # of intersection
    Q = set()
    T = set()
    q0 = (q0_1, q0_2)
    qf = {(q1, q2) for q1 in qf_1 for q2 in qf_2}
    Q.add(q0)
    Q |= qf

    # Lazy state and transition construction
    Qold, Qnew = set(), {q0}
    while len(Qnew) != 0:
        Qold, Qnew = Qnew, Qold
        Qnew.clear()
        for q in Qold:
            q1, q2 = q
            for t1 in transition_map1[q1]:
                for t2 in transition_map2[q2]:
                    if t2.olabel != t1.olabel:
                        continue
                    r = (t1.dest, t2.dest)
                    T.add(Transition(src=q, olabel=t1.olabel, dest=r))
                    if r not in Qnew and r not in Q:
                        Qnew.add(r)
        Q.update(Qnew)
    M = FST(Q, T, q0, qf)
    M_trim = connect(M)
    return M_trim


def connect(M):
    """
    Remove dead states and transitions from FST M
    """
    # Forward pass
    forward_transitions = {q: [] for q in M.Q}
    for t in M.T:
        forward_transitions[t.src].append(t)

    Qforward = {M.q0}
    Qold, Qnew = set(), {M.q0}
    while len(Qnew) != 0:
        Qold, Qnew = Qnew, Qold
        Qnew.clear()
        for q in Qold:
            for t in forward_transitions[q]:
                r = t.dest
                if r not in Qforward:
                    Qforward.add(r)
                    Qnew.add(r)

    Q = Qforward.copy()
    T = {t for t in M.T \
            if (t.src in Q) and (t.dest in Q)}

    # Backward pass
    backward_transitions = {q: [] for q in Q}
    for t in T:
        backward_transitions[t.dest].append(t)

    Qbackward = {q for q in M.qf}
    Qold, Qnew = set(), {q for q in M.qf}
    while len(Qnew) != 0:
        Qold, Qnew = Qnew, Qold
        Qnew.clear()
        for r in Qold:
            for t in backward_transitions[r]:
                q = t.src
                if q not in Qbackward:
                    Qbackward.add(q)
                    Qnew.add(q)

    Q &= Qbackward
    T = {t for t in T \
             if t.src in Q and t.dest in Q}

    q0 = M.q0 if M.q0 in Q else None
    qf = {q for q in M.qf if q in Q}
    M_trim = FST(Q, T, q0, qf)
    return M_trim


def linear_acceptor(x):
    """
    Linear acceptor for space-delimited string x
    """
    Q = {0}
    T = set()
    x = x.split(' ')
    for i in range(len(x)):
        Q.add(i + 1)
        T.add(Transition(src=i, olabel=x[i], dest=i + 1))
    M = FST(Q, T, 0, {len(x)})
    return M


def trellis(max_len):
    """
    Trellis for strings of length 0 to max_len
    (not counting begin/end delimiters)
    """
    bos = config.bos
    eos = config.eos
    Sigma = config.Sigma

    Q, T = set(), set()
    q0 = 0
    Q.add(q0)
    q1 = 1
    Q.add(q1)
    T.add(Transition(src=q0, olabel=bos, dest=q1))

    qe = max_len + 1
    Q.add(qe)
    qf = max_len + 2
    Q.add(qf)
    T.add(Transition(src=qe, olabel=eos, dest=qf))

    for i in range(max_len):
        q = i + 1
        r = i + 2
        for x in Sigma:
            Q.add(r)
            T.add(Transition(src=q, olabel=x, dest=r))
        T.add(Transition(src=q, olabel=eos, dest=qf))
    return FST(Q, T, q0, {qf})


def map_states(M, f):
    """
    Apply function f to each state (e.g., to simplify 
    state labels after intersection)
    """
    Q = {f(q) for q in M.Q}
    T = { Transition(src = f(t.src), olabel = t.olabel, dest = f(t.dest)) \
            for t in M.T }
    q0 = f(M.q0)
    qf = {f(q) for q in M.qf}
    return FST(Q, T, q0, qf)


def accepted_strings(M, max_len):
    """
    Output strings accepted by machine up to 
    maximum length (not counting begin/end delimiters)
    """
    transition_map = {q: [] for q in M.Q}
    for t in M.T:
        transition_map[t.src].append(t)

    prefixes = {(M.q0, '')}
    prefixes_new = prefixes.copy()
    for i in range(max_len + 2):
        prefixes_old = set(prefixes_new)
        prefixes_new = set()
        for prefix in prefixes_old:
            for t in transition_map[prefix[0]]:
                prefixes_new.add((t.dest, prefix[1] + ' ' + t.olabel))
        prefixes |= prefixes_new
        #print(i, prefixes_new); print()

    accepted = { prefix for (state, prefix) in prefixes \
                    if re.search(config.eos+'$', prefix) }
    accepted = {re.sub('^ ', '', prefix) for prefix in accepted}
    return accepted


def draw(M, fname, Sigma_tier=None):
    """
    Write FST to file in dot/graphviz format
    """
    Q, T, q0, qf = \
        M.Q, M.T, M.q0, M.qf
    if Sigma_tier is None:
        Sigma_tier = config.Sigma
    Sigma_skip = config.Sigma - Sigma_tier

    # Write header
    f = open(fname, 'w')
    f.write('digraph G {\n')
    f.write('rankdir=LR;\n')

    # Write states
    stateid = {q: i for i, q in enumerate(Q)}
    for q in stateid:
        if isinstance(q, tuple):
            label = ' '.join([str(x) for x in q])
        else:
            label = str(q)
        style = 'bold' if q == q0 \
            else 'solid'
        shape = 'doublecircle' if q in qf \
            else 'circle'
        f.write(f'{str(stateid[q])} [style={style}, '
                f'shape={shape}, label=\"{label}\"]\n')

    # Write transitions
    for t in T:
        try:
            label = str(t.olabel) if t.ilabel == t.olabel \
                else str(t.ilabel) +':'+ str(t.olabel)
            style = 'dotted' if t.olabel in Sigma_skip \
                else 'solid'
            f.write(f'{str(stateid[t.src])} -> {str(stateid[t.dest])} '\
                    f'[style={style}, label=\"{label}\"]\n')
        except:
            pass
    f.write('}\n')

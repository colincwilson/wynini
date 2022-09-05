# -*- coding: utf-8 -*-
import re, sys
import pynini

from . import fst_config as config
from .fst_config import *
from .simple_fst import *

config.verbosity = 0


def prefix_tree(D, Sigma=None, Lambda=None):
    """
    Given training data D = {(x,y) | f(x) = y}, create a prefix tree transducer as in Chandlee (2014:116), Chandlee, Eyraud & Heinz (2014:497) (similar to de la Higuera, Algorithm 18.1, originally Oncina et al. 1993). Strings in D should *not* terminate in eos, which is added as part of ptt construction.
    """
    Q = set()
    F = set()
    T = set()

    # Input and output alphabets
    if Sigma is None or Lambda is None:
        Sigma, Lambda = set(), set()
        for (x, y) in D:
            Sigma |= set(x.split())
            Lambda |= set(y.split())

    # Non-final states
    Q.add(λ)
    Q |= set([u for (x, _) in D for u in prefixes(x)])
    #Q = list(Q)
    #Q.sort(key=lambda q: (len(q.split()), q))

    # Transitions with empty outputs
    for qa in Q:
        if qa == λ:
            continue
        q_a = qa.split()
        q, a = ' '.join(q_a[:-1]), q_a[-1]
        T.add(Transition(src=q, ilabel=a, olabel=λ, dest=qa))

    # Final states and incoming transitions
    for (x, y) in D:
        qf = concat(x, eos)
        Q.add(qf)
        F.add(qf)
        T.add(Transition(src=x, ilabel=eos, olabel=y, dest=qf))

    fst = SimpleFst(Q, λ, F, T)
    return fst, Sigma, Lambda


def onward_tree(fst, q, u):
    """
    Make prefix tree transducer onward, as in Chandlee, Eyraud & Heinz (2014:498) (similar to de la Higuera, definition 18.2.1 and Algorithm 18.2; in an onward transducer "the output is assigned to the transitions in such a way as to be produced as soon as we have enough information to do so."). 
    Initial call: onward_prefix_tree(fst, q0, λ); note u is not used
    """
    fst.T = list(fst.T)  # Allow modification of transitions
    fst, _, _ = _onward_tree(fst, q, u)
    fst.T = set(fst.T)
    return fst


def _onward_tree(fst, q, u):
    print(f'q = {q}, u = {u}')
    for t in filter(lambda t: t.src == q, fst.T):
        _, _, w = _onward_tree(fst, t.dest, t.ilabel)
        t.olabel = concat(t.olabel, w)

    F = [t.olabel for t in filter(lambda t: t.src == q, fst.T)]
    f = lcp(F)

    if f != λ and q != λ:  # Aksënova
        for t in filter(lambda t: t.src == q, fst.T):
            t.olabel = delete_prefix(t.olabel, f)

    return fst, q, f


def prefixes(u):
    """
    Set of prefixes of string u
    (set includes empty string λ)
    de la Higuera, p. 50
    """
    val = None
    if u == unk:
        val = set()
    else:
        u = u.split(' ')
        val = set([' '.join(u[:i]) for i in range(len(u) + 1)])
    return val


def concat(u, a):
    """
    Separator- and lambda- aware concatentation of strings u and a
    """
    if u == unk or a == unk:  # de la Higuera, section 18.2.1
        ua = unk
    elif a == λ:  # Definition of empty string
        ua = u
    elif u == λ:
        ua = a
    else:
        ua = u + ' ' + a
    if config.verbosity > 0:
        if u == λ:
            u = 'λ'
        report(f'concat: (u = {u}, a = {a}) => {ua}')
    return ua


def delete_prefix(x, u):
    """
    Separator-aware deletion of prefix u from string x
    """
    if config.verbosity > 0:
        _x = 'λ' if x == λ else x
        _u = 'λ' if u == λ else u

    if x == λ or x == unk:
        if config.verbosity > 0:
            report(f'delete prefix: (x = {_x}, u = {_u}) => {_x}')
        return x
    if u == λ:
        if config.verbosity > 0:
            report(f'delete prefix: (x = {_x}, u = {_u}) => {_x}')
        return x

    if not re.search(f'^{u}', x):
        raise Error(f'<{_u}> not a prefix of <{_x}>')

    xs = x.split(' ')
    us = u.split(' ')
    if len(us) < len(xs):
        y = ' '.join(xs[len(us):])
    else:
        y = λ
    if config.verbosity > 0:
        report(f'delete prefix: (x = {_x}, u = {_u}) => {y}')
    return y


def suffix(x, k):
    """
    Separator-aware extraction of length-k suffix from string x
    """
    x = x.split(' ')
    n = len(x)
    if n < k:
        sfx = ' '.join(x)
    else:
        sfx = ' '.join(x[(n - k):])
    report(f'suffix: (x = {x}, k = {k}) => {sfx}')
    return sfx


def lcp(F):
    """
    Longest common prefix of a list of strings
    """
    if len(F) == 0 or λ in F:
        return λ
    if len(F) == 1:
        return F[0]  # incl. unk; de la Higuera, errata
    F = [x for x in F if x != unk]  # ignore ⊥, de la Higuera section 18.2.1

    #f = os.path.commonprefix(F)
    #f = f.strip()
    #return f

    F = [x.split(' ') for x in F]
    n = min([len(x) for x in F])
    f = F[0][:n]
    for i in range(n):
        # Process matching symbols
        if len(set(x[i] for x in F)) == 1:
            continue
        # Trim after first mismatch
        f = F[0][:i]
        break
    f = ' '.join(f)
    report(f'lcp({F}) => {f}')
    return f


def lcs(F):
    """
    Longest common suffix of a list of strings
    """
    F = [x for x in F if x != unk]  # ignore ⊥
    if len(F) == 0 or λ in F:
        return λ
    if len(F) == 1:
        return F[0]

    F = [x.split(' ') for x in F]
    n = min([len(x) for x in F])
    f = F[0][-n:]
    for i in range(-1, -(n + 1), -1):
        # Process matching symbols
        if len(set(x[i] for x in F)) == 1:
            continue
        # Trim before first mismatch
        if i == -1:
            f = λ
        else:
            f = F[0][(i + 1):]
        break
    f = ' '.join(f)
    report(f'lcs: {F} => {f}')
    return f


def clean_string(x):
    x = re.sub('⊥', '', x)
    x = re.sub('[ ]+', ' ', x)
    x = x.strip()
    return x


def report(msg, level=10, end=None):
    if config.verbosity >= level:
        if end is not None:
            print(msg, end=end)
        else:
            print(msg)
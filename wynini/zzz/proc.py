# -*- coding: utf-8 -*-

import re, sys
import pynini

from . import fst_config as config
from .fst_config import *
from .fst import Fst

config.verbosity = 0


def prefix_tree(D, Sigma=None, Lambda=None):
    """
    Given training data D = {(x,y) | f(x) = y}, create a prefix tree transducer as in Chandlee (2014:116), Chandlee, Eyraud & Heinz (2014:497) (similar to de la Higuera, Algorithm 18.1, originally Oncina et al. 1993). Strings in D should *not* terminate in eos, which is added as part of ptt construction.
    """
    symtable = pynini.SymbolTable()
    fst = Fst(symtable)

    # Input and output alphabets
    if Sigma is None or Lambda is None:
        Sigma, Lambda = set(), set()
        for (x, y) in D:
            Sigma |= set(x.split())
            Lambda |= set(y.split())

    # Non-final states
    Q = set([λ])
    Q |= set([u for (x, _) in D for u in prefixes(x)])
    Q = list(Q)
    Q.sort(key=lambda q: (len(q.split()), q))
    for q in Q:
        fst.add_state(q)
    fst.set_start(λ)
    #print(fst._state2label)

    # Transitions with empty outputs
    for qa in Q:
        if qa == λ:
            continue
        q_a = qa.split()
        q, a = ' '.join(q_a[:-1]), q_a[-1]
        fst.add_arc(src=q, ilabel=a, olabel=λ, dest=qa)

    # Final states and incoming transitions
    for (x, y) in D:
        qf = concat(x, eos)
        fst.add_state(qf)
        fst.set_final(qf)
        #print(f'{x}, {eos}, {y}, {qf}')
        fst.add_arc(src=x, ilabel=eos, olabel=y, dest=qf)

    fst = fst.connect()
    return fst, Sigma, Lambda


def onward_tree(fst, q, u):
    """
    Make prefix tree transducer onward, as in Chandlee, Eyraud & Heinz (2014:498) (similar to de la Higuera, definition 18.2.1 and Algorithm 18.2; in an onward transducer "the output is assigned to the transitions in such a way as to be produced as soon as we have enough information to do so."). 
    Initial call: onward_prefix_tree(fst, q0, λ); note u is not used
    """
    #print(f'q = {fst.state_label(q)}, u = {u}')

    q_arcs = fst.mutable_arcs(q)
    for t in q_arcs:
        _, _, w = onward_tree(fst, t.nextstate, t.ilabel)
        olabel_str = fst.output_symbols().find(t.olabel)
        #print(f'concat: {olabel_str}, {w} ->', end=' ')
        olabel_str = concat(olabel_str, w)
        #print(f'{olabel_str}')
        t.olabel = fst.mutable_output_symbols().add_symbol(olabel_str)
        q_arcs.set_value(t)

    F = [t.olabel for t in fst.arcs(q)]
    F = [fst.output_symbols().find(x) for x in F]
    f = lcp(F)
    #print(F, f)

    if f != λ and fst.state_label(q) != λ:  # Aksënova
        q_arcs = fst.mutable_arcs(q)
        for t in q_arcs:
            olabel_str = fst.output_symbols().find(t.olabel)
            #print(f'delete prefix: {olabel_str}, {f} ->', end=' ')
            olabel_str = delete_prefix(olabel_str, f)
            #print(f'{olabel_str}')
            t.olabel = fst.mutable_output_symbols().add_symbol(olabel_str)
            q_arcs.set_value(t)

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
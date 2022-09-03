# -*- coding: utf-8 -*-

import sys
import pynini
from pynini import Fst, Arc, Weight
from . import config


class Wfst():
    """
    Pynini Fst wrapper with automatic handling of labels for inputs / outputs / states and output strings. State labels must be hashable (strings, tuples, ...). Pynini constructive and destructive operations generally lose track of state ids and symbol labels, so some operations are reimplemented here (e.g., connect).
    """

    def __init__(self,
                 input_symtable=None,
                 output_symtable=None,
                 arc_type='standard'):
        if input_symtable is None:
            input_symtable = pynini.SymbolTable()
            input_symtable.add_symbol(config.epsilon)
            input_symtable.add_symbol(config.bos)
            input_symtable.add_symbol(config.eos)
        if output_symtable is None:
            output_symtable = input_symtable
        fst = Fst(arc_type)
        fst.set_input_symbols(input_symtable)
        fst.set_output_symbols(output_symtable)
        self.fst = fst  # Wrapped Fst
        self._state2label = {}  # State id -> state label
        self._label2state = {}  # State label -> state id
        self.sigma = {}  # State id -> output string

    # Input/output labels.

    def input_symbols(self):
        """ Get input symbol table. """
        return self.fst.input_symbols()

    def output_symbols(self):
        """ Get output symbol table. """
        return self.fst.output_symbols()

    def input_label(self, sym):
        """ Get input label for symbol id. """
        return self.fst.input_symbols().find(sym)

    def input_index(self, sym):
        """ Get input id for symbol label. """
        return self.fst.input_symbols().find(sym)

    def output_label(self, sym):
        """ Get output label for symbol id. """
        return self.fst.output_symbols().find(sym)

    def output_index(self, sym):
        """ Get output id for symbol label. """
        return self.fst.output_symbols().find(sym)

    # States.

    def add_state(self, label=None):
        """ Add new state, optionally specifying its label. """
        # Enforce unique labels
        if label is not None:
            if label in self._label2state:
                return self._label2state[label]
        # Create new state
        q = self.fst.add_state()
        # Self-labeling by string as default
        if label is None:
            q = str(q)
        # State <-> label
        self._state2label[q] = label
        self._label2state[label] = q
        return q

    def states(self, labels=True):
        """ Iterator over state labels (or ids). """
        fst = self.fst
        if not labels:
            return fst.states()
        return map(lambda q: self.state_label(q), fst.states())

    def set_start(self, q):
        """ Set start state by id or label. """
        if not isinstance(q, int):
            q = self.state_id(q)
        return self.fst.set_start(q)

    def start(self, label=True):
        """ Start state label (or id). """
        if not label:
            return self.fst.start()
        return self.state_label(self.fst.start())

    def is_start(self, q):
        """ Check start status by id or label. """
        if not isinstance(q, int):
            q = self.state_id(q)
        return q == self.fst.start()

    def set_final(self, q, weight=None):
        """ Set final weight of state by id or label. """
        if not isinstance(q, int):
            q = self.state_id(q)
        if weight is None:
            weight = Weight.one(self.weight_type())
        return self.fst.set_final(q, weight)

    def is_final(self, q):
        """ Check final status by id or label. """
        if not isinstance(q, int):
            q = self.state_id(state)
        zero = Weight.zero(self.weight_type())
        return self.final(q) != zero

    def final(self, q):
        """ Final weight of state by id or label. """
        if not isinstance(q, int):
            q = self.state_id(q)
        return self.fst.final(q)

    def finals(self, label=True):
        """
        Iterator over labels or ids of states with non-zero final weights.
        """
        fst = self.fst
        zero = pynini.Weight.zero(fst.weight_type())
        state_iter = fst.states()
        state_iter = filter(lambda q: fst.final(q) != zero, state_iter)
        if label:
            state_iter = map(lambda q: self.state_label(q), state_iter)
        return state_iter

    def state_label(self, q):
        """ State label from id. """
        return self._state2label[q]

    def state_id(self, q):
        """ State id from label. """
        return self._label2state[q]

    # Arcs.

    def add_arc(self,
                src=None,
                ilabel=None,
                olabel=None,
                weight=None,
                dest=None):
        """ Add arc (accepts int or string args) """
        fst = self.fst
        if not isinstance(src, int):
            src = self.state_id(src)
        if olabel is None:
            olabel = ilabel
        if not isinstance(ilabel, int):
            ilabel = fst.mutable_input_symbols().add_symbol(ilabel)
        if not isinstance(olabel, int):
            olabel = fst.mutable_output_symbols().add_symbol(olabel)
        if weight is None:
            weight = Weight.one(self.weight_type())
        if not isinstance(dest, int):
            dest = self.state_id(dest)
        arc = Arc(ilabel, olabel, weight, dest)
        return fst.add_arc(src, arc)

    def arcs(self, src):
        """ Iterator over arcs from a state. """
        # todo: decorate with labels if requested
        if not isinstance(src, int):
            src = self.state_id(src)
        return self.fst.arcs(src)

    def mutable_arcs(self, src):
        """ Mutable iterator over arcs from a state. """
        if not isinstance(src, int):
            src = self.state_id(src)
        return self.fst.mutable_arcs(src)

    def num_arcs(self):
        """ Total count of arcs. """
        fst = self.fst
        val = 0
        for q in fst.states():
            val += fst.num_arcs(q)
        return val

    def arc_type(self):
        """ Arc type (standard, ...). """
        return self.fst.arc_type()

    def weight_type(self):
        """ Weight type (tropical, log, log64). """
        return self.fst.weight_type()

    def map_weights(self, map_type='identity'):
        """ Map weights to a different semiring. """
        if map_type == 'identity':
            return self
        self.fst = pynini.arcmap(self.fst, map_type=map_type)
        return self

    # Algorithms.

    def istrings(self):
        """
        Iterator over input strings of paths through this machine (assumed to be acyclic).
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        strpath_iter = fst.paths(input_token_type=isymbols)
        return strpath_iter.istrings()

    def ostrings(self):
        """
        Iterator over output strings of paths through this machine (assumed acyclic).
        """
        fst = self.fst
        osymbols = fst.output_symbols()
        strpath_iter = fst.paths(output_token_type=osymbols)
        return strpath_iter.ostrings()

    def connect(self):
        """
        Remove states and arcs not on successful paths. [nondestructive]
        """
        accessible = self.accessible(forward=True)
        coaccessible = self.accessible(forward=False)
        live_states = accessible & coaccessible
        dead_states = filter(lambda q: q not in live_states, self.states())
        dead_states = list(dead_states)
        wfst = self.delete_states(dead_states, connect=False)
        return wfst

    def accessible(self, forward=True):
        """
        Ids of states accessible from initial state -or- coaccessible from final states.
        """
        fst = self.fst

        if forward:
            # Initial state id and forward transitions
            Q = set([fst.start()])
            T = {}
            for src in fst.states():
                T[src] = set()
                for t in fst.arcs(src):
                    dest = t.nextstate
                    T[src].add(dest)
        else:
            # Final state ids and backward transitions
            Q = set([q for q in fst.states() if self.is_final(q)])
            T = {}
            for src in fst.states():
                for t in fst.arcs(src):
                    dest = t.nextstate
                    if dest not in T:
                        T[dest] = set()
                    T[dest].add(src)

        # (Co)accessible state ids
        Q_old = set()
        Q_new = set(Q)
        while len(Q_new) != 0:
            Q_old, Q_new = Q_new, Q_old
            Q_new.clear()
            for src in filter(lambda src: src in T, Q_old):
                for dest in filter(lambda dest: dest not in Q, T[src]):
                    Q.add(dest)
                    Q_new.add(dest)
        return Q

    def delete_states(self, dead_states, connect=True):
        """
        Remove states by id while preserving labels [nondestructive].
        """
        fst = self.fst

        # Preserve input/output symbols and weight type
        wfst = Wfst(fst.input_symbols(), fst.output_symbols(), fst.arc_type())

        # Reindex live states, copying labels
        state_map = {}
        q0 = fst.start()
        for q in filter(lambda q: q not in dead_states, fst.states()):
            q_label = self.state_label(q)
            q_id = wfst.add_state(q_label)
            state_map[q] = q_id
            if q == q0:
                wfst.set_start(q_id)
            wfst.set_final(q_id, self.final(q))

        # Copy transitions between live states
        for q in filter(lambda q: q not in dead_states, fst.states()):
            src = state_map[q]
            for t in filter(lambda t: t.nextstate not in dead_states,
                            fst.arcs(q)):
                dest = state_map[t.nextstate]
                wfst.add_arc(src, t.ilabel, t.olabel, t.weight, dest)

        if connect:
            wfst.connect()
        return wfst

    def delete_arcs(self, dead_arcs):
        """
        Delete arcs [destructive]
        Implemented by deleting all arcs from relevant states then adding back all non-dead arcs, as suggested in the OpenFst forum:
        https://www.openfst.org/twiki/bin/view/Forum/FstForumArchive2014
        """
        fst = self.fst

        # Group dead arcs by source state
        dead_arcs_ = {}
        for (src, t) in dead_arcs:
            if src in dead_arcs_:
                dead_arcs_[src].append(t)
            else:
                dead_arcs_[src] = [t]

        # Process states with dead arcs
        for q in filter(lambda q: q in dead_arcs_, fst.states()):
            # Remove all arcs from state
            arcs = [t for t in self.arcs(q)]
            fst.delete_arcs(q)
            # Add back live arcs
            for t1 in arcs:
                live = True
                for t2 in dead_arcs_[q]:
                    if arc_equal(t1, t2):
                        live = False
                        break
                if live:
                    self.add_arc(q, t1.ilabel, t1.olabel, t1.weight,
                                 t1.nextstate)
        return self

    def transduce(self, x):
        """
        Transduce space-separated input x with this machine, returning iterator over paths with input/output labels (but not state labels).
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()
        try:
            fst_in = pynini.accep(x, token_type=isymbols)
        except:
            # todo: warn
            return []
        fst_out = fst_in @ fst
        strpath_iter = fst_out.paths(
            input_token_type=isymbols, output_token_type=osymbols)
        return strpath_iter

    def push_weights(self,
                     delta=1.0e-6,
                     remove_total_weight=False,
                     reweight_type="to_initial"):
        """
        Push weights.
        """
        self.fst = self.fst.push(delta, remove_total_weight, reweight_type)
        return self

    def sample(self,
               npath=1,
               seed=0,
               select="uniform",
               max_length=2147483647,
               weighted=False,
               remove_total_weight=False):
        """
        Randomly generate paths through this machine.
        """
        fst = pynini.randgen(
            self.fst,
            npath=npath,
            seed=seed,
            select=select,
            max_length=max_length,
            weighted=weighted,
            remove_total_weight=remove_total_weight)
        wfst = Wfst(fst.input_symbols(), fst.output_symbols(), fst.arc_type())
        wfst.fst = fst
        wfst._state2label = {q: str(q) for q in fst.states()}
        wfst._label2state = {v: k for k, v in wfst._state2label.items()}
        return wfst

    # Copying

    def copy(self):
        """
        Deep copy preserving input/output/state symbols and string outputs.
        """
        fst = self.fst
        wfst = Wfst(fst.input_symbols(), fst.output_symbols(), fst.arc_type())
        wfst.fst = self.fst.copy()
        wfst._state2label = dict(self._state2label)
        wfst._label2state = dict(self._label2state)
        wfst.sigma = dict(self.sigma)
        return wfst

    # Printing/drawing

    def print(self, **kwargs):
        # State symbol table
        state_symbols = pynini.SymbolTable()
        for q, label in self._state2label.items():
            state_symbols.add_symbol(str(label), q)
        fst = self.fst
        return fst.print(
            isymbols=fst.input_symbols(),
            osymbols=fst.output_symbols(),
            ssymbols=state_symbols,
            **kwargs)

    def draw(self, source, acceptor=True, portrait=True, **kwargs):
        # State symbol table
        state_symbols = pynini.SymbolTable()
        for q, label in self._state2label.items():
            state_symbols.add_symbol(str(label), q)
        fst = self.fst
        return fst.draw(
            source,
            isymbols=fst.input_symbols(),
            osymbols=fst.output_symbols(),
            ssymbols=state_symbols,
            acceptor=acceptor,
            portrait=portrait,
            **kwargs)


def arc_equal(arc1, arc2):
    """
    Arc equality (missing from pynini?)
    """
    val = (arc1.ilabel == arc2.ilabel) and \
        (arc1.olabel == arc2.olabel) and \
            (arc1.weight == arc2.weight) and \
                (arc1.nextstate == arc2.nextstate)
    return val


def compose(wfst1, wfst2):
    """
    FST composition, retaining contextual info from original machines by labeling each state q = (q1, q2) with (label(q1), label(q2))
    todo: matcher options; handle weights; flatten labels from repeated calls
    """
    wfst = Wfst(config.symtable)
    Zero = Weight.zero(wfst.weight_type())

    q0_1 = wfst1.start()
    q0_2 = wfst2.start()
    q0 = (wfst1.state_label(q0_1), wfst2.state_label(q0_2))
    wfst.add_state(q0)
    wfst.set_start(q0)

    # Lazy state and transition construction
    Q = set([q0])
    Q_old, Q_new = set(), set([q0])
    while len(Q_new) != 0:
        Q_old, Q_new = Q_new, Q_old
        Q_new.clear()
        for src in Q_old:
            src1, src2 = src  # State labels in M1, M2
            for t1 in wfst1.arcs(src1):
                for t2 in wfst2.arcs(src2):  # (todo: sort arcs)
                    if t1.olabel != t2.ilabel:
                        continue
                    dest1 = t1.nextstate
                    dest2 = t2.nextstate
                    dest = (wfst1.state_label(dest1), wfst2.state_label(dest2))
                    wfst.add_state(dest)  # No change if state already exists
                    wfst.add_arc(
                        src=src, ilabel=t1.ilabel, olabel=t2.olabel, dest=dest)
                    if wfst1.final(dest1) != Zero and wfst2.final(
                            dest2) != Zero:
                        wfst.set_final(dest)  # Final if both dest1, dest2 are
                    if dest not in Q:
                        Q.add(dest)
                        Q_new.add(dest)

    return wfst.connect()


def accepted_strings(wfst, side='input', max_len=10):
    """
    Strings accepted by wfst on designated side, up to max_len (not including bos/eos); cf. pynini for paths through acyclic fst
    todo: epsilon-handling
    """
    q0 = wfst.start(label=False)
    Zero = Weight.zero(wfst.weight_type())

    accepted = set()
    prefixes = {(q0, None)}
    prefixes_new = set()
    for i in range(max_len + 2):
        for (src, prefix) in prefixes:
            for t in wfst.arcs(src):
                dest = t.nextstate
                if side == 'input':
                    tlabel = wfst.input_label(t.ilabel)
                else:
                    tlabel = wfst.output_label(t.olabel)
                if prefix is None:
                    prefix_new = tlabel
                else:
                    prefix_new = prefix + ' ' + tlabel
                prefixes_new.add((dest, prefix_new))
                if wfst.final(dest) != Zero:
                    accepted.add(prefix_new)
                    #print(prefix_new)
        prefixes, prefixes_new = prefixes_new, prefixes
        prefixes_new.clear()

    return accepted


def left_context_acceptor(context_length=1, sigma_tier=None):
    """
    Acceptor (identity transducer) for segments in immediately preceding contexts (histories) of specified length. If Sigma_tier is specified as  a subset of Sigma, only contexts over Sigma_tier are tracked (other member of Sigma are skipped, i.e., label self-loops on each interior state)
    """
    epsilon = config.epsilon
    bos = config.bos
    eos = config.eos
    if sigma_tier is None:
        sigma_tier = set(config.sigma)
        sigma_skip = set()
    else:
        sigma_skip = set(config.sigma) - sigma_tier
    wfst = Wfst(config.symtable)

    # Initial and peninitial states
    q0 = ('λ',)
    q1 = (epsilon,) * (context_length - 1) + (bos,)
    wfst.add_state(q0)
    wfst.set_start(q0)
    wfst.add_state(q1)
    wfst.add_arc(src=q0, ilabel=bos, dest=q1)

    # Interior arcs
    # xα -- y --> αy for each y
    Q = {q0, q1}
    Qnew = set(Q)
    for l in range(context_length + 1):
        Qold = set(Qnew)
        Qnew = set()
        for q1 in Qold:
            if q1 == q0:
                continue
            for x in sigma_tier:
                q2 = _suffix(q1, context_length - 1) + (x,)
                wfst.add_state(q2)
                wfst.add_arc(src=q1, ilabel=x, dest=q2)
                Qnew.add(q2)
        Q |= Qnew

    # Final state and incoming arcs
    qf = (eos,)
    wfst.add_state(qf)
    wfst.set_final(qf)
    for q1 in Q:
        if q1 == q0:
            continue
        wfst.add_arc(src=q1, ilabel=eos, dest=qf)
    Q.add(qf)

    # Self-transitions labeled by skipped symbols
    # on interior states
    for q in Q:
        if (q == q0) or (q == qf):
            continue
        for x in sigma_skip:
            wfst.add_arc(src=q, ilabel=x, dest=q)

    return wfst


def right_context_acceptor(context_length=1, sigma_tier=None):
    """
    Acceptor (identity transducer) for segments in immediately following contexts (futures) of specified length. If Sigma_tier is specified as a subset of Sigma, only contexts over Sigma_tier are tracked (other members of Sigma are skipped, i.e., label self-loops on each interior state)
    """
    epsilon = config.epsilon
    bos = config.bos
    eos = config.eos
    if sigma_tier is None:
        sigma_tier = set(config.sigma)
        sigma_skip = set()
    else:
        sigma_skip = set(config.sigma) - sigma_tier
    wfst = Wfst(config.symtable)

    # Final and penultimate state
    qf = ('λ',)
    qp = (eos,) + (epsilon,) * (context_length - 1)
    wfst.add_state(qf)
    wfst.set_final(qf)
    wfst.add_state(qp)
    wfst.add_arc(src=qp, ilabel=eos, dest=qf)

    # Interior transitions
    # xα -- x --> αy for each y
    Q = {qf, qp}
    Qnew = set(Q)
    for l in range(context_length + 1):
        Qold = set(Qnew)
        Qnew = set()
        for q2 in Qold:
            if q2 == qf:
                continue
            for x in sigma_tier:
                q1 = (x,) + _prefix(q2, context_length - 1)
                wfst.add_state(q1)
                wfst.add_arc(src=q1, ilabel=x, dest=q2)
                Qnew.add(q1)
        Q |= Qnew

    # Initial state and outgoing transitions
    q0 = (bos,)
    wfst.add_state(q0)
    wfst.set_start(q0)
    for q in Q:
        if q == qf:
            continue
        wfst.add_arc(src=q0, ilabel=bos, dest=q)
    Q.add(q0)

    # Self-transitions labeled by skipped symbols
    # on interior states
    for q in Q:
        if (q == q0) or (q == qf):
            continue
        for x in sigma_skip:
            wfst.add_arc(src=q, ilabel=x, dest=q)

    return wfst


def _prefix(x, l):
    """ Length-l prefix of tuple x """
    if l < 1:
        return ()
    if len(x) < l:
        return x
    return x[:l]


def _suffix(x, l):
    """ Length-l suffix of tuple x """
    if l < 1:
        return ()
    if len(x) < l:
        return x
    return x[-l:]

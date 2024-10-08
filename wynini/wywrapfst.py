import re, sys, pickle
import bisect
import itertools
import numpy as np

import pynini
from pynini import Fst, Arc, Weight, \
    SymbolTable, SymbolTableView
from graphviz import Source

from wynini import config

verbose = 0


class Wfst():
    """
    Wrapper for pynini.Fst with automatic handling of labels for 
    inputs / outputs / states and output strings. State labels must 
    be hashable (strings, tuples, etc.). Pynini constructive and 
    destructive operations generally lose track of state ids and 
    symbol labels, therefore many operations are reimplemented here 
    (e.g., connect, compose) in a way that preserves those labels.
    
    OpenFst / Fst(_pywrapfst.VectorFst) arc types and weights:
    - Fst argument arc_type: "standard" | "log" | "log64".
    - "The OpenFst library predefines TropicalWeight and LogWeight 
    as well as the corresponding StdArc and LogArc."
    - https://www.openfst.org/doxygen/fst/html/arc_8h_source.html
    - https://www.openfst.org/twiki/bin/view/FST/FstQuickTour#FstWeights
    - https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#Weights
    - Weight constructor is Weight(weight_type, weight_value) 
    where weight_type is "tropical" | "log" | "log64";
    there are special constructors Weight.zero(weight_type), 
    Weight.one(weight_type).

    OpenFst advanced usage:
    - https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#OpenFst%20Advanced%20Usage
    """

    def __init__(self, isymbols=None, osymbols=None, arc_type='standard'):
        # Symbol tables.
        # todo: require SymbolTable(View) args
        if isymbols is None:  # todo: rename input_symbols
            isymbols, _ = config.make_symtable([])
        if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
            isymbols, _ = config.make_symtable(isymbols)
        if osymbols is None:  # todo: rename output_symbols
            osymbols = isymbols
        if not isinstance(osymbols, (SymbolTable, SymbolTableView)):
            osymbols, _ = config.make_symtable(osymbols)
        self.fst = fst = Fst(arc_type)  # Wrapped Fst.
        fst.set_input_symbols(isymbols)  # Arc input symbols.
        fst.set_output_symbols(osymbols)  # Arc output symbols.
        self._state2label = {}  # State id -> state label.
        self._label2state = {}  # State label -> state id.
        # note: state id <-> state label assumed to be one-to-one.
        self.sigma = {}  # State id -> output string.
        self.phi = {}  # Arc -> loglinear features ({f_k: v_k}).

    # Input/output labels (most delegate to pynini.Fst).

    def input_symbols(self):
        """ Get input symbol table. """
        return self.fst.input_symbols()

    def output_symbols(self):
        """ Get output symbol table. """
        return self.fst.output_symbols()

    def mutable_input_symbols(self):
        """ Get mutable input symbol table. """
        return self.fst.mutable_input_symbols()

    def mutable_output_symbols(self):
        """ Get mutable output symbol table. """
        return self.fst.mutable_output_symbols()

    def set_input_symbols(self, isymbols):
        """ Set input symbol table. """
        self.fst.set_input_symbols(isymbols)
        return self

    def set_output_symbols(self, osymbols):
        """ Set output symbol table. """
        self.fst.set_output_symbols(osymbols)
        return self

    def input_label(self, sym):
        """ Get input label for symbol id. """
        return self.fst.input_symbols().find(sym)

    def input_id(self, sym):
        """ Get input id for symbol label. """
        return self.fst.input_symbols().find(sym)

    def output_label(self, sym):
        """ Get output label for symbol id. """
        return self.fst.output_symbols().find(sym)

    def output_id(self, sym):
        """ Get output id for symbol label. """
        return self.fst.output_symbols().find(sym)

    # States.

    def state_label(self, q):
        """ State label from id. """
        if not isinstance(q, int):
            return q
        return self._state2label[q]

    def state_id(self, q):
        """ State id from label. """
        if isinstance(q, int):
            return q
        return self._label2state[q]

    def set_state_label(self, q, label):
        """ Update label of state q. """
        # Enforce one-to-one state labeling.
        if label in self._label2state:
            print(f'Cannot set label of state {q} to {label} '
                  f'(label already used).')
            return None
        self._state2label[q] = label
        self._label2state[label] = q
        return None

    def add_state(self, label=None, start=False, initial=False, final=False):
        """ Add new state, optionally specifying its label. """
        # Enforce one-to-one state labeling.
        if label in self._label2state:
            if verbose:
                print(f'State with label {label} already exists '
                      f'(returning it).')
            return self._label2state[label]
        # Add new state.
        q = self.fst.add_state()
        # State id self-labeling by default.
        if label is None:
            label = q
        if verbose and isinstance(label, int) and label != q:
            print(f'Warning: labeling state {q} with '
                  f'integer other than {q}.')
        # State <-> label maps.
        self._state2label[q] = label
        self._label2state[label] = q
        # Initial and final state properties.
        if start or initial:
            self.set_start(q)
        if final:
            # todo: accept final weight argument
            self.set_final(q, final)

        return q

    def states(self, label=True):
        """ Iterator over state labels or ids. """
        fst = self.fst
        if not label:
            return fst.states()
        return map(lambda q: self.state_label(q), fst.states())

    def state_ids(self):
        """ Iterator over state ids. """
        return self.states(label=False)

    def num_states(self):
        """ Number of states in this machine. """
        return self.fst.num_states()

    def set_start(self, q):
        """ Set start state by id or label. """
        q = self.state_id(q)
        return self.fst.set_start(q)

    # Alias for set_start().
    set_initial = set_start

    def start(self, label=True):
        """ Start state label (or id). """
        q0 = self.fst.start()
        if not label:
            return q0
        return self.state_label(q0)

    # Alias for start().
    initial = start

    def is_start(self, q):
        """ Check start status by id or label. """
        q = self.state_id(q)
        return q == self.fst.start()

    # Alias for is_start().
    is_initial = is_start

    def set_final(self, q, weight=None):
        """
        Set final weight of state by id or label.
        note: default weight is one, not zero.
        """
        q = self.state_id(q)
        if weight is None or weight is True:
            weight = Weight.one(self.weight_type())
        return self.fst.set_final(q, weight)

    def is_final(self, q):
        """ Check final status by id or label. """
        q = self.state_id(q)
        zero = Weight.zero(self.weight_type())
        return self.final(q) != zero

    def final(self, q):
        """ Final weight of state by id or label. """
        q = self.state_id(q)
        return self.fst.final(q)

    # Alias for final.
    final_weight = final

    def finals(self, label=True):
        """
        Iterator over states with non-zero final weights.
        """
        fst = self.fst
        zero = Weight.zero(fst.weight_type())
        state_iter = fst.states()  # equiv.: self.state_ids()
        state_iter = filter(lambda q: fst.final(q) != zero, state_iter)
        if label:
            state_iter = map(lambda q: self.state_label(q), state_iter)
        return state_iter

    def relabel_states(self, func=None):
        """
        Relabel states with their state ids (default) 
        or using function argument.
        (see pynini.topsort)
        """
        if func:
            return self._relabel_states(func)
        state2label = {}  # State id -> state label.
        label2state = {}  # State label -> state id.
        for q in self.state_ids():
            state2label[q] = q
            label2state[q] = q
        self._state2label = state2label
        self._label2state = label2state
        return self

    def _relabel_states(self, func):
        """
        Relabel states using function, monitoring
        for label collisions.
        """
        state2label = {}
        label2state = {}
        for q in self.state_ids():
            label = func(self, q)
            if label in label2state:
                print(f'Relabeling function assigns the same label '
                      f'to multiple states; ignoring.')
                return self
            state2label[q] = label
            label2state[label] = q
        self._state2label = state2label
        self._label2state = label2state
        return self

    # Arcs.

    def make_arc(self,
                 src=None,
                 ilabel=None,
                 olabel=None,
                 weight=None,
                 dest=None):
        """
        Create (but do not add) arc. Accepts id or label
        for each of src / ilabel / olabel / dest.
        """
        fst = self.fst
        src_id = self.state_id(src)
        if isinstance(ilabel, str):
            ilabel = fst.mutable_input_symbols().add_symbol(ilabel)
        if isinstance(olabel, str):
            olabel = fst.mutable_output_symbols().add_symbol(olabel)
        if olabel is None:
            olabel = ilabel
        if weight is None:
            weight = Weight.one(self.weight_type())
        dest_id = self.state_id(dest)
        arc = Arc(ilabel, olabel, weight, dest_id)
        return src_id, arc

    def make_epsilon_arc(self, src):
        """
        Create (but do not add) 'virtual' epsilon:epsilon self-arc on a state.
        (see https://www.openfst.org/doxygen/fst/html/compose_8h_source.html)
        """
        one = Weight.one(self.weight_type())
        src_id, arc = self.make_arc( \
            src, config.epsilon, config.epsilon,
            one, src)
        return src_id, arc

    def add_arc(self,
                src=None,
                ilabel=None,
                olabel=None,
                weight=None,
                dest=None):
        """
        Add arc. Accepts id or label for each of 
        src / ilabel / olabel / dest.
        todo: add src/dest states if they do not already exist
        todo: ensure that weights on epsilon:epsilon 
        self-transitions are always one (or None).
        """
        src_id, arc = self.make_arc( \
            src, ilabel, olabel, weight, dest)
        self.fst.add_arc(src_id, arc)
        return src_id, arc

    def add_path(self,
                 src=None,
                 ilabel=None,
                 olabel=None,
                 weight=None,
                 dest=None):
        """
        Add path labeled by space-separated ilabel and 
        olabel (possibly of different lengths, either
        of which can be null).
        """
        # Ensure same-length input/output sequences.
        if ilabel is None and olabel is None:
            # no-op
            return self
        if ilabel is None:
            ilabel = config.epsilon
        if olabel is None:
            olabel = config.epsilon
        ilabels = ilabel.split(' ')
        olabels = olabel.split(' ')
        ilength = len(ilabels)
        olength = len(olabels)
        if ilength < olength:
            ilabels += [config.epsilon] * (olength - ilength)
        if ilength > olength:
            olabels += [config.epsilon] * (ilength - olength)
        #print(ilabels, olabels)
        # Add path.
        n = len(ilabels)  # equiv. len(olabels)
        q = src
        r = None
        for i in range(n - 1):
            r = self.add_state()
            self.add_arc(q, ilabels[i], olabels[i], None, r)
            q = r
        self.add_arc(q, ilabels[n - 1], olabels[n - 1], weight, dest)
        return self

    def arcs(self, src=None):
        """
        Iterator over arcs from designated state or all states.
        """
        # todo: decorate arcs with input/output labels if requested
        if src is None:
            for src in self.state_ids():
                for t in self.fst.arcs(src):
                    yield (src, t)  # (src, arc) pair
            return
        src = self.state_id(src)
        for t in self.fst.arcs(src):
            yield t  # arc
        #return self.fst.arcs(src) # fixme: does not work as expected

    # Alias for arcs().
    transitions = arcs

    def mutable_arcs(self, src):
        """
        Mutable iterator over arcs from state.
        todo: checkme
        """
        src = self.state_id(src)
        return self.fst.mutable_arcs(src)  # _MutableArcIterator

    # Alias for mutable_arcs().
    # mutable_transitions = mutable_arcs

    def arcsort(self, sort_type='ilabel'):
        """
        Sort arcs from each state.
        arg sort_type = 'ilabel' | 'olabel'.
        """
        if sort_type == 'input':
            sort_type = 'ilabel'
        if sort_type == 'output':
            sort_type = 'olabel'
        self.fst.arcsort(sort_type)
        return self

    def relabel_arcs(self, ifunc=None, ofunc=None):
        """
        Relabel arc input and/or output symbols
        (see: pynini.relabel_tables).
        note: epsilon, bos, eos should be mapped to themselves.
        todo: checkme
        """
        if ifunc is None and ofunc is None:
            return self

        # Relabel input symbols.
        isymbols = self.input_symbols()
        if ifunc is not None:
            idx = 0
            isymbols_idx = {}  # Old index -> new index.
            isymbols_label = {}  # New label -> new index.
            for (i, x) in isymbols():
                y = ifunc(x)
                if y is None:  # Allow partial functions.
                    y = x
                if y in isymbols_label:  # Existing symbol.
                    isymbols_idx[i] = isymbols_label[y]
                else:  # New symbol.
                    isymbols_label[y] = idx
                    isymbols_idx[i] = idx
                    idx += 1
            isymbols = SymbolTable()
            for (y, idx) in isymbols_label:
                isymbols.add_symbol(y)

        # Relabel output symbols.
        osymbols = self.output_symbols()
        if ofunc is not None:
            idx = 0
            osymbols_idx = {}  # Old index -> new index.
            osymbols_label = {}  # New label -> new index.
            for (i, x) in osymbols:
                y = ofunc(x)
                if y is None:  # Allow partial functions.
                    y = x
                if y in osymbols_label:  # Existing symbol.
                    osymbols_idx[i] = osymbols_label[y]
                else:  # New symbol.
                    osymbols_label[y] = idx
                    osymbols_idx[i] = idx
                    idx += 1
            osymbols = SymbolTable()
            for (y, idx) in osymbols_label:
                osymbols.add_symbol(y)

        # Relabel arc inputs/outputs in underlying fst.
        fst = self.fst
        fst.set_input_symbols(isymbols)
        fst.set_output_symbols(osymbols)
        for q in self.states:
            arcs = fst.arcs(q)
            fst.delete_arcs(q)
            for t in arcs:
                ilabel = t.ilabel
                if ifunc is not None:
                    ilabel = isymbols_idx[ilabel]
                olabel = t.olabel
                if ofunc is not None:
                    olabel = osymbols_idx[olabel]
                self.add_arc(q, ilabel, olabel, t.weight, t.nextstate)

        # Relabel arcs (keys) in feature mapping phi.
        # note: features may be invalidated by relabeling.
        phi = self.phi
        if phi is not None:
            phi_ = {}
            for (t, ftrs) in phi.items():
                (q, ilabel, olabel, nextstate) = t
                if ifunc is not None:
                    ilabel = isymbols_idx[ilabel]
                if ofunc is not None:
                    olabel = osymbols_idx[olabel]
                phi_[(q, ilabel, olabel, nextstate)] = ftrs
        self.phi = phi_
        return self

    def num_arcs(self, src=None):
        """
        Number of arcs from designated state (out-degree)
        or total number of arcs.
        """
        fst = self.fst
        if src is None:
            n = 0
            for q in fst.states():
                n += fst.num_arcs(q)
            return n
        src = self.state_id(src)
        return fst.num_arcs(src)

    def num_input_epsilons(self, src=None):
        """
        Number of arcs with input epsilon from one
        state or from all states.
        """
        fst = self.fst
        if src is None:
            n = 0
            for src in self.state_ids():
                n += fst.num_input_epsilons(src)
            return n
        src = self.state_id(src)
        return fst.num_input_epsilons(src)

    def num_output_epsilons(self, src):
        """
        Number of arcs with output epsilon from one
        state or from all states.
        """
        fst = self.fst
        if src is None:
            n = 0
            for src in self.state_ids():
                n += fst.num_output_epsilons(src)
            return n
        src = self.state_id(src)
        return fst.num_output_epsilons(src)

    def info(self):
        """
        Number of states and arcs.
        """
        nstate = self.num_states()
        narc = self.num_arcs()
        return f'{nstate} states | {narc} arcs'

    def ilabel(self, x):
        """ Arc input label. """
        if isinstance(x, Arc):
            x = x.ilabel
        return self.fst.input_symbols().find(x)

    def olabel(self, x):
        """ Arc output label. """
        if isinstance(x, Arc):
            x = x.olabel
        return self.fst.output_symbols().find(x)

    def weight(self, arc):
        """ Weight on arc. """
        return arc.weight

    def arc_type(self):
        """ Arc type (standard, log, log64). """
        return self.fst.arc_type()

    def weight_type(self):
        """ Weight type (tropical, log, log64). """
        return self.fst.weight_type()

    def project(self, project_type):
        """ Project input or output labels. """
        # assumption: Fst.project() does not reindex states.
        fst = self.fst
        if project_type == 'input':
            isymbols = fst.input_symbols()
            self.set_output_symbols(isymbols)
        if project_type == 'output':
            osymbols = fst.output_symbols()
            self.set_input_symbols(osymbols)
        fst.project(project_type)
        return self

    def arc_labels(self, sep=None):
        """
        Get all ordinary input-output labels.
        note: get epsilon, bos, eos labels from config.
        """
        epsilon = config.epsilon
        bos = config.bos
        eos = config.eos
        for q, t in self.arcs():
            ilabel = self.ilabel(t)
            olabel = self.olabel(t)
            if ilabel in [bos, eos] or olabel in [bos, eos]:
                continue
            if ilabel == epsilon and ilabel == olabel:
                continue
            if sep:
                yield f'{ilabel} {sep} {olabel}'  # string
            else:
                yield (ilabel, olabel)  # tuple
        return

    def map_weights(self, map_type='identity', **kwargs):
        """
        Map arc weights (see pynini.arcmap).
        map_type is "identity", "invert", "quantize", "plus", 
        "power", "rmweight", "times", "to_log", "to_log64",
        or "to_std" (which maps to the tropical semiring).
        """
        # assumption: pynini.arcmap() does not reindex states.
        if map_type == 'identity':
            return self
        if map_type in ('to_trop', 'to_tropical', 'to_standard'):
            map_type = 'to_std'
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()
        fst_out = pynini.arcmap(fst, map_type=map_type, **kwargs)
        fst_out.set_input_symbols(isymbols)
        fst_out.set_output_symbols(osymbols)
        self.fst = fst_out
        return self

    def assign_weights(self, func=None):
        """
        Assign weights to arcs in this machine with an arbitrary 
        function func (<Wfst, q, t> -> Weight) that receives this 
        machine, a source state id, and an arc as inputs and 
        returns an arc weight. The function can examine the src/ 
        input/output/dest and associated labels of the arc.
        (see map_type options in map_weights(): "identity", 
        "plus", "power", "times")
        """
        if func is None:
            # Identity function by default.
            return self
        fst = self.fst
        weight_type = self.weight_type()
        for q in fst.states():
            q_arcs = fst.mutable_arcs(q)
            for t in q_arcs:  # note: unstable arc reference
                w = func(self, q, t)
                if isinstance(w, int) or isinstance(w, float):
                    w = Weight(weight_type, w)
                    # todo: handle non-numerical weights
                t.weight = w
                q_arcs.set_value(t)
        return self

    def features(self, q, t, default={}):
        """
        Get features of arc t from state q.
        """
        q = self.state_id(q)
        t_ = (q, t.ilabel, t.olabel, t.nextstate)
        return self.phi.get(t_, default)

    def set_features(self, q, t, phi_t):
        """
        Set features of arc t from state q.
        """
        if not phi_t:
            return
        q = self.state_id(q)
        t_ = (q, t.ilabel, t.olabel, t.nextstate)
        self.phi[t_] = phi_t
        return  # todo: return type

    def assign_features(self, func):
        """
        Assign features (as in loglinear/maxent/HG/OT models) 
        to arcs in this machine with an arbitrary function
        func (<Wfst, q, t> -> feature violations) that receives 
        this machine, a source state id, and an arc as inputs 
        and returns a dictionary of feature 'violations'.
        The function can examine the src/input/output/dest and 
        associated labels of the arc.
        """
        phi = {}
        for q in self.fst.states():
            for t in self.fst.arcs(q):
                phi_t = func(self, q, t)
                if phi_t:  # Handle partial functions / empty phi_t.
                    t_ = (q, t.ilabel, t.olabel, t.nextstate)
                    phi[t_] = phi_t
        self.phi = phi
        return self

    # Algorithms.
    # todo: determinize(), difference(), epsnormalize(),
    # minimize(), rmepsilon()

    def encode_labels(self, iosymbols=None, sep=':'):
        """
        Convert transducer to acceptor by combining
        input and output label of each arc.
        arg iosymbols: symbol table for encoded labels
        todo: destructive version, to avoid copying.
        """
        # Symbol table for input/output label pairs.
        if not iosymbols:
            iosymbols, _ = Wfst.pair_symbols(self.input_symbols(),
                                             self.output_symbols(),
                                             sep=sep)
        # Machine with encoded labels.
        wfst = Wfst(isymbols=iosymbols, arc_type=self.arc_type())
        # Copy states.
        for q in self.states():
            wfst.add_state(q)
        wfst.set_initial(self.initial())
        for q in self.finals():
            wfst.set_final(q, self.final_weight(q))
        # Copy arcs, encoding labels.
        for q in self.state_ids():
            for t in self.arcs(q):
                ilabel = self.ilabel(t)
                olabel = self.olabel(t)
                iolabel = self.pair_symbol(ilabel, olabel)
                phi_t = self.features(q, t)
                q_, t_ = wfst.add_arc( \
                    q,
                    iolabel,
                    None,
                    t.weight,
                    t.nextstate)
                wfst.set_features(q_, t_, phi_t)
        return wfst, iosymbols

    def decode_labels(self, isymbols, osymbols, sep=':'):
        """
        Convert acceptor to transducer by splitting
        input label of each arc.
        arg isymbols: symbol table for input labels
        arg osymbols: symbol table for output labels
        todo: destructive version, to avoid copying.
        """
        wfst = Wfst(isymbols=isymbols,
                    osymbols=osymbols,
                    arc_type=self.arc_type())
        # Copy states.
        for q in self.states():
            wfst.add_state(q)
        wfst.set_initial(self.initial())
        for q in self.finals():
            wfst.set_final(q, self.final_weight(q))
        # Copy arcs, decoding labels.
        for q in self.state_ids():
            for t in self.arcs(q):
                iolabel = self.ilabel(t)
                ilabel, olabel = Wfst.unpair_symbol(iolabel)
                phi_t = self.features(q, t)
                q_, t_ = wfst.add_arc( \
                    q,
                    ilabel,
                    olabel,
                    t.weight,
                    t.nextstate)
                wfst.set_features(q_, t_, phi_t)
        return wfst

    @classmethod
    def pair_symbols(cls, input_symbols, output_symbols, sep=':'):
        """
        Combine all input and output symbols for encoding.
        note: epsilon, bos, eos are retained from config.
        todo: checkme
        """
        iosymbols = set()
        for _, isym in input_symbols:
            for _, osym in output_symbols:
                iosym = Wfst.pair_symbol(isym, osym, sep)
                iosybols.add(iosym)
        return config.make_symtable(list(iosymbols))

    @classmethod
    def pair_symbol(cls, isym, osym, sep=':'):
        """
        Combine input and output symbols for encoding.
        note: epsilon, bos, eos retained from config.
        todo: move to string util
        """
        epsilon = config.epsilon
        bos = config.bos
        eos = config.eos
        if isym in [bos, eos] or osym in [bos, eos]:
            return isym
        if isym == osym == epsilon:
            return epsilon
        iosym = f'{isym} {sep} {osym}'
        return iosym

    @classmethod
    def unpair_symbol(cls, iosym, sep=':'):
        """
        Split input and output symbols for decoding.
        todo: move to string util
        """
        # Special symbols.
        # (epsilon, bos, eos, lambda, ...)
        if sep not in iosym:
            return (iosym, iosym)
        iosym = re.match(f'^(.+) {sep} (.+)$', iosym)
        return (iosym[1], iosym[2])

    def paths(self):
        """
        Iterator over paths through this machine (must be acyclic). 
        pynini.StringPathIterator is not iterable (!) but has methods: 
        next(); ilabels(), istring(), labels(), ostring(), 
        weights(); istrings(), ostrings(), weights(), items().
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()
        path_iter = fst.paths(input_token_type=isymbols,
                              output_token_type=osymbols)
        return path_iter

    def istrings(self):
        """
        Iterator over input strings of paths through this 
        machine (must be acyclic).
        """
        return self.paths().istrings()

    def ostrings(self):
        """
        Iterator over output strings of paths through this 
        machine (must be acyclic).
        """
        return self.paths().ostrings()

    def iostrings(self, sep=':'):
        """
        Generate aligned input:output sequences representing
        paths through this machine (must be acyclic).
        """
        path_iter = self.paths()
        while not path_iter.done():
            path = list(zip( \
                path_iter.ilabels(), path_iter.olabels()))
            path = [f'{self.ilabel(x)}{sep}{self.olabel(y)}' \
                for (x, y) in path]
            path = ' '.join(path)
            path_iter.next()
            yield path

    def accepted_strings(self,
                         side='input',
                         weights=True,
                         max_len=10,
                         delete_epsilon=True):
        """
        Generate input (default) or output labels of paths 
        through this machine (possibly cyclic) up to max_len
        (excluding bos/eos), optionally with weights.
        For acyclic machines, see paths() above.
        """
        fst = self.fst
        q0 = fst.start()
        weight_type = fst.weight_type()
        One = Weight.one(weight_type)
        Zero = Weight.zero(weight_type)

        paths_old = {(q0, None)}
        paths_new = set()
        if weights:
            path2weight = {(q0, None): One}
        # note: pynini weights are unhashable.

        for _ in range(max_len + 2):
            for path_old in paths_old:
                (src, label) = path_old
                for t in fst.arcs(src):
                    dest = t.nextstate

                    # Extend label.
                    if side == 'input':
                        tlabel = self.ilabel(t)
                    else:
                        tlabel = self.olabel(t)

                    if delete_epsilon and (tlabel == config.epsilon):
                        label_new = label
                    else:
                        if label is None:
                            label_new = tlabel
                        else:
                            label_new = label + ' ' + tlabel

                    # Update path.
                    path_new = (dest, label_new)
                    paths_new.add(path_new)

                    # Extend weight, update total weight.
                    if weights:
                        weight_old = path2weight[path_old]
                        weight_new = pynini.times( \
                            weight_old, t.weight)
                        if path_new in path2weight:
                            path2weight[path_new] = pynini.plus( \
                                path2weight[path_new], weight_new)
                        else:
                            path2weight[path_new] = weight_new

                    # Yield accepted path.
                    if fst.final(dest) != Zero:
                        if weights:
                            weight = pynini.times( \
                                path2weight[path_new],
                                fst.final(dest))
                            yield (label_new, weight)
                        else:
                            yield label_new

            paths_old, paths_new = paths_new, paths_old
            paths_new.clear()

    def connect(self):
        """
        Remove states and arcs that are not on successful paths.
        [nondestructive]
        """
        accessible = self.accessible()
        coaccessible = self.coaccessible()
        live_states = set(accessible) & set(coaccessible)
        dead_states = set(self.fst.states()) - live_states
        wfst = self.delete_states(dead_states, connect=False)
        return wfst

    def accessible(self, forward=True):
        """
        Ids of states accessible from initial state (forward) 
        -or- coaccessible from final states (backward).
        States are sorted (reverse-)topologically.
        """
        fst = self.fst

        states = []
        if forward:
            # Initial state id; forward arcs.
            Q = set([fst.start()])
            states += Q
            T = {}
            for src in fst.states():
                T[src] = set()
                for t in fst.arcs(src):
                    dest = t.nextstate
                    T[src].add(dest)
        else:
            # Final state ids; backward arcs
            # todo: use finals()
            Q = set([q for q in fst.states() if self.is_final(q)])
            states += Q
            T = {}
            for src in fst.states():
                for t in fst.arcs(src):
                    dest = t.nextstate
                    if dest not in T:
                        T[dest] = set()
                    T[dest].add(src)

        # (Co)accessible state ids.
        Q_old = set()
        Q_new = set(Q)
        while len(Q_new) != 0:
            Q_old, Q_new = Q_new, Q_old
            Q_new.clear()
            for src in filter(lambda q1: q1 in T, Q_old):
                for dest in filter(lambda q2: q2 not in Q, T[src]):
                    Q.add(dest)
                    Q_new.add(dest)
                    states.append(dest)
        #return Q
        return states

    def coaccessible(self):
        """ Alias for accessible. """
        return self.accessible(forward=False)

    def delete_states(self, states, connect=True):
        """
        Remove states by id while preserving state labels, 
        arc weights, and arc features. [nondestructive]
        """
        fst = self.fst
        live_states = set(fst.states()) - states

        # Preserve input/output symbols and aweight type.
        wfst = Wfst( \
            self.input_symbols(),
            self.output_symbols(),
            fst.arc_type())

        # Reindex live states, copying labels.
        state_map = {}
        q0 = fst.start()
        for q in live_states:
            q_label = self.state_label(q)
            q_id = wfst.add_state(q_label)
            state_map[q] = q_id
            if q == q0:
                wfst.set_start(q_id)
            wfst.set_final(q_id, self.final(q))

        # Copy arcs between live states.
        for q in live_states:
            src = state_map[q]
            for t in filter(lambda t: t.nextstate in live_states, fst.arcs(q)):
                dest = state_map[t.nextstate]
                wfst.add_arc(src, t.ilabel, t.olabel, t.weight, dest)
                phi_t = self.features(q, t)
                if phi_t:
                    t_ = (src, t.ilabel, t.olabel, dest)
                    wfst.phi[t_] = phi_t

        if connect:
            wfst.connect()
        return wfst

    def delete_arcs(self, dead_arcs):
        """
        Remove arcs. [destructive]
        Implemented by deleting all arcs from relevant states 
        then adding back all non-dead arcs, as suggested in the 
        OpenFst forum: 
        https://www.openfst.org/twiki/bin/view/Forum/FstForumArchive2014
        """
        fst = self.fst

        # Group dead arcs by source state.
        dead_arcs_ = {}
        for (src, t) in dead_arcs:
            if src not in dead_arcs_:
                dead_arcs_[src] = []
            dead_arcs_[src].append(t)

        # Process states with some dead arcs.
        for q in dead_arcs_:
            # Remove all arcs from state.
            arcs = fst.arcs(q)
            fst.delete_arcs(q)
            # Add back live arcs.
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

    def prune_arcs(self):
        """
        Delete duplicate arcs (which are allowed by Fst). [destructive]
        todo: sum weights of arcs with same src/ilabel/olabel/dest
        """
        fst = self.fst
        q_arc_set = set()
        q_arcs = []
        duplicates = False
        # Process each source state.
        for q in fst.states():
            # Identify unique arcs.
            q_arc_set.clear()
            q_arcs.clear()
            duplicates = False
            for t in fst.arcs(q):
                t_ = (t.ilabel, t.olabel, t.nextstate, str(t.weight))
                if t_ in q_arc_set:
                    duplicates = True
                else:
                    q_arc_set.add(t_)
                    q_arcs.append(t)
            # Skip if there are no duplicates.
            if not duplicates:
                continue
            # Delete all arcs.
            fst.delete_arcs(q)
            # Add back unique arcs.
            for t in q_arcs:
                self.add_arc( \
                    q, t.ilabel, t.olabel, t.weight, t.nextstate)
        return self

    def remove_arcs(self, func):
        """
        Delete arcs for which an arbitracy function 
        func(<Wfst, q, t> -> boolean) is True. The function 
        receives this machine, a source state id, and an arc 
        as arguments; it can examine the src/input/output/dest and 
        associated labels of the arc. [destructive]
        """
        dead_arcs = []
        fst = self.fst
        for q in fst.states():
            for t in fst.arcs(q):
                if func(self, q, t):
                    dead_arcs.append(t)
        return delete_arcs(dead_arcs)

    def transduce(self, x, add_delim=True, ret_type='outputs'):
        """
        Transduce space-separated input with this machine, 
        returning iterator over output strings (default) or 
        machine that preserves input/output labels but not 
        state labels, output strings, or arc features.
        Alternative that also preserves state labels 
        and arc features: create an accept for the input string
        with accep() and then compose with this machine.
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()

        if not isinstance(x, str):
            x = ' '.join(x)
        if add_delim:
            x = config.bos + ' ' + x + ' ' + config.eos
        fst_in = pynini.accep(x, token_type=isymbols)

        fst_out = fst_in @ fst
        fst_out.set_input_symbols(isymbols)
        fst_out.set_output_symbols(osymbols)

        if ret_type == 'fst':
            return fst_out
        if ret_type == 'wfst':
            wfst_out = Wfst.from_fst(fst_out)
            return wfst_out
        # default: return output strings
        path_iter = fst_out.paths(output_token_type=osymbols)
        return path_iter.ostrings()

    def push_weights(self,
                     delta=1e-6,
                     reweight_type='to_initial',
                     remove_total_weight=True):
        """
        Push arc weights and optionally remove total weight
        (see pynini.push/Fst.push, pynini.reweight/Fst.reweight).
        [destructive]
        note: assume that Fst.push() does not change state ids
        or reorder arcs within state.
        """
        self.fst = self.fst.push( \
            delta=delta,
            reweight_type=reweight_type,
            remove_total_weight=remove_total_weight)
        return self

    def normalize(self, delta=1e-6):
        """
        Globally normalize this machine.
        (see pynini.push, pynini.reweight, Fst.push).
        Equivalent to:
            dist = shortestdistance(wfst, reverse=True)
            wfst.reweight(potentials=dist, reweight_type='to_initial')
            // then remove total weight
        """
        return self.push_weights(delta=delta,
                                 reweight_type='to_initial',
                                 remove_total_weight=True)

    def reweight(self, potentials, reweight_type='to_initial'):
        # see pynini.reweight / Fst.reweight
        self.fst.reweight(potentials, reweight_type)
        return self

    def push_labels(self, reweight_type='to_initial', **kwargs):
        """
        Push labels; see pynini.push with arguments
        remove_common_affix (False) and 
        reweight_type ("to_initial" or "to_final").
        [destructive]
        note: assume that Fst.push() does not change state ids
        or reorder arcs within state.
        todo: testme
        """
        self.fst = pynini.push(self.fst,
                               push_labels=True,
                               reweight_type=reweight_type,
                               **kwargs)
        return self

    def randgen(self, npath=1, select=None, ret_type='outputs', **kwargs):
        """
        Randomly generate paths through this machine, returning
        an iterator over output strings (default) or machine 
        accepting the paths. pynini.randgen() arguments: npath, 
        seed, select ("uniform", "log_prob", or "fast_log_prob"),
        max_length, weighted, remove_total_weight.
        note: state labels / output strings / arc features
        are not retained in output machine.
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()
        if select is None:
            if fst.weight_type() in ('log', 'log64'):
                select = 'log_prob'
            else:
                select = 'uniform'

        fst_out = pynini.randgen( \
            fst, npath=npath, select=select, **kwargs)
        fst_out.set_input_symbols(isymbols)
        fst_out.set_output_symbols(osymbols)

        if ret_type == 'fst':
            return fst_out
        elif ret_type == 'wfst':
            wfst_out = Wfst.from_fst(fst_out)
            return wfst_out
        # default: return output strings
        path_iter = fst_out.paths(output_token_type=osymbols)
        return path_iter.ostrings()

    def invert(self):
        """
        Invert mapping (exchange input and output labels).
        note: assume Fst.invert() does not change state ids
        or order of arcs within state.
        """
        fst = self.fst
        isymbols = fst.input_symbols()
        osymbols = fst.output_symbols()
        fst.invert()
        self.set_input_symbols(osymbols)
        self.set_output_symbols(isymbols)
        return self

    # Copy/create machines.

    def copy(self):
        """
        Copy preserving input/output/state symbols 
        and string outputs.
        """
        fst = self.fst
        wfst = Wfst( \
            fst.input_symbols(),
            fst.output_symbols(),
            fst.arc_type())
        wfst.fst = fst.copy()
        wfst._state2label = dict(self._state2label)
        wfst._label2state = dict(self._label2state)
        wfst.sigma = dict(self.sigma)  # todo: deep copy
        wfst.phi = dict(self.phi)  # todo: deep copy
        return wfst

    @classmethod
    def from_fst(cls, fst):
        """ Wrap pynini FST / VectorFst in Wfst. """
        wfst = Wfst( \
            fst.input_symbols(),
            fst.output_symbols(),
            fst.arc_type())
        wfst.fst = fst
        wfst._state2label = {q: q for q in fst.states()}
        wfst._label2state = wfst._state2label.copy()
        return wfst

    def to_fst(self, copy=True):
        """ Copy (optional) and return wrapped pynini Fst. """
        if not copy:
            return self.fst
        return self.fst.copy()

    # Printing/drawing/saving/loading.

    def print_arc(self, q, t):
        """
        Pretty-print a single arc from state q.
        """
        q = self.state_label(q)
        ilabel = self.ilabel(t)
        olabel = self.olabel(t)
        weight = self.weight(t)
        dest = self.state_label(t.nextstate)
        return (q, ilabel, olabel, weight, dest)

    def print(self, show=True, **kwargs):
        fst = self.fst
        # Symbol table for state labels.
        state_symbols = pynini.SymbolTable()
        for q, label in self._state2label.items():
            state_symbols.add_symbol(str(label), q)
        ret = fst.print(isymbols=fst.input_symbols(),
                        osymbols=fst.output_symbols(),
                        ssymbols=state_symbols,
                        **kwargs)
        if show:
            print(ret)
        return ret

    def __str__(self):
        return self.print(show=False)

    def draw(self, source, acceptor=True, portrait=True, **kwargs):
        """ Write FST in dot format to file (source). """
        fst = self.fst
        # State symbol table.
        state_symbols = pynini.SymbolTable()
        for q, label in self._state2label.items():
            state_symbols.add_symbol(str(label), q)
        return fst.draw(source,
                        isymbols=fst.input_symbols(),
                        osymbols=fst.output_symbols(),
                        ssymbols=state_symbols,
                        acceptor=acceptor,
                        portrait=portrait,
                        **kwargs)

    def viz(self, **kwargs):
        """ Draw in ipython / jupyter notebook. """
        # todo: skip middleman file
        self.draw('.tmp.dot', **kwargs)
        ret = Source.from_file('.tmp.dot')
        return ret

    def save(self, outfile):
        """ Save to pickle file. """
        with open(outfile, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, infile):
        """ Load from pickle file. """
        with open(infile, 'rb') as f:
            wfst = pickle.load(f)
        return wfst


# # # # # # # # # #
# Machine constructors.
# todo: string_file, string_map


def accep(x, isymbols=None, add_delim=True, **kwargs):
    """
    Acceptor for space-delimited input (see pynini.accep).
    pynini.accep() arguments: weight (final weight) and 
    arc_type ("standard", "log", or "log64").
    todo: set isymbols with symbols in x
    """
    if not isinstance(x, str):
        x = ' '.join(x)
    if add_delim:
        x = f'{config.bos} {x} {config.eos}'

    if isymbols is None:
        isymbols = config.symtable
    if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
        isymbols, _ = config.make_symtable(isymbols)

    fst = pynini.accep(x, token_type=isymbols, **kwargs)
    fst.set_input_symbols(isymbols)
    fst.set_output_symbols(isymbols)
    wfst = Wfst.from_fst(fst)
    return wfst


def trans(ilabel, olabel, **kwargs):
    """
    Transducer that maps space-separated input string to
    space-separated output string. One-off alternative
    to pynini string_map(); use union() to combine multiple
    transducers as alteratives.
    todo: optionally add delimiters
    """
    if ilabel is None:
        ilabel = config.epsilon
    if olabel is None:
        olabel = config.epsilon
    ilabels = ilabel.split(' ')
    olabels = olabel.split(' ')
    ilength = len(ilabels)
    olength = len(olabels)
    if ilength < olength:
        ilabels += [config.epsilon] * (olength - ilength)
    if ilength > olength:
        olabels += [config.epsilon] * (ilength - olength)

    wfst = Wfst(**kwargs)
    n = len(ilabels)  # == len(olabels)
    src = wfst.add_state(initial=True)
    for (x, y) in zip(ilabels, olabels):
        dest = wfst.add_state()
        wfst.add_arc(src, x, y, None, dest)
        src = dest
    wfst.set_final(src)

    return wfst


def trellis(length=1,
            isymbols=None,
            tier=None,
            trellis=True,
            arc_type='standard'):
    """
    Acceptor for all strings up to specified length (trellis = True), 
    or of specified length (trellis = False), +2 for delimiters. 
    If tier is specified as a subset of the alphabet, makes 
    tier/projection acceptor for that subset with other symbols 
    labeling self-loops on interior states.
    """
    epsilon = config.epsilon
    bos = config.bos
    eos = config.eos

    # Input/output alphabet.
    if isymbols is None:
        isymbols = config.symtable
    if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
        isymbols, _ = config.make_symtable(isymbols)
    sigma = set([x for i, x in isymbols]) \
            - set([epsilon, bos, eos]) \
            - set(config.special_syms)

    # Subset of alphabet for tier.
    if tier is None:
        tier = sigma
        skip = set()
    else:
        tier = set(tier)
        skip = sigma - tier

    wfst = Wfst(isymbols, arc_type=arc_type)

    # Initial and peninitial states.
    q0 = wfst.add_state()  # id 0
    q1 = wfst.add_state()  # id 1
    wfst.set_start(q0)
    wfst.add_arc(src=q0, ilabel=bos, dest=q1)

    # Interior states.
    for l in range(length):
        wfst.add_state()  # ids 2, ...

    # Final state.
    qf = wfst.add_state()  # id (length+2)
    wfst.set_final(qf)

    # Zero-length form.
    if trellis:
        wfst.add_arc(src=q1, ilabel=eos, dest=qf)

    # Loop.
    for x in skip:
        wfst.add_arc(src=q1, ilabel=x, dest=q1)

    # Interior arcs.
    q = q1
    for l in range(1, length + 1):
        r = (l + 1)
        # Advance.
        for x in tier:
            wfst.add_arc(src=q, ilabel=x, dest=r)
        # Loop.
        # wfst.add_arc(src=r, ilabel=epsilon, dest=r)
        for x in skip:
            wfst.add_arc(src=r, ilabel=x, dest=r)
        # End.
        if trellis:
            wfst.add_arc(src=r, ilabel=eos, dest=qf)
        q = r

    wfst.add_arc(src=q, ilabel=eos, dest=qf)

    return wfst


def braid(length=1, isymbols=None, tier=None, arc_type='standard'):
    """
    Acceptor for strings of given length (+2 for delimiters).
    (see trellis())
    """
    return trellis(length, isymbols, tier, False, arc_type)


def empty_transducer(isymbols=None, osymbols=None, arc_type='standard'):
    """
    Starter transducer with three states and two arcs:
        0 -> bos:bos -> 0
        1 -> eos:eos -> 1
    isymbols and osymbols are lists of ordinary symbols
    (get epsilon, bos, eos from config).
    """
    wfst = Wfst(isymbols, osymbols, arc_type=arc_type)
    for i in range(3):
        wfst.add_state(i)
    wfst.set_initial(0)
    wfst.set_final(2)
    wfst.add_arc(0, config.bos, config.bos, None, 1)
    wfst.add_arc(1, config.eos, config.eos, None, 2)
    return wfst


def ngram(context='left',
          length=1,
          isymbols=None,
          tier=None,
          arc_type='standard'):
    """
    Acceptor (identity transducer) for segments in immediately 
    preceding (left) / following (right) / both-side contexts of 
    specified length. For both-side context, length can be tuple.
    ref. Wu, K., Allauzen, C., Hall, K. B., Riley, M., & Roark, B. (2014, September). Encoding linear models as weighted finite-state transducers.
    In INTERSPEECH (pp. 1258-1262).
    """
    if context == 'left':
        return ngram_left(length, isymbols, tier, arc_type)
    if context == 'right':
        return ngram_right(length, isymbols, tier, arc_type)
    if context == 'both':
        if isinstance(length, int):
            # Same context length on both sides.
            length_L = length_R = length
        else:
            # Independent context lengths.
            length_L, length_R = length
        L = ngram_left(length_L, isymbols, tier, arc_type)
        R = ngram_right(length_R, isymbols, tier, arc_type)
        LR = compose(L, R)
        return LR
    print(f'Bad side argument {side} to ngram_acceptor.')
    return None


def ngram_left(length=1, isymbols=None, tier=None, arc_type='standard'):
    """
    Acceptor (identity transducer) for segments in immediately 
    preceding contexts (histories) of specified length. If 
    tier is specified as a subset of the alphabet, only symbols 
    in tier are consumed by arcs and tracked in histories 
    (symbols not on the tier are skipped with self-loops on 
    each interior state).
    """
    epsilon = config.epsilon
    bos = config.bos
    eos = config.eos

    # Input/output alphabet.
    if isymbols is None:
        isymbols = config.symtable
    if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
        isymbols, _ = config.make_symtable(isymbols)
    sigma = set([x for i, x in isymbols]) \
            - set([epsilon, bos, eos]) \
            - set(config.special_syms)

    # Subset of alphabet for tier.
    if tier is None:
        tier = sigma
        skip = set()
    else:
        tier = set(tier)
        skip = sigma - tier

    wfst = Wfst(isymbols, arc_type=arc_type)

    # Initial and peninitial states; arc between them.
    q0 = ('λ', )
    q1 = (epsilon, ) * (length - 1) + (bos, )
    wfst.add_state(q0)
    wfst.set_start(q0)
    wfst.add_state(q1)
    wfst.add_arc(src=q0, ilabel=bos, dest=q1)

    # Interior arcs.
    # xα -- y --> αy for each y
    Q = {q0, q1}
    Qnew = set(Q)
    for l in range(length + 1):
        Qold = set(Qnew)
        Qnew = set()
        for q1 in Qold:
            if q1 == q0:
                continue
            for x in sigma:
                q2 = _suffix(q1, length - 1) + (x, )
                wfst.add_state(q2)
                wfst.add_arc(src=q1, ilabel=x, dest=q2)
                Qnew.add(q2)
        Q |= Qnew

    # Final state and incoming arcs.
    qf = (eos, )
    wfst.add_state(qf)
    wfst.set_final(qf)
    for q1 in Q:
        if q1 == q0:
            continue
        wfst.add_arc(src=q1, ilabel=eos, dest=qf)
    Q.add(qf)

    # Self-arcs labeled by skipped symbols
    # on interior states.
    for q in Q:
        if (q == q0) or (q == qf):
            continue
        for x in skip:
            wfst.add_arc(src=q, ilabel=x, dest=q)
        # wfst.add_arc(src=q, ilabel=epsilon, dest=q)

    return wfst


def ngram_right(length=1, isymbols=None, tier=None, arc_type='standard'):
    """
    Acceptor (identity transducer) for segments in immediately 
    following contexts (futures) of specified length.
    (see ngram_left() on tier handling)
    """
    epsilon = config.epsilon
    bos = config.bos
    eos = config.eos

    # Input/output alphabet.
    if isymbols is None:
        isymbols = config.symtable
    if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
        isymbols, _ = config.make_symtable(isymbols)
    sigma = set([x for i, x in isymbols]) \
            - set([epsilon, bos, eos]) \
            - set(config.special_syms)

    # Subset of alphabet for tier.
    if tier is None:
        tier = sigma
        skip = set()
    else:
        tier = set(tier)
        skip = sigma - tier

    wfst = Wfst(isymbols, arc_type=arc_type)

    # Initial state.
    q0 = (bos, )
    wfst.add_state(q0)
    wfst.set_start(q0)

    # Final and penultimate state.
    qf = ('λ', )
    qp = (eos, ) + (epsilon, ) * (length - 1)
    wfst.add_state(qf)
    wfst.set_final(qf)
    wfst.add_state(qp)
    wfst.add_arc(src=qp, ilabel=eos, dest=qf)

    # Interior arcs.
    # xα --> x --> αy for each y
    Q = {qf, qp}
    Qnew = set(Q)
    for l in range(length + 1):
        Qold = set(Qnew)
        Qnew = set()
        for q2 in Qold:
            if q2 == qf:
                continue
            for x in tier:
                q1 = (x, ) + _prefix(q2, length - 1)
                wfst.add_state(q1)
                wfst.add_arc(src=q1, ilabel=x, dest=q2)
                Qnew.add(q1)
        Q |= Qnew

    # Arcs from initial state.
    for q in Q:
        if q == qf:
            continue
        wfst.add_arc(src=q0, ilabel=bos, dest=q)
    Q.add(q0)

    # Self-arcs labeled by skipped symbols
    # on interior states.
    for q in Q:
        if (q == q0) or (q == qf):
            continue
        for x in skip:
            wfst.add_arc(src=q, ilabel=x, dest=q)

    return wfst


def _prefix(x, l):
    """ Length-l prefix of tuple x. """
    if l < 1:
        return ()
    if len(x) < l:
        return x
    return x[:l]


def _suffix(x, l):
    """ Length-l suffix of tuple x. """
    if l < 1:
        return ()
    if len(x) < l:
        return x
    return x[-l:]


# # # # # # # # # #
# Operations on pairs of machines.
# todo: cross(-product), difference


def compose(wfst1,
            wfst2,
            wfst1_arcs=None,
            wfst2_arcs=None,
            matchfunc1=None,
            matchfunc2=None,
            verbose=False):
    """
    Composition/intersection of two machines, retaining contextual 
    info from the original machines by labeling each state q = (q1, q2) 
    with (label(q1), label(q2)). Multiplies arc and final weights if
    machines have the same arc type. Unifies arc features; assumes
    that features for the two machines are disjoint.
    Optionally pass in organized arcs in either machine,
    precomputing organizations for faster repeated composition.
    Optionally apply functions to determine matching arc labels
    with matchfunc1(t1_olabel) == matchfunc2(t2_ilabel).
    """
    # Initialize result of composition.
    epsilon = config.epsilon
    common_weights = (wfst1.arc_type() == wfst2.arc_type())
    wfst = Wfst( \
        wfst1.input_symbols(),
        wfst2.output_symbols(),
        wfst1.arc_type() if common_weights else 'log')
    one = Weight.one(wfst.weight_type())
    zero = Weight.zero(wfst.weight_type())

    # Initial state (possibly also final).
    q1, q2 = wfst1.start(), wfst2.start()
    q0 = (q1, q2)
    wfst.add_state(q0, initial=True)
    wfinal1 = wfst1.final(q1)
    wfinal2 = wfst2.final(q2)
    # q0 is final iff both q1 and q2 are final.
    if wfinal1 != zero and wfinal2 != zero:
        wfinal = pynini.times(wfinal1, wfinal2) \
            if common_weights else one # checkme: or wfinal2?
        wfst.set_final(q0, wfinal)

    # Lazy organization of arcs in each machine by src
    # and label for fast matching and handling of implicit
    # epsilon:epsilon self-transitions.
    if not wfst1_arcs:
        wfst1_arcs = {}
    if not wfst2_arcs:
        wfst2_arcs = {}

    # Lazy state and arc construction of wfst.
    Q = set([(q1, q2, 0)])
    Q_old, Q_new = set(), Q.copy()
    while len(Q_new) != 0:
        Q_old, Q_new = Q_new, Q_old
        Q_new.clear()

        # Source states.
        for (src1, src2, q3) in Q_old:
            src = (src1, src2)  # Source label.
            src_id = wfst.state_id(src)  # Source id.
            src1_id = wfst1.state_id(src1)  # Source id in wfst1.
            src2_id = wfst2.state_id(src2)  # Source id in wfst2.
            if verbose: print(src)

            # Organize arcs from src1 by matchfunc1(olabel),
            # or use existing organization.
            if src1_id not in wfst1_arcs:
                wfst1_arcs[src1_id] = organize_arcs( \
                    wfst1, src1_id, matchfunc1, 'output')
            src1_arcs = wfst1_arcs[src1_id]
            src1_arciter = wfst1.fst.arcs(src1_id)
            if verbose: print(src1_arcs)

            # Organize arcs from src2 by matchfunc2(ilabel),
            # or use existing organization.
            if src2_id not in wfst2_arcs:
                wfst2_arcs[src2_id] = organize_arcs( \
                    wfst2, src2_id, matchfunc2, 'input')
            src2_arcs = wfst2_arcs[src2_id]
            src2_arciter = wfst2.fst.arcs(src2_id)
            if verbose: print(src2_arcs)

            # Process arc pairs with matching labels.
            for t1_olabel, src1_arcids in src1_arcs.items():
                if verbose: print(t1_olabel)
                src2_arcids = src2_arcs.get(t1_olabel, None)
                if src2_arcids is None:
                    continue

                for (t1_idx, t2_idx) in \
                    itertools.product(src1_arcids, src2_arcids):
                    # Get arcs by position / make epsilon self-arcs.
                    if t1_idx >= 0:
                        src1_arciter.seek(t1_idx)
                        t1 = src1_arciter.value()
                    else:
                        _, t1 = wfst1.make_epsilon_arc(src1_id)

                    if t2_idx >= 0:
                        src2_arciter.seek(t2_idx)
                        t2 = src2_arciter.value()
                    else:
                        _, t2 = wfst2.make_epsilon_arc(src2_id)

                    # Apply composition filter.
                    q3_ = epsilon_filter(src1_id, t1, src2_id, t2, q3)
                    if q3_ == '⊥':
                        continue

                    # Compose arcs.
                    t1_ilabel = wfst1.ilabel(t1)  # Input label.
                    dest1_id = t1.nextstate  # Destination id.
                    dest1 = wfst1.state_label(dest1_id)  # Destination label.
                    wfinal1 = wfst1.final(dest1_id)  # Final weight.
                    phi_t1 = wfst1.features(src1_id, t1)  # Arc features.

                    t2_olabel = wfst2.olabel(t2)  # Output label.
                    dest2_id = t2.nextstate  # Destination id.
                    dest2 = wfst2.state_label(dest2_id)  # Destination label.
                    wfinal2 = wfst2.final(dest2)  # Final weight.
                    phi_t2 = wfst2.features(src2_id, t2)  # Arc features.

                    # Destination state.
                    dest = (dest1, dest2)  # Destination label
                    dest_id = wfst.add_state(dest)  # Destination id.

                    # Multiply weights.
                    if common_weights:
                        weight = pynini.times(t1.weight, t2.weight)
                    else:
                        weight = one  # todo: or t2.weight?

                    # Dest is final if both dest1 and dest2 are final.
                    if wfinal1 != zero and wfinal2 != zero:
                        if common_weights:
                            wfinal = pynini.times(wfinal1, wfinal2)
                        else:
                            wfinal = one  # or wfinal2?
                        wfst.set_final(dest, wfinal)

                    # Enqueue new state.
                    q = (dest1, dest2, q3_)
                    if q not in Q:
                        Q.add(q)
                        Q_new.add(q)

                    # Add arc.
                    wfst.add_arc(src=src,
                                 ilabel=t1.ilabel,
                                 olabel=t2.olabel,
                                 weight=weight,
                                 dest=dest)

                    # Arc features: union of features assigned
                    # to source arcs (ignoring {}).
                    if phi_t1 or phi_t2:
                        phi_t = phi_t1 | phi_t2
                        t_ = (src_id, t1.ilabel, t2.olabel, dest_id)
                        wfst.phi[t_] = phi_t

    wfst = wfst.connect()
    return wfst


def organize_arcs(wfst, src=None, matchfunc=None, side='input', verbose=False):
    """
    Organize arcs by source state and input or output label 
    (optionally passed through matchfunc) for faster composition.
    Creates map: src -> matchfunc(label) -> array of arc indices
    (using arc indices/positions instead of arc references because
    references are unstable across different arc iterators).
    Adds implicit epsilon arcs to each state.
    Useful to call before repeated composition with a machine, passing
    the result to compose(), otherwise called by compose() itself 
    as needed during composition.
    note: changes to machine topology invalidate the organization.
    (see pynini.arcsort)
    """
    # Organize arcs from all states.
    if src is None:
        wfst_arcs = { \
            src:organize_arcs(wfst, src, matchfunc, side)
            for src in wfst.state_ids()}
        return wfst_arcs

    # Organize arcs from one state.
    src = wfst.state_id(src)
    src_arcs = {}
    # Organize arcs by input or output label,
    # passed through matchfunc.
    for idx, t in enumerate(wfst.arcs(src)):
        label = None
        if side == 'input':
            label = wfst.ilabel(t)
        elif side == 'output':
            label = wfst.olabel(t)
        if matchfunc:
            label = matchfunc(label)
        if label in src_arcs:
            src_arcs[label].append(idx)
        else:
            src_arcs[label] = [idx]
    # Implicit epsilon self-transition with pseudo-index -1.
    # (see https://www.openfst.org/doxygen/fst/html/compose_8h_source.html)
    if config.epsilon in src_arcs:
        src_arcs[config.epsilon].append(-1)
    else:
        src_arcs[config.epsilon] = [-1]
    if verbose: print(src, src_arcs)

    return src_arcs


def compose_sorted(wfst1, wfst2):
    """
    Composition/intersection of two machines, as for compose()
    but assuming that:
    (i) output symbol table of wfst1 is identical to input 
    symbol table of wfst2,
        wfst1.output_symbols() == wfst2.input_symbols();
    (ii) arcs from each state in wfst1 and wfst2 are sorted
    on the matching side (output for wfst1, input for wfst2).
    (see pynini.arcsort, OpenFst compose)
    todo: check conditions (i) and (ii)
    """
    # Initialize result of composition.
    epsilon = 0  # by convention, config.epsilon id
    common_weights = (wfst1.arc_type() == wfst2.arc_type())
    wfst = Wfst( \
        wfst1.input_symbols(),
        wfst2.output_symbols(),
        wfst1.arc_type() if common_weights else 'log')
    one = Weight.one(wfst.weight_type())
    zero = Weight.zero(wfst.weight_type())

    # Initial state (possibly also final).
    q1, q2 = wfst1.start(), wfst2.start()
    q0 = (q1, q2)
    wfst.add_state(q0, initial=True)
    wfinal1 = wfst1.final(q1)
    wfinal2 = wfst2.final(q2)
    # q0 is final iff both q1 and q2 are final.
    if wfinal1 != zero and wfinal2 != zero:
        wfinal = pynini.times(wfinal1, wfinal2) \
            if common_weights else one # checkme: or wfinal2?
        wfst.set_final(q0, wfinal)

    # Lazy state and arc construction of wfst.
    Q = set([(q1, q2, 0)])
    Q_old, Q_new = set(), Q.copy()
    match_func = lambda t2: t2.ilabel  # Arc matching.
    while len(Q_new) != 0:
        Q_old, Q_new = Q_new, Q_old
        Q_new.clear()

        # Source states.
        for (src1, src2, q3) in Q_old:
            src = (src1, src2)  # Source label.
            src_id = wfst.state_id(src)  # Source id.
            src1_id = wfst1.state_id(src1)  # Source id in wfst1.
            src2_id = wfst2.state_id(src2)  # Source id in wfst2.
            if verbose: print(src)

            # Process arc pairs with matching labels.
            src1_arcs = [wfst1.make_epsilon_arc(src1_id)[1]] + \
                list(wfst1.arcs(src1_id))
            src2_arcs = [wfst2.make_epsilon_arc(src2_id)[1]] + \
                list(wfst2.arcs(src2_id))
            t2_lo = 0
            t2_max = len(src2_arcs)
            t1_olabel_old = None
            for t1 in src1_arcs:
                t1_olabel = t1.olabel  # Output label.

                # Search for matching arcs in wfst2.
                if t1_olabel != t1_olabel_old:
                    t2_lo = bisect.bisect_left(src2_arcs,
                                               t1_olabel,
                                               lo=t2_lo,
                                               key=match_func)
                    t1_olabel_old = t1_olabel

                # No matching arcs found.
                if t2_lo >= t2_max:
                    break

                # Arc attributes in wfst1.
                t1_ilabel = t1.ilabel  # Input label.
                dest1_id = t1.nextstate  # Destination id.
                dest1 = wfst1.state_label(dest1_id)  # Destination label.
                wfinal1 = wfst1.final(dest1_id)  # Final weight.
                phi_t1 = wfst1.features(src1_id, t1)  # Arc features.

                # Process each matching arc.
                for t2_idx in range(t2_lo, t2_max):
                    t2 = src2_arcs[t2_idx]
                    if t2.ilabel != t1_olabel:
                        break

                    # Arc attributes in wfst2.
                    t2_olabel = t2.olabel  # Output label.
                    dest2_id = t2.nextstate  # Destination id.
                    dest2 = wfst2.state_label(dest2_id)  # Destination label.
                    wfinal2 = wfst2.final(dest2_id)  # Final weight.
                    phi_t2 = wfst2.features(src2_id, t2)  # Arc features.

                    # Apply composition filter.
                    q3_ = epsilon_filter(src1_id, t1, src2_id, t2, q3)
                    if q3_ == '⊥':
                        continue

                    # Destination state.
                    dest = (dest1, dest2)  # Destination label.
                    dest_id = wfst.add_state(dest)  # Destination id.

                    # Multiply weights.
                    if common_weights:
                        weight = pynini.times(t1.weight, t2.weight)
                    else:
                        weight = one  # or t2.weight?

                    # Dest is final if both dest1 and dest2 are final.
                    if wfinal1 != zero and wfinal2 != zero:
                        if common_weights:
                            wfinal = pynini.times(wfinal1, wfinal2)
                        else:
                            wfinal = one  # or wfinal2?
                        wfst.set_final(dest, wfinal)

                    # Enqueue new state triple.
                    q = (dest1, dest2, q3_)
                    if q not in Q:
                        Q.add(q)
                        Q_new.add(q)

                    # Add arc.
                    wfst.add_arc(src=src,
                                 ilabel=t1.ilabel,
                                 olabel=t2.olabel,
                                 weight=weight,
                                 dest=dest)

                    # Arc features: union of features assigned
                    # to source arcs (ignoring {}).
                    if phi_t1 or phi_t2:
                        phi_t = phi_t1 | phi_t2
                        t_ = (src_id, t1.ilabel, t2.olabel, dest_id)
                        wfst.phi[t_] = phi_t

    wfst = wfst.connect()
    return wfst


def epsilon_filter(q1, t1, q2, t2, q3):
    """
    Compute next state of epsilon-matching composition filter.
    Assumes that t1.olabel and t2.ilabel have been determined
    to match.
    ref. Allauzen, Riley, & Schalkwyk (2009). In INTERSPEECH.
    """
    epsilon = 0
    # Non-epsilon labels.
    if t1.olabel != epsilon and t2.ilabel != epsilon:
        return 0
    # Epsilon labels on non-self-transitions.
    if (t1.olabel == epsilon and q1 != t1.nextstate) and \
        (t2.ilabel == epsilon and q2 != t2.nextstate) and q3 == 0:
        return 0
    # Epsilon self-transition in wfst1 but not wfst2.
    if (t1.olabel == epsilon and q1 == t1.nextstate) and \
        (t2.ilabel == epsilon and q2 != t2.nextstate) and q3 != 2:
        return 1
    # Epsilon self-transition in wfst2 but not wfst12.
    if (t1.olabel == epsilon and q1 != t1.nextstate) and \
        (t2.ilabel == epsilon and q2 == t2.nextstate) and q3 != 1:
        return 2
    # Note: epsilon self-transitions in both wfst1 and wfst2
    # result in bottom.
    return '⊥'


def concat(wfst1, wfst2):
    """
    Concatenation of two machines, assumed to share the 
    same input/output symbol tables and arc type.
    """
    wfst = Wfst( \
        wfst1.input_symbols(),
        wfst1.output_symbols(),
        wfst1.arc_type())
    one = Weight.one(wfst.weight_type())

    # States and arcs from wfst1.
    for q in wfst1.states():
        wfst.add_state((q, 1))
    wfst.set_initial((wfst1.initial(), 1))
    for q in wfst1.states():
        for t in wfst1.transitions(q):
            wfst.add_arc( \
                (q, 1),
                wfst1.ilabel(t),
                wfst1.olabel(t),
                wfst1.weight(t),
                (wfst1.state_label(t.nextstate), 1))

    # States and arcs from wfst2.
    for q in wfst2.states():
        wfst.add_state((q, 2))
    for q in wfst2.finals():
        wfst.set_final((q, 2), wfst2.final_weight(q))
    for q in wfst2.states():
        for t in wfst2.transitions(q):
            wfst.add_arc( \
                (q, 2),
                wfst2.ilabel(t),
                wfst2.olabel(t),
                wfst2.weight(t),
                (wfst2.state_label(t.nextstate), 2))

    # Bridging arcs.
    for q1 in wfst1.finals():
        wfst.add_arc( \
            (q1, 1),
            config.epsilon,
            config.epsilon,
            one,
            (wfst2.initial(), 2))

    # todo: remove epsilons / minimize
    return wfst


# Alias for concat().
concatenate = concat


def union(wfst1, wfst2):
    """
    Union of two machines, assumed to share the 
    same input/output symbol tables and arc type.
    """
    wfst = Wfst( \
        wfst1.input_symbols(),
        wfst1.output_symbols(),
        wfst1.arc_type())
    one = Weight.one(wfst.weight_type())

    q0 = wfst.add_state(initial=True)

    # States and arcs from wfst1.
    for q in wfst1.states():
        wfst.add_state((q, 1))
    for q in wfst1.finals():
        wfst.set_final((q, 1), wfst1.final_weight(q))
    for q in wfst1.states():
        for t in wfst1.transitions(q):
            wfst.add_arc( \
                (q, 1),
                wfst1.ilabel(t),
                wfst1.olabel(t),
                wfst1.weight(t),
                (wfst1.state_label(t.nextstate), 1))

    # States and arcs from wfst2.
    for q in wfst2.states():
        wfst.add_state((q, 2))
    for q in wfst2.finals():
        wfst.set_final((q, 2), wfst2.final_weight(q))
    for q in wfst2.states():
        for t in wfst2.transitions(q):
            wfst.add_arc( \
                (q, 2),
                wfst2.ilabel(t),
                wfst2.olabel(t),
                wfst2.weight(t),
                (wfst2.state_label(t.nextstate), 2))

    # Bridging arcs.
    q1 = (wfst1.initial(), 1)
    q2 = (wfst2.initial(), 2)
    wfst.add_arc(q0, config.epsilon, config.epsilon, one, q1)
    wfst.add_arc(q0, config.epsilon, config.epsilon, one, q2)

    return wfst


def ques(wfst):
    """ Optionality. """
    wfst = wfst.copy()
    one = Weight.one(wfst.weight_type())
    q0 = wfst.initial()
    qf = wfst.add_state(final=True)
    wfst.add_arc( \
        q0,
        config.epsilon,
        config.epsilon,
        one,
        qf)
    return wfst


def plus(wfst):
    """ Plus operator. """
    wfst = wfst.copy()
    one = Weight.one(wfst.weight_type())
    q0 = wfst.initial()
    for qf in wfst.finals():
        wfst.add_arc( \
            qf,
            config.epsilon,
            config.epsilon,
            one,
            q0)
    return wfst


def star(wfst):
    """ Repetition. """
    wfst = wfst.copy()
    one = Weight.one(wfst.weight_type())
    q0 = wfst.initial()
    wfst.set_final(q0, one)
    for qf in wfst.finals():
        wfst.add_arc( \
            qf,
            config.epsilon,
            config.epsilon,
            one,
            q0)
    return wfst


# # # # # # # # # #
# Shortest distance / shortest paths.


def shortestdistance(wfst, delta=1e-6, reverse=False):
    """
    'Shortest distance' from the initial state to each
    state (reverse=False, the default) or from each 
    state into the final states (reverse=True).
    Pynini doc:
    "The shortest distance from p to q is the \otimes-sum of 
    the weights of all the paths between p and q."
    ref. Mohri, M. (2002). Semiring frameworks and algorithms for 
    shortest-distance problems. Journal of Automata, Languages 
    and Combinatorics, 7(3), 321-350.
    """
    return pynini.shortestdistance( \
        wfst.fst, delta=delta, reverse=reverse)


def shortestpath(wfst, delta=1e-6, ret_type='wfst', **kwargs):
    """
    Return Fst/Wfst containing shortest paths only -or- output 
    strings / io strings of that machine.
    Pynini doc:
    "Construct an FST containing the shortest path(s) in the 
    input FST.
    shortestpath(ifst, delta=1e-6, nshortest=1, nstate=NO_STATE_ID,
    queue_type="auto", unique=False, weight=None)
    [Gorman & Sproat, section 5.3.2]"
    note: ensure weights are in tropical semiring before 
    calling (e.g., using wfst.map_weights('to_std')).
    note: state labels / output strings / arc features
    of input wfst are not preserved in output machine; even
    state ids may not be preserved.
    """
    fst = wfst.fst
    isymbols = fst.input_symbols()
    osymbols = fst.output_symbols()

    fst_out = pynini.shortestpath( \
        fst, delta=delta) #, **kwargs)
    fst_out.topsort()  # checkme
    fst_out.set_input_symbols(isymbols)
    fst_out.set_output_symbols(osymbols)

    # Return path machine or strings.
    if ret_type == 'fst':
        return fst_out
    elif ret_type in ('ostrings', 'outputs'):
        path_iter = fst_out.paths(input_token_type=isymbols,
                                  output_token_type=osymbols)
        ostrings = list(path_iter.ostrings())
        return ostrings
    elif ret_type == 'iostrings':
        wfst_out = Wfst.from_fst(fst_out)
        iostrings = wfst_out.iostrings()
        return iostrings
    # default: return wfst representing shortest paths
    wfst_out = Wfst.from_fst(fst_out)
    wfst_out.relabel_states()
    return wfst_out


def shortestpath_(wfst, delta=1e-6):
    """
    Version of shortestpath that retains state labels / 
    output strings / arc features of input wfst.
    note: ensure weights are in tropical semiring before 
    calling (e.g., using wfst.map_weights('to_std')).
    todo: ret_type argument
    """
    fst = wfst.fst
    dist = pynini.shortestdistance( \
        fst, delta, reverse=True)

    dead_arcs = []
    for q in fst.states():
        for t in fst.arcs(q):
            w = pynini.times(t.weight, dist[t.nextstate])
            if np.abs(float(w) - float(dist[q])) > delta:
                dead_arcs.append((q, t))

    wfst_out = wfst.copy()
    wfst_out = wfst_out.delete_arcs(dead_arcs)
    wfst_out = wfst_out.connect()
    return wfst_out


# # # # # # # # # #
# Utility.


def arc_equal(arc1, arc2):
    """
    Test equality of arcs from the same src
    (missing from Pynini?).
    """
    val = (arc1.ilabel == arc2.ilabel) and \
            (arc1.olabel == arc2.olabel) and \
            (arc1.nextstate == arc2.nextstate) and \
            (arc1.weight == arc2.weight)
    return val


# todo: require symbol tables for input and output labels
# to be initialized outside of Wfst instances, allowing epsilon /
# bos / eos / other special symbols to be set on a per-machine
# basis; route access to epsilon / bos / eos / etc. through
# machines instead of global config

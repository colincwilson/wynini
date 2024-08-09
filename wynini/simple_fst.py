from copy import copy
from functools import total_ordering
from pynini import SymbolTable

from wynini import config, Wfst


class SimpleFst():
    """
    Bare-bones unweighted FST implementation.
    """

    def __init__(self, Q=None, q0=None, F=None, T=None):
        self.Q = set(Q) if Q else set()  # States.
        self.q0 = q0 if q0 else -1  # Initial state.
        self.F = set(F) if F else set()  # Final states.
        # Mapping state -> outgoing arcs (outgoing arc collection
        # is Set [default] or List).
        self.T = T if T else {}

    def add_state(self, q):
        """
        Add to set of states.
        """
        if q not in self.Q:
            self.Q.add(q)
            self.T[q] = set()

    def set_start(self, q):
        """
        Set unique start state to q (adding q to state 
        set if not already present).
        """
        self.add_state(q)
        self.q0 = q

    def set_final(self, q):
        """
        Add q to set of final states (adding q to state 
        set if not already present).
        """
        self.add_state(q)
        self.F.add(q)

    def add_arc(self, t):
        """
        Add arc (adding src/dest to state set if not 
        already present).
        """
        self.add_state(t.src)
        self.add_state(t.dest)
        if t.src not in self.T:
            self.T[t.src] = set()
        if isinstance(self.T[t.src], set):
            self.T[t.src].add(t)
        else:
            self.T[t.src].append(t)

    def delete_states(self, dead_states):
        """
        Delete states and their outgoing/incoming arcs.
        [nondestructive]
        """
        Q = self.Q.difference(dead_states)
        q0 = self.q0 if self.q0 not in dead_states else -1
        F = self.F.difference(dead_states)
        for q in dead_states:
            del T[q]
        dead_arcs = []
        for q in T:
            dead_arcs += [t for t in T[q] if t.dest in dead_states]
        for t in dead_arcs:
            T[q].remove(t)
        fst = SimpleFst(Q, q0, F, T)
        return fst

    def copy(self):
        """
        Deep copy of this machine.
        """
        Q = {q for q in self.Q}
        q0 = self.q0
        F = {q for q in self.F}
        T = {}
        for q, T_q in self.T.items():
            T[q] = set()
            for t in T_q:
                t_new = SimpleArc(t.src, t.ilabel, t.olabel, t.dest)
                T[q].add(t_new)
        return SimpleFst(Q, q0, F, T)

    def print(self):
        """
        String representations of Q, q0, F, T.
        """
        val = f'Q {self.Q}\n'
        val += f'q0 {self.q0}\n'
        val += f'F {self.F}\n'
        _T = []
        for q in self.T:
            _T += list(self.T[q])
        val += f'T {[str(t) for t in _T]}\n'
        return val

    def to_wfst(self):
        """
        Convert to Wfst.
        """
        # Initialize symbol table.
        input_symbols = SymbolTable()
        output_symbols = SymbolTable()
        input_symbols.add_symbol(config.epsilon)
        output_symbols.add_symbol(config.epsilon)
        wfst = Wfst(input_symbols, output_symbols)

        # States.
        for q in self.Q:
            wfst.add_state(q)

        # Initial and final states.
        wfst.set_start(self.q0)
        for q in self.F:
            wfst.set_final(q)

        # Transitions.
        for q in self.T:
            for t in self.T[q]:
                wfst.add_arc(t.src, t.ilabel, t.olabel, None, t.dest)
        return wfst


@total_ordering
class SimpleArc():
    """
    Arc of SimpleFST.
    """

    def __init__(self, src, ilabel, olabel, dest):
        self.src = src
        self.ilabel = ilabel
        self.olabel = olabel
        self.dest = dest

    def __copy__(self):
        q, a, w, r = \
            self.src, self.ilabel, self.olabel, self.dest
        return SimpleArc(copy(q), copy(a), copy(w), copy(r))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.src == other.src) and (self.ilabel == other.ilabel) and (
            self.olabel == other.olabel) and (self.dest == other.dest)

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise Error('Incorrect type for SimpleArc lt()')
        return (self.src, self.ilabel, self.olabel,
                self.dest) < (other.src, other.ilabel, other.olabel,
                              other.dest)

    def __hash__(self):
        return hash((self.src, self.ilabel, self.olabel, self.dest))

    def __str__(self):
        return f'({self.src}, {self.ilabel}, {self.olabel}, {self.dest})'

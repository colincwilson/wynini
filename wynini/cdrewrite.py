# (Weighted) Conditional rewriting as in Mohri & Sproat (1996).
# see: pynini/extensions/cdrewritescript.cc
import re, sys
import string
from pynini import SymbolTable, SymbolTableView

import wynini
from wynini import *
from wynini import Wfst
from regexp import Thompson

markers = ['_#_', '_<1_', '_<2_', '_>_']


class CDRewrite():
    """
    Compiler for conditional rewrite rules.
    """

    def __init__(self, sigma):
        # Make symbol table that includes markers.
        if isinstance(sigma, (SymbolTable, SymbolTableView)):
            sigma = [sym for (sym_id, sym) in sigma]
        self.sigma = sigma
        config.init({'special_symbols': markers})
        self.isymbols, _ = config.make_symtable(sigma)
        # Regexp compiler.
        self.regexper = Thompson(self.isymbols)

    def rule(self, phi, psi, lamb, rho, replace=None):
        """
        "A transducer corresponding to the left-to-right
        obligatory rule phi -> psi / lambda __ rho can be
        obtained by composition of five transducers:
        r * f * replace * l1 * l2
        """
        regexper = self.regexper

        # "1. The transducer r introduces in a string
        # a marker > before every instance of rho.
        r_alpha = regexper.sigma_star_regexp(rho[::-1], sigma=self.sigma)
        r = marker(r_alpha, type=1, insertions=['_>_']).reverse()

        # "2. The transducer f introduces markers <1 and <2
        # before each instance of phi that is followed by > ....
        # In other words, this transducer marks just those phi
        # that occur before rho."
        f_alpha = regexper.dot( \
            regexper.sigma_star(sigma=(self.sigma + ['_>_'])),
            regexper.dot(
                regexper.to_wfst(phi).add_self_transitions('_>_'),
                regexper.to_wfst('_>_')
                ).reverse()
            )
        f_alpha = f_alpha.determinize()
        f = marker(f_alpha, type=1, insertions=['_<1_', '_<2_'])

        # "3. The replacement transducer _replace_ replaces
        # phi with psi in the context <1 phi >, simultaneously
        # deleting > in all positions. Since >, <1, and <2 need
        # to be ignored when determining an occurrence of phi,
        # there are loops over the transitions >:eps, <1:eps,
        # <2:eps at all states of phi, or equivalently of the
        # states of the cross product transducer phi x psi."
        if not replace:
            # Note: use string_map to implement mapping.
            # Build replace arg separately for cross-product,
            # weighted mapping, etc.
            pass  # xxx fixme

        # "The transducer l1 admits only those strings in which
        # occurrences of <1 are precede by lambda and deletes <1
        # at such occurrences."
        l1_alpha = regexper.sigma_star_regexp(lamb, sigma=self.sigma)
        l1 = marker(l1_alpha, type=2, deletions=['_<1_'])
        l1.add_self_transitions(['_<2_'])

        # "The transducer l2 admits only those strings in which
        # occurrences of <2 are not preceded by lambda and deletes
        # <2 at such occurrences."
        l2_alpha = regexper.sigma_star_regexp(lamb, sigma=self.sigma)
        l2 = marker(l2_alpha, type=3, deletions=['_<2_'])

        # Rewrite rule derived by composition.
        rule = r.compose(f).compose(replace).compose(l1).compose(l2)
        #rule = rule.determinize()
        return rule

    def sigma_star_regexp(self, beta, sigma=None, add_delim=False):
        wfst = self.regexper.sigma_star_regexp(beta, sigma, add_delim)
        return wfst

    def marker(self, alpha=None, type=1, insertions=[], deletions=[]):
        """
        Transducers that introduce/remove markers.
        """
        match type:
            case 1:
                tau = self.marker_type1(alpha, insertions, deletions)
            case 2:
                tau = self.marker_type2(alpha, insertions, deletions)
            case 3:
                tau = self.marker_type3(alpha, insertions, deletions)
            case _:
                print(f'Unrecognized marker type ({type}).')
                tau = None
        return tau

    def marker_type1(self, alpha, insertions=['_#_'], deletions=[]):
        """
        Transducer that inserts a marker after all prefixes of a
        string that match the regexp beta (represented by FSA alpha).
        """
        # Split final states and transitions that introduce/delete markers.
        tau = alpha.copy()  # xxx copy needed for debugging only
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        for q in states:
            if q in finals:
                # Split final state.
                tau.set_final(q, False)
                q_ = tau.add_state(initial=tau.is_initial(q), final=True)
                # Reroute outgoing transitions.
                for t in tau.arcs(q):
                    tau.add_arc(q_, t.ilabel, t.olabel, t.weight, t.nextstate)
                tau.delete_arcs(states=[q])
                # Introduce / delete markers.
                for marker in insertions:
                    tau.add_arc(q, config.epsilon, marker, None, q_)
                for marker in deletions:
                    tau.add_arc(q, marker, config.epsilon, None, q_)
            else:
                # Make non-final state final.
                tau.set_final(q, True)
        return tau

    def marker_type2(self, alpha, insertions=[], deletions=['_#_']):
        """
        Filter transducer that checks whether each occurrence of marker
        is preceded (or followed) by an occurrence of regexp beta.
        """
        tau = alpha.copy()  # xxx
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        for q in finals:
            for marker in insertions:
                tau.add_arc(q, config.epsilon, marker, None, q)
            for marker in deletions:
                tau.add_arc(q, marker, config.epsilon, None, q)
        for q in states:
            tau.set_final(q, True)
        return tau

    def marker_type3(self, alpha, insertions=[], deletions=['_#_']):
        """
        Filter transducer that checks whether each occurrence of marker
        is *not* preceded (or followed) by an occurrence of regexp beta.
        """
        tau = alpha.copy()  # xxx
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        for q in states:
            if q in finals:
                continue
            for marker in insertions:
                tau.add_arc(q, config.epsilon, marker, None, q)
            for marker in deletions:
                tau.add_arc(q, marker, config.epsilon, None, q)
        for q in states:
            tau.set_final(q, True)
        return tau


if __name__ == "__main__":
    sigma = ['a', 'b', 'c']
    cdrewrite = CDRewrite(sigma)

    beta1 = '(a|b)'
    alpha1 = cdrewrite.sigma_star_regexp(beta1)
    tau1 = cdrewrite.marker(alpha1, type=1, insertions=['_#_'])
    tau1.draw('fig/tau1.dot', acceptor=False)

    beta2 = '(a|b)'
    alpha2 = cdrewrite.sigma_star_regexp(beta2)
    tau2 = cdrewrite.marker(alpha2, type=2, deletions=['_#_'])
    tau2.draw('fig/tau2.dot', acceptor=False)

    tau3 = cdrewrite.marker(alpha2, type=3, deletions=['_#_'])
    tau3.draw('fig/tau3.dot', acceptor=False)

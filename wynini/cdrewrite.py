# (Weighted) Conditional rewriting as in Mohri & Sproat (1996).
# reference implementation: pynini/extensions/cdrewrite.h
import re, sys
import string
from pynini import SymbolTable, SymbolTableView

import wynini
from wynini import *
from wynini import Wfst
from wynini.regexp import Thompson
from wynini import loglinear

markers = ['_<1_', '_<2_', '_>_', '_#_']  # Markers used internally.


class CDRewrite():
    """
    Compiler for conditional rewrite rules and constraints.
    To compile a constraint, pass transducer 'replace'
    with weights or loglinear features on targeted arcs.
    """

    def __init__(self, sigma):
        # Symbol table that includes markers.
        if isinstance(sigma, (SymbolTable, SymbolTableView)):
            sigma = [sym for (sym_id, sym) in sigma]
        self.sigma = sigma
        config.init({'special_syms': markers})
        self.isymbols, _ = config.make_symtable(sigma)
        # Regexp compiler.
        self.regexper = Thompson(self.isymbols, self.sigma)

    def to_rule(self,
                phi,
                psi,
                lam,
                rho,
                replace=None,
                simplify=True,
                verbose=False):
        """
        "A transducer corresponding to the left-to-right
        obligatory rule phi -> psi / lambda __ rho can be
        obtained by composition of five transducers:
        r * f * replace * l1 * l2
        Arguments
            phi, psi: list/tuple inputs for string_map
            lam: regexp string (use empty string for wildcard)
            rho: regexp string (use empty string for wildcard)
            replace: precompiled phi -> psi transducer
        """
        regexper = self.regexper

        # "1. The transducer r introduces in a string
        # a marker > before every instance of rho.
        r_alpha = self.sigma_star_regexp(rho[::-1], sigma=self.sigma)
        r = self.marker(r_alpha, type=1, insertions=['_>_'])
        r = r.reverse()
        if verbose:
            print(r.info())
            r.draw('fig/r_alpha.dot', acceptor=False)
            r.draw('fig/r.dot', acceptor=False)

        # "2. The transducer f introduces markers <1 and <2
        # before each instance of phi that is followed by > ...
        # In other words, this transducer marks just those phi
        # that occur before rho."
        f_alpha = regexper.dot( \
            wynini.sigma_star( \
                isymbols=self.isymbols, sigma=(self.sigma + ['_>_'])),
            regexper.dot(
                regexper.to_wfst(phi).add_self_arcs('_>_'),
                regexper.to_wfst('_>_')
                ).reverse()
            )
        f_alpha = f_alpha.determinize()
        f = self.marker(f_alpha, type=1, insertions=['_<1_', '_<2_'])
        f = f.reverse()
        if verbose:
            print(f.info())
            f.draw('fig/f.dot', acceptor=False)

        # "3. The replacement transducer _replace_ replaces
        # phi with psi in the context <1 phi >, simultaneously
        # deleting > in all positions. Since >, <1, and <2 need
        # to be ignored when determining an occurrence of phi,
        # there are loops over the transitions >:eps, <1:eps,
        # <2:eps at all states of phi, or equivalently of the
        # states of the cross product transducer phi x psi."
        if not replace:
            replace = wynini.string_map( \
                phi, psi,
                isymbols=self.isymbols,
                add_delim=False)
        replace = self.replace_with_markers(replace)
        if verbose:
            print(replace.info())
            replace.draw('fig/replace.dot', acceptor=False)

        # "The transducer l1 admits only those strings in which
        # occurrences of <1 are preceded by lambda and deletes <1
        # at such occurrences."
        l1_alpha = self.sigma_star_regexp(lam, sigma=self.sigma)
        l1 = self.marker(l1_alpha, type=2, deletions=['_<1_'])
        l1.add_self_arcs(['_<2_'])
        if verbose:
            print(l1.info())
            l1.draw('fig/l1.dot', acceptor=False)

        # "The transducer l2 admits only those strings in which
        # occurrences of <2 are not preceded by lambda and deletes
        # <2 at such occurrences."
        l2_alpha = self.sigma_star_regexp(lam, sigma=self.sigma)
        l2 = self.marker(l2_alpha, type=3, deletions=['_<2_'])
        if verbose:
            print(l2.info())
            l2.draw('fig/l2.dot', acceptor=False)

        # Rewrite rule derived by composition.
        weight_type = replace.weight_type()
        for m in [r, f, l1, l2]:
            m.map_weights(weight_type)

        rule = r.compose(f).compose(replace).compose(l1).compose(l2)
        rule = rule.relabel_states().connect()
        if simplify:
            rule = rule.determinize(acceptor=False)
        if verbose:
            print(rule.info())
            rule.draw('fig/rule.dot', acceptor=False, show_weight_one=True)

        return rule, (r, f, replace, l1, l2)

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
        # Split final states and add transitions for
        # inserted/deleted markers.
        tau = alpha.copy()  # xxx copy needed for debugging only
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        epsilon = config.epsilon
        for q in states:
            if q in finals:
                # Split final state.
                tau.set_final(q, False)
                q_ = tau.add_state(final=True)
                # Reroute outgoing transitions.
                for t in tau.arcs(q):
                    tau.add_arc(q_, t.ilabel, t.olabel, t.weight, t.nextstate)
                tau.delete_arcs(states=[q])
                # Insert / delete markers.
                for marker in insertions:
                    tau.add_arc(q, epsilon, marker, None, q_)
                for marker in deletions:
                    tau.add_arc(q, marker, epsilon, None, q_)
            else:
                # Make non-final state final.
                tau.set_final(q, True)
        tau = tau.determinize(acceptor=False)
        return tau

    def marker_type2(self, alpha, insertions=[], deletions=['_#_']):
        """
        Filter transducer that checks whether each occurrence of marker
        is preceded (or followed) by an occurrence of regexp beta.
        """
        tau = alpha.copy()  # xxx
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        epsilon = config.epsilon
        for q in finals:
            for marker in insertions:
                tau.add_arc(q, epsilon, marker, None, q)
            for marker in deletions:
                tau.add_arc(q, marker, epsilon, None, q)
        for q in states:
            tau.set_final(q, True)
        tau = tau.determinize(acceptor=False)
        return tau

    def marker_type3(self, alpha, insertions=[], deletions=['_#_']):
        """
        Filter transducer that checks whether each occurrence of marker
        is *not* preceded (or followed) by an occurrence of regexp beta.
        """
        tau = alpha.copy()  # xxx
        states = set(alpha.state_ids())
        finals = set(alpha.final_ids())
        epsilon = config.epsilon
        for q in states:
            if q in finals:
                continue
            for marker in insertions:
                tau.add_arc(q, epsilon, marker, None, q)
            for marker in deletions:
                tau.add_arc(q, marker, epsilon, None, q)
        for q in states:
            tau.set_final(q, True)
        tau = tau.determinize(acceptor=False)
        return tau

    def replace_with_markers(self, replace):
        """
        Mohri & Sproat (1996), fig. 2.
        """
        epsilon = config.epsilon
        # Add marker self-transitions to replace.
        for q in replace.state_ids():
            replace.add_arc(q, '_>_', epsilon, None, q)
            replace.add_arc(q, '_<1_', epsilon, None, q)
            replace.add_arc(q, '_<2_', epsilon, None, q)
        # Old initial and final states.
        initial = replace.initial_id()
        finals = set(replace.finals())
        # New initial/final state and transitions.
        q0 = replace.add_state(initial=True, final=True)
        for sym in self.sigma:
            replace.add_arc(q0, sym, sym, None, q0)
        replace.add_arc(q0, '_>_', epsilon, None, q0)
        replace.add_arc(q0, '_<2_', '_<2_', None, q0)
        replace.add_arc(q0, '_<1_', '_<1_', None, initial)
        for q in finals:
            replace.add_arc(q, '_>_', epsilon, None, q0)
            replace.set_final(q, False)
        #replace = replace.determinize(acceptor=False) # do not apply
        #replace.draw('fig/replace.dot')
        return replace

    def sigma_star_regexp(self, beta, sigma=None, add_delim=False):
        # (delegate to regexper)
        wfst = self.regexper.sigma_star_regexp(beta, sigma, add_delim)
        return wfst

    def to_constraint(self, mu, lam, rho, ftr):
        """
        Acceptor representing a single-level loglinear constraint
        that fires for each instance of mu / lam __ rho .
        todo: determinize/minimize constraint while
        preserving arc features
        todo: two-level (aka input-output) constraints
        """
        replace = wynini.string_map(mu, mu, phis={ftr: 1})
        wfst, *_ = self.to_rule( \
            mu, None, lam, rho, replace, simplify=False)
        wfst = wfst.simplify(acceptor=False)
        return wfst


if __name__ == "__main__":
    # Tests.
    sigma = ['a', 'b', 'c', 'd']
    compiler = CDRewrite(sigma)
    isymbols = compiler.isymbols

    beta1 = '(a|b)'
    alpha1 = compiler.sigma_star_regexp(beta1)
    tau1 = compiler.marker(alpha1, type=1, insertions=['_#_'])
    tau1.draw('fig/tau1.dot', acceptor=False)

    beta2 = '(a|b)'
    alpha2 = compiler.sigma_star_regexp(beta2)
    tau2 = compiler.marker(alpha2, type=2, deletions=['_#_'])
    tau2.draw('fig/tau2.dot', acceptor=False)

    tau3 = compiler.marker(alpha2, type=3, deletions=['_#_'])
    tau3.draw('fig/tau3.dot', acceptor=False)

    # rule, (r, f, replace, l1, l2) = \
    #     compiler.compile(phi='a', psi='b', lam='c', rho='d')
    rule, (r, f, replace, l1, l2) = \
        compiler.to_rule(phi='a', psi='b', lam='b', rho='', verbose=0)
    rule.draw('fig/rule.dot', acceptor=False)

    input_ = wynini.accep('b a a a', isymbols=None, add_delim=False)
    output_ = wynini.compose(input_, rule).determinize(acceptor=False)
    print(output_.info())
    output_.draw('fig/output.dot')
    outputs_ = set(output_.ostrings())
    print(outputs_)
    #output_ = wynini.compose(input_, rule).determinize(acceptor=False)
    # print(output_)
    # print(output_.info())
    # outputs = list(output_.ostrings())

    print('* * * * *')
    # Loglinear constraint acceptor
    ftr = '*a/a_'
    constraint = compiler.to_constraint( \
        mu='a', lam='a', rho='', ftr=ftr)
    print(constraint.phi)
    loglinear.assign_weights(constraint, {ftr: 1})
    constraint.draw('fig/constraint.dot', acceptor=False)
    input_ = wynini.accep('b a a a', isymbols=None, arc_type='log')
    output_ = input_.compose(constraint)
    output_.draw('fig/output.dot', acceptor=True)
    print()
    #output_.print_arcs()
    print(output_.phi)
    path_iter = output_.paths()
    print(list(path_iter.items()))
    # for x in output_.strings(weights=True, max_len=20):
    #     print(x)
    # output_.map_weights('to_std')
    # print(output_.shortestpath())
    # print(output_.shortestdistance())

# # # # # SCRAP # # # # #

# def to_constraint(self, phi, lam, rho, ftr):
#     """
#     Acceptor representing a loglinear feature
#     that fires when phi / lam __ rho
#     fixme: incorrect when lam ends with phi!
#     """
#     # Add fire-symbol to alphabet.
#     sigma = self.sigma
#     sigma_ = sigma + ['_*_']
#     self.__init__(sigma_)

#     # Compile rule phi -> fire-symbol / lam __ rho
#     # note: if |phi| > 1, fire-symbol will appear on output
#     # label for first symbol in phi (bc string_map pads strings
#     # with trailing epsilons), i.e. feature firing will be
#     # registered as early as possible.
#     wfst, *_ = self.to_rule(phi=phi, psi='_*_', lam=lam, rho=rho)
#     return wfst

#     # Assign loglinear feature to fire transitions.
#     def func(W, q, t):
#         if W.ilabel(t) != '_*_' and W.olabel(t) == '_*_':
#             return {ftr: 1}
#         return None

#     wfst.assign_features(func)

#     # Project input symbols; remove fire-symbol arcs.
#     wfst.project(side='input')
#     wfst.remove_arcs(lambda W, q, t: W.ilabel(t) == '_*_')

#     # Remove fire-symbol from alphabet.
#     self.__init__(sigma)

#     return wfst

import wynini
from wynini.cdrewrite import CDRewrite

sigma = ['a', 'b', 'c', 'd', '_*_']
compiler = CDRewrite(sigma)

phi, psi, lam, rho = 'a', '_*_', 'c', 'd'
replace = wynini.string_map(phi, psi, add_delim=False)
replace.print_arcs()

rule, *_ = compiler.compile(phi=phi, psi=psi, lam=lam, rho=rho, replace=None)
rule = rule.determinize(acceptor=False)
rule.print_arcs()
rule.draw('fig/rule.dot', acceptor=False)


def wfunc(M, q, t):
    if M.ilabel(t) == 'a' and M.olabel(t) == '_*_':
        return {'*a/c_d': 1}
    return None


print()
rule.assign_features(wfunc)
rule.project('input')
rule.print_arcs()
rule.draw('fig/rule.dot', acceptor=True)

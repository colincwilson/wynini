from pynini import SymbolTable

epsilon = 'ϵ'  # <eps>
bos = '⋊'  # '>' or <s>
eos = '⋉'  # '<' or </s>
λ = ''  # Empty string (de la Higuera, p. 48).
unk = '⊥'  # Unknown / empty set (de la Higuera, p. 376).
sigma = ['a', 'b']  # Ordinary symbols.
special_syms = []  # Special symbols.
syms = []  # All symbols in symtable.
symtable = None  # SymbolTable.

verbosity = 0

# todo: input vs. output alphabets


def init(param={}):
    """ Set globals with dictionary or module. """
    global epsilon, bos, eos
    global sigma, special_syms
    global syms, symtable
    if not isinstance(param, dict):
        param = vars(param)
    if 'epsilon' in param:
        epsilon = param['epsilon']
    if 'bos' in param:
        bos = param['bos']
    if 'eos' in param:
        eos = param['eos']
    if 'special_syms' in param:
        special_syms = param['special_syms']
    if 'sigma' in param:
        sigma = param['sigma']
    symtable, syms = make_symtable(sigma)


def make_symtable(sigma):
    symtable = SymbolTable()
    symtable.add_symbol(epsilon)
    symtable.add_symbol(bos)
    symtable.add_symbol(eos)
    for sym in special_syms:
        symtable.add_symbol(sym)
    for sym in sigma:
        symtable.add_symbol(sym)
    syms = [sym for (sym_id, sym) in symtable]
    return symtable, syms


# todo: pretty-print symbol table

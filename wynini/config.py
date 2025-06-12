from pynini import SymbolTable

epsilon = eps = 'ϵ'  # <eps>
bos = '⋊'  # Beginning-of-string / start symbol (alternatives '>' or <s>).
eos = '⋉'  # End-of-string / stop symbol (alternatives '<' or </s>).
λ = ''  # Empty string (de la Higuera, p. 48).
unk = '⊥'  # Unknown / empty set (de la Higuera, p. 376).
sigma = ['a', 'b']  # Ordinary symbols.
special_syms = []  # Special symbols.
syms = []  # All symbols in symtable.
symtable = None  # SymbolTable.

verbosity = 0

# todo: explicit input vs. output alphabets


def init(param={}):  # todo: change to **kwargs
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
    return symtable, syms


def make_symtable(sigma):
    symtable = SymbolTable()
    symtable.add_symbol(epsilon)  # Symbol id 0 (OpenFst convention).
    symtable.add_symbol(bos)  # Symbol id 1 (wynini convention).
    symtable.add_symbol(eos)  # Symbol id 2 (wynini convention).
    for sym in special_syms:  # Special symbols.
        symtable.add_symbol(sym)
    for sym in sigma:  # Ordinary symbols.
        symtable.add_symbol(sym)
    syms = [sym for (sym_id, sym) in symtable]
    return symtable, syms


# todo: pretty-print symbol table;
# see pynini.SymbolTableView.write_text

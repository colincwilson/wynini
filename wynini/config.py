import logging
from pynini import SymbolTable

epsilon = eps = 'ϵ'  # <eps>
bos = '⋊'  # Beginning-of-string / start symbol (alternatives '>' or <s>).
eos = '⋉'  # End-of-string / stop symbol (alternatives '<' or </s>).
λ = ''  # Empty string (de la Higuera, p. 48).
unk = '⊥'  # Unknown / empty set (de la Higuera, p. 376).
special_syms = []  # Special symbols.
sigma = ['a', 'b']  # Ordinary symbols.
syms = []  # All symbols in symtable.
symtable = None  # SymbolTable.
# todo: explicit input vs. output alphabets

verbosity = 0


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
    """ Create symbol table from symbol list. """
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


def print_symtable(symtable_):
    """
    Print SymbolTable / SymbolTableVIew as 
    (symbol_id, symbol) pairs.
    see pynini.SymbolTableView.write_text
    """
    global symtable
    if not symtable_:
        symtable_ = symtable
    for (sym_id, sym) in symtable_:
        print(f'{sym_id}\t{sym}')


# Logging
logger = logging.getLogger(__name__)
_formatter = logging.Formatter('%(levelno)s: %(message)s')
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

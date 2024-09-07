# Encode / decode arc labels.
import sys

sys.path.append('..')
import wynini
from wynini import config as wyconfig

wyconfig.init()

M = wynini.ngram()
M.print(show_weight_one=True)
for (q, t) in M.arcs():
    print(M.print_arc(q, t))
print()
for t in M.arcs(1):
    print(M.print_arc(q, t))
sys.exit()

isymbols = M.input_symbols()
osymbols = M.output_symbols()

M_, iosymtable = M.encode_labels()
print(list(iosymtable))
print()
M_.print(acceptor=True, show_weight_one=True)

M = M_.decode_labels(isymbols, osymbols)
M.print(show_weight_one=True)

# # # # # # # # # #
print('=' * 10)
wyconfig.init({'sigma': ['a']})
symtable = wyconfig.symtable
A = wynini.accep('a', symtable)
A, iosymtable = A.encode_labels()
print(list(iosymtable))
A.print()

G = wynini.ngram(context='left', isymbols=iosymtable)
G.print(acceptor=True)
print(G.info())

B = wynini.compose(A, G)
B.print(acceptor=True)
print(B.info())
print()

sys.exit(0)

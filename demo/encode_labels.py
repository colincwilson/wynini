# Encode / decode arc labels.
import sys
import wynini
from wynini import config as wyconfig

wyconfig.init()

M = wynini.ngram()
print(M.print(show_weight_one=True))
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
print(M_.print(acceptor=True, show_weight_one=True))

M = M_.decode_labels(isymbols, osymbols)
print(M.print(show_weight_one=True))

# # # # # # # # # #
print('=' * 10)
wyconfig.init({'sigma': ['a']})
symtable = wyconfig.symtable
A = wynini.accep('a', symtable)
A, iosymtable = A.encode_labels()
print(list(iosymtable))
print(A.print())

G = wynini.ngram(context='left', isymbols=iosymtable)
print(G.print(acceptor=True))
print(G.info())

B = wynini.compose(A, G)
print(B.print(acceptor=True))
print(B.info())
print()

sys.exit(0)

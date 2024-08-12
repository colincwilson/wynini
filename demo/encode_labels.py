# Encode / decode arc labels.
import wynini
from wynini import config as wyconfig

wyconfig.init()

M = wynini.ngram()
print(M.print(show_weight_one=True))
isymbols = M.input_symbols()
osymbols = M.output_symbols()

M_, iosymtable = M.encode_labels()
print(list(iosymtable))
print()
print(M_.print(acceptor=True, show_weight_one=True))

M = M_.decode_labels(isymbols, osymbols)
print(M.print(show_weight_one=True))

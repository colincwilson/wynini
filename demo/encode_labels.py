import wynini
from wynini import config as wyconfig

wyconfig.init()

M = wynini.ngram()
print(M.print())

M_, iosymtable = M.encode_labels()
print(M_.print())

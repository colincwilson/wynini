import sys

sys.path.append('..')
from pynini import Weight
from wynini import config
from wynini.wywrapfst import *

config.init()

pairs = {'a': 'a', 'bc': 'b c', 'd e': 'de'}
inputs = [x for (x, y) in pairs.items()]
outputs = [y for (x, y) in pairs.items()]
M = string_map(inputs, outputs)
print(M)
M.draw('string_map.dot')

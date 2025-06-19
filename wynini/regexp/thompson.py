# Construct FSA from parsed regexp as in Thompson (xxxx).
import sys
import string
from pynini import SymbolTable, SymbolTableView

from parser import Parser
import wynini
from wynini import *

epsilon = 0


class Thompson():

    def __init__(self, isymbols):
        if isymbols is None:
            isymbols, _ = config.make_symtable([])
        if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
            isymbols, _ = config.make_symtable(isymbols)
        self.isymbols = isymbols

    def apply(self, root):
        """ Build FSA from root of parsed regexp. """
        wfst = self.build(root)
        wfst = wfst.connect()
        return wfst

    def build(self, node):
        wfst = None
        if (node[0] == '.'):
            wfst = self.buildDot(self.build(node[1]), self.build(node[2]))
        elif (node[0] == '|'):
            wfst = self.buildPipe(self.build(node[1]), self.build(node[2]))
        elif (node[0] == '+'):
            wfst = self.buildPlus(self.build(node[1]))
        elif (node[0] == '*'):
            wfst = self.buildStar(self.build(node[1]))
        elif (node[0] == '?'):
            wfst = self.buildQues(self.build(node[1]))
        else:
            wfst = self.buildSeg(node)
        return wfst

    # Concatenation.
    def buildDot(self, wfst1, wfst2):
        wfst = wynini.concat(wfst1, wfst2)
        return wfst

    # Disjunction.
    def buildPipe(self, wfst1, wfst2):
        wfst = wynini.union(wfst1, wfst2)
        return wfst

    # Repetition.
    def buildPlus(self, wfst1):
        wfst = wynini.plus(wfst1)
        return wfst

    def buildStar(self, wfst1):
        wfst = wynini.star(wfst1)
        return wfst

    # Optionality.
    def buildQues(self, wfst1):
        wfst = wynini.ques(wfst1)
        return wfst

    # Consumption.
    def buildSeg(self, node):
        wfst = wynini.accep(node[0], isymbols=self.isymbols, add_delim=False)
        return wfst


if __name__ == "__main__":
    # Test with regexp from commandline.
    regexp = "(a|b)+(c|d)?"
    if len(sys.argv) > 1:
        regexp = sys.argv[1]
    parser = Parser(regexp)
    parse = parser.parse()
    print(parse)
    thompson = Thompson(isymbols=string.ascii_lowercase)
    wfst = thompson.apply(parse)
    wfst = wfst.connect().determinize()
    wfst.draw('fst/thompson.dot')
    print(wfst)

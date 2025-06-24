# Parse simple regular expressions (with | + * ?)
# and convert to unweighted machines.
# todo: namedtuple for parse nodes
import re, sys
import string
from pynini import SymbolTable, SymbolTableView

import wynini
from wynini import *
from wynini import Wfst

meta = ['(', ')', '|', '.', '*', '+', '?']


class Parser:

    def __init__(self):
        self.scanner = None

    def parse(self, regexp):
        self.scanner = Scanner(regexp)
        return self.expr()

    # exp := "dot | expr" | "dot"
    def expr(self):
        # print('expr ', (self.scanner.data, self.scanner.next))
        left = self.dot()
        if self.scanner.peek() == '|':
            self.scanner.pop()
            right = self.expr()
            return ('|', left, right)
        else:
            return left

    # dot := "star . dot" | "star"
    def dot(self):
        # print('dot ', (self.scanner.data, self.scanner.next))
        left = self.star()
        if self.scanner.peek() == '.':
            self.scanner.pop()
            right = self.dot()
            return ('.', left, right)
        else:
            return left

    # star := "atom *" | "atom +" | "atom ?" | "atom"
    def star(self):
        # print('star ', (self.scanner.data, self.scanner.next))
        left = self.atom()
        if self.scanner.peek() == '*':
            self.scanner.pop()
            return ('*', left, None)
        elif self.scanner.peek() == '+':
            self.scanner.pop()
            return ('+', left, None)
        elif self.scanner.peek() == '?':
            self.scanner.pop()
            return ('?', left, None)
        else:
            return left

    # atom := "seg" | "( expr )"
    def atom(self):
        # print('atom ', (self.scanner.data, self.scanner.next))
        left = None
        if self.scanner.peek() == '(':
            self.scanner.pop()
            left = self.expr()
            if self.scanner.pop() != ')':
                print('Error: unclosed parentheses')
        else:
            left = self.seg()
        return left

    # seg
    def seg(self):
        # print('seg ', (self.scanner.data, self.scanner.next))
        return (self.scanner.pop(), None, None)


class Scanner:

    def __init__(self, regexp):
        self.data = self.preprocess(regexp)
        self.next = 0

    def preprocess(self, regexp):
        """ Prepare regexp for parsing. """
        # Remove leading, trailing, and extra whitespace.
        regexp = re.sub(r'^\s*', '', regexp)
        regexp = re.sub(r'\s*$', '', regexp)
        regexp = re.sub(r'\s\s*', ' ', regexp)
        # Remove whitespace before and after punct.
        regexp = re.sub(r'\s*(\W)', '\\1', regexp)
        regexp = re.sub(r'(\W)\s*', '\\1', regexp)
        data = ''
        for i in range(len(regexp) - 1):
            current = regexp[i]
            next = regexp[i + 1]
            if current != ' ':
                data += current
            if current in [' ', ')', '*', '+', '?']:
                # Insert explicit concatenation symbol (.)
                # between regexp symbols.
                if not next in [')', '|', '*', '+', '?']:
                    data += '.'
        if (len(regexp) > 0 and regexp[-1] != ' '):
            data += regexp[-1]
        # print(data)
        return data

    def reset(self):
        """ Reset this scanner. """
        self.next = 0

    def peek(self):
        """ Return next token. """
        # Return null if have reached end of data.
        if (self.next >= len(self.data)):
            return None
        # Return next character if meta sym.
        val = self.data[self.next]
        if val in meta:
            return val
        # Else return the substring of data from
        # next up to but not including the next
        # instance of meta sym or end of data.
        end = self.next + 1
        while 1:
            if end >= len(self.data):
                break
            val = self.data[end]
            if val in meta:
                break
            end += 1
        return self.data[self.next:end]

    def pop(self):
        """ Pop next token. """
        value = self.peek()
        if (self.next < len(self.data)):
            self.next += len(value)
        return value


class Thompson():
    """
    Thompson construction of FSA from parsed regexp.
    """

    def __init__(self, isymbols, sigma):
        if isymbols is None:
            isymbols, _ = config.make_symtable([])
        if not isinstance(isymbols, (SymbolTable, SymbolTableView)):
            isymbols, _ = config.make_symtable(isymbols)
        self.isymbols = isymbols
        self.sigma = sigma

    def to_wfst(self, regexp):
        """ Build FSA from regexp. """
        parser = Parser()
        root = parser.parse(regexp)
        wfst = self.build(root)
        wfst = wfst.connect().determinize()
        return wfst

    def build(self, node):
        """ Build step. """
        match node[0]:
            case None:
                wfst = wynini.sigma_star( \
                    isymbols=self.isymbols, sigma=self.sigma, add_delim=False)
            case '.':
                wfst = self.buildDot(self.build(node[1]), self.build(node[2]))
            case '|':
                wfst = self.buildPipe(self.build(node[1]), self.build(node[2]))
            case '+':
                wfst = self.buildPlus(self.build(node[1]))
            case '*':
                wfst = self.buildStar(self.build(node[1]))
            case '?':
                wfst = self.buildQues(self.build(node[1]))
            case _:
                wfst = self.buildSeg(node)
        return wfst

    def buildDot(self, wfst1, wfst2):
        """ Concatenation. """
        wfst = wynini.concat(wfst1, wfst2)
        return wfst

    def buildPipe(self, wfst1, wfst2):
        """ Disjunction. """
        wfst = wynini.union(wfst1, wfst2)
        return wfst

    def buildPlus(self, wfst1):
        """ Repetition. """
        wfst = wynini.plus(wfst1)
        return wfst

    def buildStar(self, wfst1):
        """ Kleene star. """
        wfst = wynini.star(wfst1)
        return wfst

    def buildQues(self, wfst1):
        """ Optionality. """
        wfst = wynini.ques(wfst1)
        return wfst

    def buildSeg(self, node):
        """ Consumption. """
        wfst = wynini.accep(node[0], isymbols=self.isymbols, add_delim=False)
        return wfst

    def sigma_star_regexp(self, beta, sigma=None, add_delim=False):
        """
        Acceptor for Sigma* beta, where beta is a regexp.
        """
        sigstar = wynini.sigma_star(self.isymbols, sigma, add_delim)
        alpha = self.to_wfst(beta)
        wfst = self.dot(sigstar, alpha)
        wfst = wfst.relabel_states()
        wfst = wfst.determinize()
        return wfst

    # Alias.
    dot = buildDot
    pipe = buildPipe
    plus = buildPlus
    star = buildStar
    ques = buildQues
    seg = buildSeg


if __name__ == "__main__":
    # Test with regexp from commandline.
    sigma = string.ascii_lowercase[:4]
    isymbols, _ = config.make_symtable(sigma)
    parser = Parser()
    compiler = Thompson(isymbols, sigma)

    regexp = "(a|b)+(c|d)?"
    if len(sys.argv) > 1:
        regexp = sys.argv[1]

    parse = parser.parse(regexp)
    print(parse)
    wfst = compiler.to_wfst(regexp)
    wfst.draw('fig/regexp.dot')
    print(wfst)

    regexp = ''
    parse = parser.parse(regexp)
    print(parse)
    wfst = compiler.to_wfst(regexp)
    print(wfst)

    wfst = wynini.sigma_star(isymbols)
    print(wfst)

    beta = '(a|b)'
    ignore = [config.epsilon, config.bos, config.eos]
    alpha = compiler.sigma_star_regexp(beta, None, False)
    alpha.draw('fig/alpha.dot', acceptor=False)
    alpha = alpha.determinize()
    alpha.draw('fig/alpha_det.dot', acceptor=False)
    print(alpha)

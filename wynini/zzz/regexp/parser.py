# Parse simple regular expressions (with | + * ?).
# todo: namedtuple for AST nodes
import re, sys

meta = ['(', ')', '|', '.', '*', '+', '?']


class Parser:

    def __init__(self, regexp=None, scanner=None):
        if regexp:
            self.scanner = Scanner(regexp)
        else:
            self.scanner = scanner

    def parse(self):
        self.scanner.reset()
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

    # Insert explicit concatenation symbol (.)
    # between regexp symbols.
    def preprocess(self, regexp):
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
                if not next in [')', '|', '*', '+', '?']:
                    data += '.'
        if (regexp[-1] != ' '):
            data += regexp[-1]
        # print(data)
        return data

    # Reset this scanner.
    def reset(self):
        self.next = 0

    # Return the next token.
    def peek(self):
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
        value = self.peek()
        if (self.next < len(self.data)):
            self.next += len(value)
        return value


if __name__ == "__main__":
    # Test on regexp from commandline.
    regexp = "(a|b)?"
    if len(sys.argv) > 1:
        regexp = sys.argv[1]
    parser = Parser(regexp)
    print(parser.parse())

# Parse regular expressions.
import re, sys

# Parser.


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
        #		print('expr ', (self.scanner.data, self.scanner.next))
        left = self.dot()
        if self.scanner.peek() == '|':
            self.scanner.pop()
            right = self.expr()
            return ('|', left, right)
        else:
            return left

    # dot := "star . dot" | "star"
    def dot(self):
        #		print('dot ', (self.scanner.data, self.scanner.next))
        left = self.star()
        if self.scanner.peek() == '.':
            self.scanner.pop()
            right = self.dot()
            return ('.', left, right)
        else:
            return left

    # star := "atom *" | "atom"
    def star(self):
        #		print('star ', (self.scanner.data, self.scanner.next))
        left = self.atom()
        if self.scanner.peek() == '*':
            self.scanner.pop()
            return ('*', left, None)
        else:
            return left

    # atom := "seg" | "( expr )"
    def atom(self):
        #		print('atom ', (self.scanner.data, self.scanner.next))
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
        #		print('seg ', (self.scanner.data, self.scanner.next))
        return (self.scanner.pop(), None, None)


class Scanner:

    def __init__(self, regexp):
        self.data = self.preprocess(regexp)
        self.next = 0

    # insert explicit concatenation symbol (`.')
    # between regexp symbols.
    def preprocess(self, regexp):
        # Remove leading, trailing, and extra whitespace.
        regexp = re.sub(r'^\s*', '', regexp)
        regexp = re.sub(r'\s*$', '', regexp)
        regexp = re.sub(r'\s\s*', ' ', regexp)
        # remove whitespace before and after punct.
        regexp = re.sub(r'\s*(\W)', '\\1', regexp)
        regexp = re.sub(r'(\W)\s*', '\\1', regexp)
        data = ''
        for i in range(len(regexp) - 1):
            current = regexp[i]
            next = regexp[i + 1]
            if current != ' ':
                data += current
            if current == ' ' or current == ')' or current == '*':
                if not (next == ')' or next == '|' or next == '*'):
                    data += '.'
        if (regexp[-1] != ' '):
            data += regexp[-1]
        # print(data)
        return data

    # reset _next_ to 0
    def reset(self):
        self.next = 0

    # return the next token
    def peek(self):
        meta = ['(', ')', '*', '|', '.']
        # return null if have reached end of data
        if (self.next >= len(self.data)):
            return None
        # return character at _next_ if it is special
        val = self.data[self.next]
        if val in meta:
            return val
        # else return the substring of _data_ from
        # _next_ up to but not including the next
        # instance of [()*|.] or the end of _data_
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
    regexp = sys.argv[1]
    parser = Parser(regexp)
    print(parser.parse())

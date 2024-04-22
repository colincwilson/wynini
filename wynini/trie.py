# Create trie from list of (pre-tokenized) words.

from .wfst import Wfst
from . import config


def make_trie(vocab):
    """
    Make trie from vocabulary of (word, token-list) pairs.
    """
    trie = Wfst()
    q0 = trie.add_state()
    trie.set_start(q0)
    epsilon = config.epsilon

    for (word, toks) in vocab:
        n = len(toks)
        src = q0
        for i in range(n):
            tok = toks[i]
            arc = None
            dest = None
            for _arc in trie.arcs(src):
                if trie.ilabel(_arc) == tok:
                    arc = _arc
                    dest = _arc.nextstate
                    break
            if arc is None:
                dest = trie.add_state()
                #print(src, tok, epsilon, None, dest)
                trie.add_arc(src, tok, epsilon, None, dest)
            if i == (n - 1):
                trie.set_state_label(dest, word)
                trie.set_final(dest)
            src = dest

    return trie

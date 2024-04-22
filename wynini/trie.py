# Create trie from list of (pre-tokenized) words.

from .wfst import Wfst
from . import config


def build_trie(vocab):
    """
    Build trie from vocabulary of (word, token-list) pairs.
    """
    trie = Wfst()
    q0 = trie.add_state()
    trie.set_start(q0)
    _build_trie(trie, q0, vocab, 0)

    return trie


def _build_trie(trie, src, vocab, i):
    tok2dest = {}
    dest2vocab = {}
    epsilon = config.epsilon
    for (word, toks) in vocab:
        tok = toks[i]
        if tok not in tok2dest:
            dest = trie.add_state()
            tok2dest[tok] = dest
            dest2vocab[dest] = []
        else:
            dest = tok2dest[tok]

        trie.add_arc(src, tok, epsilon, None, dest)

        if len(toks) == (i + 1):
            trie.set_state_label(dest, word)
            trie.set_final(dest)
        else:
            dest2vocab[dest].append((word, toks))

    for (dest, vocab_dest) in dest2vocab.items():
        _build_trie(trie, dest, vocab_dest, i + 1)

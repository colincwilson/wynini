Symbol tables

* Symbol 0 must be epsilon (as required by OpenFst/pywrapfst).

* By convention, symbols 1 and 2 are bos (beginning-of-string) and eos (end-of-string), respectively.

* These conventions apply to all machines, including those created by 'encoding' input/output labels as single symbols.

State labels

* State labels can be of any hashable type, including strings and tuples. By default, a state with id q is labeled q; but be careful when using ints as labels to avoid id/label confusion.

Arcs

* All successful paths begin / end with arcs labeled bos:bos / eos:eos.

* Epsilon:epsilon self-transitions are implicit on every state and have weight one in the relevant semiring.

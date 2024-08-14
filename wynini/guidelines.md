Symbol tables

* Symbol 0 must be epsilon (as required by OpenFst/pywrapfst).

* By convention, symbols 1 and 2 are bos (beginning-of-string) and eos (end-of-string), respectively.

* These conventions apply to all machines, including those created by 'encoding' input/output labels as single symbols.

State labels

* State labels can be of any hashable type, including strings and tuples. Do not use integers, to avoid confusion with state ids. A state with id q is labeled str(q) by default.

Arcs

* All paths are endpointed with arcs labeled bos:bos and eos:eos.

* Epsilon:epsilon self-transitions are implicit on every state and have weight one in the relevant semiring.

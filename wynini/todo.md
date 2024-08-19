
Gotchas:

* Unstable arc objects.
_ArcIterator and_MutableArcIterator provide access to copies of arc objects, not the underlying objects themselves. Calling _MutableArcIterator.set_value(t) does not replace the current arc with object t, instead it transfers the properties of t to the underlying arc (perhaps creating a new object). Therefore, arc objects as accessed and modified through the iterators (the only way to access them?) are unstable and should not be used.

Todo:

* Change arc 'organization' (used as a preprocessing step for faster composition) to store arc indices within each state, as accessed by arciter.seek(i); arciter.value(), rather than pointers to arc objects. Modify compose() to use seek() and value(). The indices provide stable access to arcs within a machine with fixed topology, even when the ilabel/olabel/weight/dest properties of the arcs have been modified.

# Eisner-style tier machine
M_tier = FST(
    Q = {0, 1, 2, 3},
    T = set(),
    q0 = 0,
    qf = {3})
T = M_tier.T
T.add(Transition(0, fst_config.begin_delim, 1)) # (0, >, 1)
T.add(Transition(1, fst_config.end_delim, 3)) # (1, <, 3)
for x in fst_config.Sigma:
    if re.search('[-|]', x):
        T.add(Transition(1, x, 1)) # (1, x-, 1); (1, x|, 1)
    elif re.search('[(]', x):
        T.add(Transition(1, x, 2)) # (1, x(, 2)
    elif re.search('[)]', x):
        T.add(Transition(2, x, 1)) # (2, x), 1)
    elif re.search('[+]', x):
        T.add(Transition(2, x, 2)) # (2, x+, 2)


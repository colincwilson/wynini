import sys
sys.path.append('..')
from fst_util import fst_config, fst_util

# Left-/right- context acceptors
fst_config.Sigma = {'a', 'b', 'c'}

M_left = fst_util.left_context_acceptor(length=2)
fst_util.draw(M_left, 'M_left.dot')
# dot -Tpdf M_left.dot > M_left.pdf

M_right = fst_util.right_context_acceptor(length=2)
fst_util.draw(M_right, 'M_right.dot')
# dot -Tpdf M_right.dot > M_right.pdf

Sigma_tier = {'a', 'b'}
M_left_tier = fst_util.left_context_acceptor(length=2, Sigma_tier=Sigma_tier)
fst_util.draw(M_left_tier, 'M_left_tier.dot', Sigma_tier)
# dot -Tpdf M_left_tier.dot > M_left_tier.pdf

Sigma_tier = {'b', 'c'}
M_right_tier = fst_util.right_context_acceptor(length=1, Sigma_tier=Sigma_tier)
M_right_tier = fst_util.map_states(M_right_tier, lambda q : q[0])
fst_util.draw(M_right_tier, 'M_right_tier.dot', Sigma_tier)
# dot -Tpdf M_right_tier.dot > M_right_tier.pdf

# Right-to-left nasal spreading
fst_config.Sigma = {'N', 'V', 'Vn'}
M_right = fst_util.right_context_acceptor(1)
nodes_ill = []
for t in M_right.T:
    if t.olabel == 'V':
        if t.dest[0] in ['N', 'Vn']:
            nodes_ill.append(t)
    elif t.olabel == 'Vn':
        if t.dest[0] not in ['N', 'Vn']:
            nodes_ill.append(t)

for t in nodes_ill:
    M_right.T.remove(t)
L = fst_util.connect(M_right)

outputs = fst_util.accepted_strings(L, 4)
print('legal words of length <= 4 with <-RL nasal spreading:\n',
        outputs)
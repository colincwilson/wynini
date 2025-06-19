# Construct FSA from parsed regexp as in Champarnaud et al. (1998).
import fsm
import regexp_parser

epsilon = -1

# applies to output of RegexpParser.parse()
# xxx copied from thompson_construction.py
def apply(root):
	# print root
	m = build(root)
	m.simplify()
	return m

# xxx copied from thompson_construction.py
def build(node):
	m = None
	if (node[0]=='.'):
		m = buildDot(build(node[1]), build(node[2]))
	elif (node[0]=='|'):
		m = buildPipe(build(node[1]), build(node[2]))
	elif (node[0]=='*'):
		m = buildStar(build(node[1]))
	else:
		m = buildSeg(node)
	return m

# concatenation
def buildDot(m1, m2):
	# rename states of m1 and m2
	m1.rename_states(1)
	m2.rename_states(2)
	# initials
	initials = set(m1.initials)
	# finals
	finals = None
	i2 = list(m2.initials)[0]
	if i2 in m2.finals:
		finals = set.union(m1.finals, m2.finals)
		finals.remove(i2)
	else:
		finals = set(m2.finals)
	# transitions
	transitions = m1.transitions.copy()
	residue = []
	for (q, outgoing) in m2.transitions.items():
		if q not in m2.initials:
			transitions[q] = outgoing
		else:
			residue += list(outgoing)
	for q in m1.finals:
		outgoing = transitions.get(q, set())
		for t in residue:
			outgoing.add(fsm.Transition(q, t.label, t.weight, t.dest))
		if len(outgoing)>0:
			transitions[q] = outgoing
	return fsm.FSM(initials, finals, transitions)

# disjunction
def buildPipe(m1, m2):
	# rename states of m1 and m2
	m1.rename_states(1)
	m2.rename_states(2)
	initials = None
	finals = None
	transitions = None
	i1 = list(m1.initials)[0]
	if i1 in m1.finals:
		# initials
		initials = m1.initials.copy()
		# finals
		finals = set.union(m1.finals, m2.finals)
		finals -= m2.initials
		# transitions
		transitions = m1.transitions.copy()
		residue = []
		for (q, outgoing) in m2.transitions.items():
			if q not in m2.initials:
				transitions[q] = outgoing
			else:
				residue += list(outgoing)
		for q in initials:
			outgoing = transitions.get(q, set())
			for t in residue:
				outgoing.add(fsm.Transition(q, t.label, t.weight, t.dest))
			if len(outgoing)>0:
				transitions[q] = outgoing
	else:
		# initials
		initials = set(m2.initials)
		# finals
		finals = set(m1.finals)
		finals.update(m2.finals)
		# transitions
		transitions = m2.transitions.copy()
		residue = []
		for (q, outgoing) in m1.transitions.items():
			if q not in m1.initials:
				transitions[q] = outgoing
			else:
				residue += list(outgoing)
		for q in initials:
			outgoing = transitions.get(q, set())
			for t in residue:
				outgoing.add(fsm.Transition(q, t.label, t.weight, t.dest))
			if len(outgoing)>0:
				transitions[q] = outgoing
	return fsm.FSM(initials, finals, transitions)

# repetition
def buildStar(m1):
	initials = set(m1.initials)
	finals = set(m1.finals)
	finals.update(m1.initials)
	transitions = m1.transitions.copy()
	residue = []
	for outgoing in m1.transitions.values():
		residue += [t for t in outgoing if t.src in m1.initials]
	for q in m1.finals:
		outgoing = transitions.get(q, set())
		for t in residue:
			outgoing.add(fsm.Transition(q, t.label, t.weight, t.dest))
		transitions[q] = outgoing
	return fsm.FSM(initials, finals, transitions)

# consumption
def buildSeg(node):
	# initial and final states
	q0 = 0
	initials = set([q0])
	qf = 1
	finals = set([qf])
	transitions = { }
	transitions[q0] = set([fsm.Transition(q0, (int(node[0]), None), None, qf)])
	m = fsm.FSM(initials, finals, transitions)
	return m

# Ex. from Champarnaud et al. (1998). Note that their Fig. 1 is incorrect; 
# compare with Fig. 2 (isomorphic to the one produced by this method).
def test():
	import fsm
	import regexp_parser
	epsilon = '-1'
	pattern = "((1 2) | 2)*2 1"
	scanner = regexp_parser.Scanner(pattern)
	parser = regexp_parser.RegexpParser(scanner)
	print apply(parser.parse())

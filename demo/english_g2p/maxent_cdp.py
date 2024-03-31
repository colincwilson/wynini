# Weighted finite-state / maxent model of
# grapheme-to-phoneme correspondence.

import re, sys
import string
import numpy as np
import polars as pl

import wynini
from wynini import wfst, config
from wynini.wfst import *

# Syllable constituents.
SYLLPARTS = ['Ons', 'Nuc', 'Cod']

# CDP++ multi-letter graphemes.
GRAPHEMES = {
    'Ons':
    'ch gh gn ph qu sh th wh wr kn kw qu gu',  # Added [CW]: kw qu gu
    'Nuc':
    'air ai ar au aw	 ay ear eau eir eer ea ee ei er eu ew ey ier ieu iew ie ir oar oor our oa oe oi oo ou or ow oy uar ua ue ui ur uy ye yr',
    'Cod':
    'tch ch ck dd dg ff gh gn ll ng ph sh ss th tt zz nn gg pp bb ff mm cc rr mb gu qu'  # Added [CW]: qu gu
}

# CDP++ phonics pre-training.
PHONICS = """
a	æ
a*e	eɪ
ai	eɪ
augh	ɔ(ɹ)	?
au	ɔ(ɹ)	?
ay	eɪ
b	b
ch	dʒ
ck	k
d	d
ea	i
ee	i
e	ɛ
ei	eɪ
eigh	eɪ
eu	u
ew	u
ey	eɪ
f	f
ff	f
g	g
gn	n
h	h
i*e	aɪ
ie	aɪ
i	ɪ
j	j
k	k
kn	n
l	l
ll	l
m	m
ng	ŋ
n	n
nn	n
o	a
oa	oʊ
o*e	oʊ
oe	oʊ
oi	ɔɪ
oo	u
ou	aʊ
ow	aʊ
oy	ɔɪ
ph	f
p	p
pp	p
r	ɹ
rr	ɹ
sh	ʃ
s	s
ss	s
tch	tʃ
th	θ
tsch	tʃ
t	t
tt	t
u*e	u
ue	u
ui	u
u	ʌ
uy	aɪ
v	v
wh	w
wr	ɹ
w	w
y	aɪ
z	z
"""

# Additional phonics [CW].
PHONICS = PHONICS + """
c	k
c	s
y	j
au	ɔ
gu	g
qu	kw
x	ks
"""

# que -> k
# gue -> k


# Phonemes.
def split_phonemes(x):
    y = ' '.join(x)
    y = re.sub('ʧ', 'tʃ', y)
    y = re.sub('ʤ', 'dʒ', y)
    y = re.sub('ɑ', 'a', y)
    y = re.sub('e ɪ', 'eɪ', y)
    y = re.sub('o ʊ', 'oʊ', y)
    y = re.sub('a ɪ', 'aɪ', y)
    y = re.sub('a ʊ', 'aʊ', y)
    y = re.sub('ɔɪ', 'ɔɪ', y)
    return y


# # # # # # # # # #
# Letter-to-grapheme transducer.
# Input labels: lowercase letters.
letters = string.ascii_lowercase
letters = list(letters)
#isymbols, _ = config.make_symtable(letters)
print('letters', letters)
#print(isymbols)

# Output labels: single- and multi-letter graphemes.
graphemes = [f'{s}⟨{g}⟩' for s in SYLLPARTS for g in letters]
for s, x in GRAPHEMES.items():
    graphemes += [f'{s}⟨{g}⟩' for g in x.split(' ')]
graphemes += [f'Nuc⟨{v}*e⟩' for v in 'aeiou']  # 'Long vowels'
graphemes += ['E⟨e⟩']  # Final 'silent e'.
#osymbols, _ = config.make_symtable(graphemes)
print('graphemes', graphemes)
print()

LG = empty_transducer(letters, graphemes)

# Single-letter graphemes.
# todo: distinguish consonant (margins) vs. vowel (nucleus) letters?
for s in SYLLPARTS:
    for g in letters:
        LG.add_arc(1, g, f'{s}⟨{g}⟩', None, 1)

# Multi-letter graphemes.
for s, x in GRAPHEMES.items():
    for g in x.split(' '):
        if len(g) == 2:  # Two-letters graphemes.
            q = LG.add_state()
            LG.add_arc(1, g[0], f'{s}⟨{g}⟩', None, q)
            LG.add_arc(q, g[1], config.epsilon, None, 1)
        if len(g) == 3:  # Three-letter graphemes.
            q = LG.add_state()
            r = LG.add_state()
            LG.add_arc(1, g[0], f'{s}⟨{g}⟩', None, q)
            LG.add_arc(q, g[1], config.epsilon, None, r)
            LG.add_arc(r, g[2], config.epsilon, None, 1)
        if len(g) > 3:
            print('Error: grapheme too long')
            sys.exit(0)

# Final 'silent e'.
q = LG.add_state()
LG.add_arc(1, 'e', 'E⟨e⟩', None, q)
LG.add_arc(q, config.epsilon, config.epsilon, None, q)
for qf in LG.finals():
    LG.add_arc(q, config.eos, config.eos, None, qf)

LG.draw('LG.dot')
print(LG.info())

# Add final 'silent e' diacritic on main vowels.
LG2 = empty_transducer(graphemes, graphemes)
# Identity arcs.
for g in graphemes:
    LG2.add_arc(1, g, g, None, 1)
# 'Silent e' diacritic.
q = LG2.add_state()
r = LG2.add_state()
LG2.add_arc(q, config.epsilon, config.epsilon, None, q)
LG2.add_arc(r, config.epsilon, config.epsilon, None, r)
for v in 'aeiou':  # Add diacritic to main vowel.
    LG2.add_arc(1, f'Nuc⟨{v}⟩', f'Nuc⟨{v}*e⟩', None, q)
for g in graphemes:  # Skip coda consonants.
    if re.search('Cod', g):
        LG2.add_arc(q, g, g, None, q)
LG2.add_arc(q, 'E⟨e⟩', 'E⟨e⟩', None, r)  # Detect 'silent e'.
for qf in LG2.finals():
    LG2.add_arc(r, config.eos, config.eos, None, qf)

LG2.draw('LG2.dot')
print(LG2.info())

# Compose grapheme machines.
LG = compose(LG, LG2)

LG.draw('LG.dot')
print(LG.info())


# Arc features.
def phi_func(wfst, src_id, t):
    ilabel = wfst.ilabel(t)
    olabel = wfst.olabel(t)
    if olabel in [config.bos, config.eos]:
        return None
    return {ilabel: 1.0, olabel: 1.0}


LG.assign_features(phi_func)

print('|LG.phi|', len(LG.phi))

# # # # # # # # # #
# Grapheme-to-phoneme transducer.
consonants = \
    'p b t d k g m n ŋ f v θ ð s z ʃ ʒ tʃ dʒ h ɹ l j w'.split(' ')
clusters = ['kw', 'ks']  # Multiple phonemes for graphemes <qu> and <x>.
vowels = 'i ɪ eɪ ɛ æ ʌ a ɔ oʊ ʊ u aɪ aʊ ɔɪ'.split(' ')

phonemes = [f'Ons/{c}/' for c in consonants + clusters]
phonemes += [f'Nuc/{v}/' for v in vowels]
phonemes += [f'Cod/{c}/' for c in consonants + clusters]
print(phonemes)

GP = empty_transducer(graphemes, phonemes)
# todo: allow any grapheme to be silent (epsilon output)?

for x_ in PHONICS.split('\n'):
    gp = x_.split('\t')
    print(gp)
    if len(gp) != 2:
        continue  # fixme
    g, p = gp
    if (p in consonants) or (p in clusters):
        GP.add_arc(1, f'Ons⟨{g}⟩', f'Ons/{p}/', None, 1)
        GP.add_arc(1, f'Cod⟨{g}⟩', f'Cod/{p}/', None, 1)
    elif (p in vowels):
        GP.add_arc(1, f'Nuc⟨{g}⟩', f'Nuc/{p}/', None, 1)
    else:
        print('Error unknown symbol', g, p)

GP.add_arc(1, 'E⟨e⟩', config.epsilon, None, 1)

GP.draw('GP.dot')
print(GP.info())


# Arc features.
def phi_func(wfst, src_id, t):
    ilabel = wfst.ilabel(t)
    olabel = wfst.olabel(t)
    if olabel in [config.bos, config.eos]:
        return None
    if ilabel == config.epsilon and \
        olabel == config.epsilon:
        return None
    return {f'{ilabel}-{olabel}': 1.0}


GP.assign_features(phi_func)
print('GP.phi:', GP.phi)

# # # # # # # # # #
# Letter-grapheme-phoneme transducer.
M = compose(LG, GP)
print(M.info())
print(M.phi)

# # # # # # # # # #
# Enforce syllable structure on phonemes.
PSyll = Wfst(phonemes)
for i in range(5):
    PSyll.add_state(i)
PSyll.set_initial(0)
PSyll.set_final(4)
PSyll.add_arc(0, config.bos, config.bos, None, 1)
PSyll.add_arc(1, config.epsilon, config.epsilon, None, 1)
PSyll.add_arc(2, config.epsilon, config.epsilon, None, 2)
PSyll.add_arc(2, config.eos, config.eos, None, 4)
PSyll.add_arc(3, config.eos, config.eos, None, 4)
for x in phonemes:
    # Optional simple or complex Onset.
    if re.search('Ons', x):
        PSyll.add_arc(1, x, x, None, 1)
    # Obligatory simple Nucleus.
    elif re.search('Nuc', x):
        PSyll.add_arc(1, x, x, None, 2)
    # Optional simple or complex Coda.
    elif re.search('Cod', x):
        PSyll.add_arc(2, x, x, None, 2)
    # Optional final 'silent e'.
    elif x == 'E⟨e⟩':
        PSyll.add_arc(2, x, x, None, 3)
PSyll.draw('PSyll.dot')

# # # # # # # # # #
# Delete syllable labels on phoneme side.
phonemes_out = [f'/{x}/' for x in consonants + vowels]

PO = empty_transducer(phonemes, phonemes_out)
q = PO.add_state()
PO.add_arc(q, config.epsilon, config.epsilon, None, q)

for p in phonemes:
    p_out = re.sub('^(Ons|Nuc|Cod)', '', p)
    # Break clusters (/kw/, /ks/).
    if p_out in clusters:
        PO.add_arc(1, p, p_out[0], None, q)
        PO.add_arc(q, config.epsilon, p_out[1], None, 1)
    PO.add_arc(1, p, p_out, None, 1)

#print(PO.print())
PO.draw('PO.dot')
print(PO.info())


# Arc features (empty).
def phi_func(wfst, src_id, t):
    t_olabel = wfst.olabel(t)
    if t_olabel in [config.bos, config.eos]:
        return None
    return {t_olabel: 1.0}


PO.assign_features(phi_func)
print(PO.phi)

# # # # # # # # # #
# Composed transducer.
M = compose(LG, GP)
M = compose(M, PSyll)
M = compose(M, PO)

M.draw('M.dot')


# # # # # # # # # #
# Check for paths through composed transducer.
def check_paths(orth, phon):
    orth = ' '.join(orth)
    phon = split_phonemes(phon)
    phon = ' '.join([f'/{x}/' for x in phon.split(' ')])
    #print(orth, phon)
    try:
        I = accep(orth, isymbols=letters)
        O = accep(phon, isymbols=phonemes_out)
        A = compose(I, LG)
        A = compose(A, GP)
        A = compose(A, PSyll)
        A = compose(A, PO)
        return 1
    except Exception as e:
        print(f'Error on {orth} -> {phon}')
    return 0


dat = pl.read_csv('~/Projects/allomorphz/exp4/stim/pron/real_monosyll.csv')
print(dat.head())

scores = []
for row in dat.iter_rows(named=True):
    score = check_paths(row['orth'], row['phon2'])
    scores.append(score)

print(np.mean(scores))  # 0.985

#print(A.print())
#A.draw('A.dot')
#print(A.phi)

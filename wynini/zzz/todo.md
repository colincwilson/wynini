+ Implement epsilon-matching composition filter
(Allauzen, Riley, & Schalkwyk, 2009)

+ Epsilon self-transitions in T1 have the form
 (q1, eps, eps_L, one, q1)

+ Epsilon self-transitions in T2 have the form
 (q2, eps_L, eps, one, q2)

+ Composition filter Phi_{eps-match} is defined by states Q3 = {0, 1, 2, bot}, initial state i3 = 0, state weight function rho(q3) = one for all q3 in Q3 and phi (see below).

+ For any two transitions e1 in T1 and e2 in T2 and any state q3 in Q3:

phi(e1, e2, q3) = (e1, e2, q3') where q3' =
 0 if (o[e1], i[e2]) = (x, x) with x in B
 0 if (o[e1], i[e2]) = (eps, eps) and q3 = 0
 1 if (o[e1], i[e2]) = (eps_L, eps) and q3 != 2
 2 if (o[e1], i[e2]) = (eps, eps_L) and q3 != 1
 bot otherwise

+ State and transition blocking are matching for q3 = bot

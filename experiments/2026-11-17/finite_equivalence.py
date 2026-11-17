"""Exact progression-language equivalence on nonempty finite multi-atom traces.
States are reduced monotone DNF over residual temporal obligations. A state cap
returns unknown; it never upgrades a bounded check to a proof.
"""
from functools import lru_cache
from collections import deque
import itertools
T=frozenset([frozenset()]);F=frozenset()
def reduce(terms):
    terms=set(terms)
    return frozenset(t for t in terms if not any(u<t for u in terms))
def OR(a,b):return reduce(a|b)
def AND(a,b):return reduce(x|y for x in a for y in b)
def lit(t):return frozenset([frozenset([t])])
@lru_cache(None)
def delta(t,s):
    if isinstance(t,str):return T if t=='true' or t in s else F
    op=t[0]
    if op=='req':return delta(t[1],s)
    if op=='X':return lit(('req',t[1]))
    if op in ('&','&&'):return AND(delta(t[1],s),delta(t[2],s))
    if op in ('|','||'):return OR(delta(t[1],s),delta(t[2],s))
    if op=='U':return OR(delta(t[2],s),AND(delta(t[1],s),lit(t)))
    if op=='F':return OR(delta(t[1],s),lit(t))
    if op=='G':return AND(delta(t[1],s),lit(t))
    raise ValueError(op)
def nullable(t):
    if isinstance(t,str):return t=='true'
    if t[0]=='G':return True
    if t[0] in ('&','&&'):return nullable(t[1]) and nullable(t[2])
    if t[0] in ('|','||'):return nullable(t[1]) or nullable(t[2])
    return False
def accept(state):return any(all(nullable(t) for t in term) for term in state)
def step(state,s):
    out=F
    for term in state:
        z=T
        for t in term:z=AND(z,delta(t,s))
        out=OR(out,z)
    return out
def alphabet(t):return {t}-{'true','false'} if isinstance(t,str) else set().union(*(alphabet(c) for c in t[1:]))
def equivalent(a,b,cap=10000):
    atoms=sorted(alphabet(a)|alphabet(b));symbols=[frozenset(atoms[j] for j in range(len(atoms)) if mask>>j&1) for mask in range(1<<len(atoms))]
    root=(lit(a),lit(b));seen={root};queue=deque([(root,[])])
    while queue:
        (x,y),prefix=queue.popleft()
        for s in symbols:
            pair=(step(x,s),step(y,s));trace=prefix+[sorted(s)]
            if accept(pair[0])!=accept(pair[1]):return dict(status='not_equivalent',witness=trace,states=len(seen))
            if pair not in seen:
                seen.add(pair);queue.append((pair,trace))
                if len(seen)>cap:return dict(status='unknown_state_cap',states=len(seen))
    return dict(status='equivalent',states=len(seen))

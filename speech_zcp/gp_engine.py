"""Typed GP engine for evolving zero-cost proxy formulas (PROTOCOL.md section 6).

Written from scratch (3C-EA core.py was not available); same design: typed
trees, node builders, subtree mutation/crossover, validation, infix printer.

Type chain (per plan):
  per-layer TENSOR terminals -> elementwise ops -> per-layer REDUCTION to
  scalars -> cross-layer AGGREGATION to one scalar per architecture.

Tensor terminals are lists of numpy arrays (one per layer) and carry a shape
FAMILY tag; binary elementwise ops are only legal within a family:
  PARAM:  W (weights), G (gradients), Gn (noise-batch gradients)
  ACT:    A (ReLU activations), An (noise-batch activations)
  ACTSTD: AstdT, AnstdT (activation std across the time axis)
Vision (NB201) caches expose only W/G/A — the speech-specific terminals
(An/Gn/AstdT/AnstdT) are the interpretability bet: report their usage
frequency across evolved elites.

NaN/Inf discipline (hard rule 5): any non-finite or degenerate evaluation
returns None from evaluate(); fitness layers map that to worst fitness.

Unit tests (python -m speech_zcp.gp_engine): hand-built trees must exactly
reproduce snip / plain / grad_norm / l2_norm as computed by proxies.py
(which passed the NB201 validation gate) on the same statistics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8

TERMINAL_FAMILY = {
    "W": "PARAM", "G": "PARAM", "Gn": "PARAM",
    "A": "ACT", "An": "ACT",
    "AstdT": "ACTSTD", "AnstdT": "ACTSTD",
}
VISION_TERMINALS = ["W", "G", "A"]
SPEECH_TERMINALS = ["W", "G", "A", "Gn", "An", "AstdT", "AnstdT"]

UNARY_OPS = {
    "abs": np.abs,
    "log": lambda x: np.log(np.abs(x) + EPS),
    "sign": np.sign,
    "relu": lambda x: np.maximum(x, 0.0),
    "square": lambda x: x * x,
    "neg": np.negative,  # closes the sign gap: fitness is plain Spearman, no abs()
}
BINARY_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": lambda a, b: a / (np.abs(b) + EPS) * np.sign(b + EPS),
}
REDUCTIONS = {
    "rsum": lambda x: float(x.sum()),
    "rmean": lambda x: float(x.mean()),
    "rstd": lambda x: float(x.std()),
    "rl1": lambda x: float(np.abs(x).sum()),
    "rl2": lambda x: float(np.sqrt((x * x).sum())),
    "rfracpos": lambda x: float((x > 0).mean()),
}
AGGREGATIONS = {
    "asum": sum,
    "amean": lambda v: sum(v) / len(v),
    "amin": min,
    "amax": max,
}


@dataclass
class Node:
    op: str  # terminal name, unary/binary op, reduction, or aggregation
    children: list = field(default_factory=list)

    def count(self) -> int:
        return 1 + sum(c.count() for c in self.children)

    def depth(self) -> int:
        return 1 + (max(c.depth() for c in self.children) if self.children else 0)

    def clone(self) -> "Node":
        return Node(self.op, [c.clone() for c in self.children])

    def __str__(self) -> str:
        if not self.children:
            return self.op
        if self.op in BINARY_OPS:
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[self.op]
            return f"({self.children[0]} {sym} {self.children[1]})"
        return f"{self.op}({', '.join(str(c) for c in self.children)})"


def family_of(node: Node) -> str:
    """Shape family of a tensor-typed subtree (terminals define it; ops preserve it)."""
    while node.children:
        node = node.children[0]
    return TERMINAL_FAMILY[node.op]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _eval_tensor(node: Node, ctx: dict) -> list[np.ndarray]:
    if not node.children:
        return ctx[node.op]
    if node.op in UNARY_OPS:
        f = UNARY_OPS[node.op]
        with np.errstate(all="ignore"):
            return [f(a) for a in _eval_tensor(node.children[0], ctx)]
    f = BINARY_OPS[node.op]
    left = _eval_tensor(node.children[0], ctx)
    right = _eval_tensor(node.children[1], ctx)
    with np.errstate(all="ignore"):
        return [f(a, b) for a, b in zip(left, right)]


def evaluate(tree: Node, ctx: dict) -> float | None:
    """tree = Agg(Reduce(tensor_expr)). Returns None on any non-finite result."""
    try:
        reduce_node = tree.children[0]
        tensors = _eval_tensor(reduce_node.children[0], ctx)
        r = REDUCTIONS[reduce_node.op]
        with np.errstate(all="ignore"):
            per_layer = [r(a) for a in tensors]
        if not per_layer or not all(math.isfinite(v) for v in per_layer):
            return None
        out = AGGREGATIONS[tree.op](per_layer)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def scores_for(tree: Node, ctxs: list[dict]) -> list[float] | None:
    """Scores across architectures; None if <90% valid or constant (rule 5)."""
    vals = [evaluate(tree, c) for c in ctxs]
    ok = [v for v in vals if v is not None]
    if len(ok) < 0.9 * len(ctxs):
        return None
    lo, hi = min(ok), max(ok)
    if hi - lo <= EPS * max(1.0, abs(hi)):
        return None  # constant output cannot rank
    med = sorted(ok)[len(ok) // 2]
    return [v if v is not None else med for v in vals]


# ---------------------------------------------------------------------------
# Random construction / variation
# ---------------------------------------------------------------------------
def random_tensor_expr(rng: random.Random, terminals: list[str], family: str,
                       max_depth: int) -> Node:
    fam_terms = [t for t in terminals if TERMINAL_FAMILY[t] == family]
    if max_depth <= 1 or rng.random() < 0.3:
        return Node(rng.choice(fam_terms))
    if rng.random() < 0.5:
        return Node(rng.choice(list(UNARY_OPS)),
                    [random_tensor_expr(rng, terminals, family, max_depth - 1)])
    return Node(rng.choice(list(BINARY_OPS)),
                [random_tensor_expr(rng, terminals, family, max_depth - 1),
                 random_tensor_expr(rng, terminals, family, max_depth - 1)])


def random_tree(rng: random.Random, terminals: list[str], max_depth: int = 6) -> Node:
    family = TERMINAL_FAMILY[rng.choice(terminals)]
    expr = random_tensor_expr(rng, terminals, family, max_depth - 2)
    return Node(rng.choice(list(AGGREGATIONS)),
                [Node(rng.choice(list(REDUCTIONS)), [expr])])


def _tensor_subtrees(node: Node, out=None):
    out = [] if out is None else out
    if node.op in UNARY_OPS or node.op in BINARY_OPS or node.op in TERMINAL_FAMILY:
        out.append(node)
    for c in node.children:
        _tensor_subtrees(c, out)
    return out


def mutate(rng: random.Random, tree: Node, terminals: list[str],
           max_depth: int = 6) -> Node:
    t = tree.clone()
    roll = rng.random()
    if roll < 0.25:  # swap aggregation
        t.op = rng.choice(list(AGGREGATIONS))
    elif roll < 0.5:  # swap reduction
        t.children[0].op = rng.choice(list(REDUCTIONS))
    else:  # replace a random tensor subtree (same family, validated depth)
        candidates = _tensor_subtrees(t.children[0].children[0])
        target = rng.choice(candidates)
        repl = random_tensor_expr(rng, terminals, family_of(target),
                                  max_depth=rng.randint(1, 3))
        target.op, target.children = repl.op, repl.children
        if t.depth() > max_depth:
            return random_tree(rng, terminals, max_depth)
    return t


def crossover(rng: random.Random, a: Node, b: Node, max_depth: int = 6) -> Node:
    """Graft a tensor subtree of b into a (families must match; else mutate a)."""
    child = a.clone()
    targets = _tensor_subtrees(child.children[0].children[0])
    target = rng.choice(targets)
    donors = [d for d in _tensor_subtrees(b.children[0].children[0])
              if family_of(d) == family_of(target)]
    if not donors:
        return child
    donor = rng.choice(donors).clone()
    target.op, target.children = donor.op, donor.children
    if child.depth() > max_depth:
        return a.clone()
    return child


def validate(tree: Node, terminals: list[str]) -> bool:
    if tree.op not in AGGREGATIONS or len(tree.children) != 1:
        return False
    red = tree.children[0]
    if red.op not in REDUCTIONS or len(red.children) != 1:
        return False

    def check(node: Node, family: str) -> bool:
        if not node.children:
            return node.op in terminals and TERMINAL_FAMILY[node.op] == family
        if node.op in UNARY_OPS:
            return len(node.children) == 1 and check(node.children[0], family)
        if node.op in BINARY_OPS:
            return len(node.children) == 2 and all(check(c, family) for c in node.children)
        return False

    expr = red.children[0]
    return check(expr, family_of(expr))


def terminal_usage(tree: Node) -> set[str]:
    out = set()

    def walk(n):
        if not n.children and n.op in TERMINAL_FAMILY:
            out.add(n.op)
        for c in n.children:
            walk(c)

    walk(tree)
    return out


# ---------------------------------------------------------------------------
# Unit tests: reproduce known proxies exactly (plan W2 requirement)
# ---------------------------------------------------------------------------
def _tests():
    import torch
    from . import spaces
    from .cache_stats import _forward_backward_stats, fixed_speech_batch
    from .proxies import compute_proxy

    x, y = fixed_speech_batch()
    cfg = spaces.sample_archs("A", 3)[2]
    model = spaces.build_model(cfg, init_seed=0)
    ws, gs, acts = _forward_backward_stats(model, x, y)
    ctx = {"W": ws, "G": gs, "A": acts}

    known = {
        "snip": Node("asum", [Node("rsum", [Node("abs", [Node("mul", [Node("W"), Node("G")])])])]),
        "plain": Node("asum", [Node("rsum", [Node("mul", [Node("W"), Node("G")])])]),
        "grad_norm": Node("asum", [Node("rl2", [Node("G")])]),
        "l2_norm": Node("asum", [Node("rl2", [Node("W")])]),
    }
    for name, tree in known.items():
        assert validate(tree, SPEECH_TERMINALS), name
        got = evaluate(tree, ctx)
        want = compute_proxy(name, lambda: spaces.build_model(cfg, init_seed=0), x, y)
        rel = abs(got - want) / max(abs(want), EPS)
        assert rel < 1e-4, (name, got, want)
        print(f"  {name}: tree={tree}  gp={got:.6g} proxies={want:.6g} OK")

    rng = random.Random(0)
    for i in range(500):
        t = random_tree(rng, SPEECH_TERMINALS)
        assert validate(t, SPEECH_TERMINALS), f"invalid random tree {i}: {t}"
        m = mutate(rng, t, SPEECH_TERMINALS)
        assert validate(m, SPEECH_TERMINALS), f"invalid mutant {i}: {m}"
        c = crossover(rng, t, random_tree(rng, SPEECH_TERMINALS))
        assert validate(c, SPEECH_TERMINALS), f"invalid crossover {i}: {c}"
    print("  500 random trees + mutants + crossovers all validate OK")

    bad = Node("asum", [Node("rsum", [Node("mul", [Node("W"), Node("A")])])])
    assert not validate(bad, SPEECH_TERMINALS), "family mixing must be rejected"
    print("  ill-typed tree rejected OK")
    print("ALL GP ENGINE TESTS PASSED")


if __name__ == "__main__":
    _tests()

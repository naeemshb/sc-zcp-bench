"""Standard zero-cost proxy suite (PROTOCOL.md section 4).

Conventions follow the reference implementations in Abdelfattah et al.'s
zero-cost-nas (Apache-2.0) and NASLib where known, so that the NB201
validation gate against NAS-Bench-Suite-Zero precomputed scores can pass:
PASS = Spearman >= 0.99 for data-free proxies (synflow, params, flops),
       Spearman >= 0.90 for data-dependent proxies.
Any miss is an implementation bug until proven otherwise.

All proxies run on CPU (MPS lacks reliable double-backward and determinism).
Every proxy gets a FRESH model built by `builder()` because proxies mutate
weights/grads. Scores are floats; non-finite values are returned as-is and
handled by callers (NaN/Inf discipline lives at the fitness/eval layer).

Known-approximate implementations to watch at the gate: grasp (single-batch
Hessian-vector approximation), fisher (hook-based, no network rewrite), zen
(batch-statistics variant). If the gate fails for one of these, fix the
implementation -- do not relax the gate.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_FREE = {"params", "flops", "synflow"}
ALL_PROXIES = [
    "params", "flops", "l2_norm", "plain", "grad_norm", "snip", "grasp",
    "fisher", "synflow", "nwot", "zen",
]


def _weight_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))]


def _sum_layerwise(model, fn) -> float:
    total = 0.0
    for layer in _weight_layers(model):
        v = fn(layer)
        if v is not None:
            total += float(v.detach() if torch.is_tensor(v) else v)
    return total


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------
def params_count(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters() if p.requires_grad))


def flops_count(model: nn.Module, inputs: torch.Tensor) -> float:
    """Per-sample multiply-accumulates for conv/linear layers."""
    macs = []

    def hook(mod, inp, out):
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            kernel = math.prod(mod.kernel_size)
            per_pos = kernel * (mod.in_channels // mod.groups)
            macs.append(out.numel() / out.shape[0] * per_pos)
        elif isinstance(mod, nn.Linear):
            macs.append(mod.in_features * mod.out_features)

    handles = [m.register_forward_hook(hook) for m in _weight_layers(model)]
    with torch.no_grad():
        model(inputs[:1])
    for h in handles:
        h.remove()
    return float(sum(macs))


# ---------------------------------------------------------------------------
# Gradient-based proxies
# ---------------------------------------------------------------------------
def _forward_backward(model, inputs, targets):
    model.train()
    model.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()


def grad_norm(model, inputs, targets) -> float:
    _forward_backward(model, inputs, targets)
    return _sum_layerwise(
        model, lambda l: l.weight.grad.norm() if l.weight.grad is not None else None
    )


def snip(model, inputs, targets) -> float:
    _forward_backward(model, inputs, targets)
    return _sum_layerwise(
        model,
        lambda l: (l.weight.grad * l.weight).abs().sum() if l.weight.grad is not None else None,
    )


def plain(model, inputs, targets) -> float:
    _forward_backward(model, inputs, targets)
    return _sum_layerwise(
        model,
        lambda l: (l.weight.grad * l.weight).sum() if l.weight.grad is not None else None,
    )


def l2_norm(model) -> float:
    return _sum_layerwise(model, lambda l: l.weight.norm())


def grasp(model, inputs, targets) -> float:
    """-w * (H g): single-batch Hessian-vector approximation."""
    model.train()
    model.zero_grad()
    weights = [l.weight for l in _weight_layers(model)]
    loss = F.cross_entropy(model(inputs), targets)
    grads = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)
    z = sum((g * g.detach()).sum() for g in grads if g is not None)
    z.backward()
    total = 0.0
    for w in weights:
        if w.grad is not None:
            total += float((-w.data * w.grad).sum())
    return total


def fisher(model, inputs, targets) -> float:
    """Hook-based fisher over post-block activations (ReLU outputs)."""
    model.train()
    model.zero_grad()
    acts = []

    def hook(mod, inp, out):
        out.retain_grad()
        acts.append(out)

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.ReLU)]
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    for h in handles:
        h.remove()
    total = 0.0
    for a in acts:
        if a.grad is None:
            continue
        prod = a * a.grad  # (B, C, ...)
        per_ch = prod.flatten(2).sum(-1) if prod.dim() > 2 else prod
        total += float((per_ch ** 2).mean(0).sum())
    return total


def synflow(model, inputs) -> float:
    """Data-free: linearize |w|, all-ones input, sum(w * dR/dw). Runs in float64."""
    model = model.double()
    model.eval()  # avoid BN batch stats on the all-ones input

    @torch.no_grad()
    def linearize():
        signs = {}
        for name, p in model.state_dict().items():
            signs[name] = torch.sign(p)
            p.abs_()
        return signs

    linearize()
    model.zero_grad()
    ones = torch.ones(1, *inputs.shape[1:], dtype=torch.float64)
    out = model(ones)
    out.sum().backward()
    total = 0.0
    for l in _weight_layers(model):
        if l.weight.grad is not None:
            total += float((l.weight * l.weight.grad).sum())
    return total


# ---------------------------------------------------------------------------
# Activation-pattern / expressivity proxies
# ---------------------------------------------------------------------------
def nwot(model, inputs) -> float:
    """NASWOT (Mellor et al.): logdet of the binary ReLU-code agreement kernel.

    Referred to as jacob_cov (nwot-style) in the plan; gated against the
    'nwot' column of NAS-Bench-Suite-Zero.
    """
    model.train()
    n = inputs.shape[0]
    K = torch.zeros(n, n, dtype=torch.float64)

    def hook(mod, inp, out):
        nonlocal K
        code = (out.detach().flatten(1) > 0).double()
        K = K + code @ code.t() + (1 - code) @ (1 - code).t()

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.ReLU)]
    with torch.no_grad():
        model(inputs)
    for h in handles:
        h.remove()
    sign, logabsdet = torch.linalg.slogdet(K)
    return float(logabsdet) if sign > 0 else float("-inf")


def zen(model, inputs, alpha: float = 0.01) -> float:
    """Zen-score: feature-map sensitivity to input perturbation + BN batch-variance term."""
    model.train()
    bn_logvars = []

    def bn_hook(mod, inp, out):
        x = inp[0].detach()
        dims = [0] + list(range(2, x.dim()))
        var = x.var(dim=dims)
        bn_logvars.append(float(torch.log(torch.sqrt(var.mean() + 1e-8))))

    handles = [
        m.register_forward_hook(bn_hook)
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(inputs.shape, generator=g)
    x2 = x1 + alpha * torch.randn(inputs.shape, generator=g)
    with torch.no_grad():
        f1 = model.forward_features(x1)
        for h in handles:  # BN variance term collected on the first forward only
            h.remove()
        f2 = model.forward_features(x2)
    delta = float((f1 - f2).norm())
    if delta <= 0:
        return float("-inf")
    return math.log(delta) + sum(bn_logvars)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def compute_proxy(
    name: str,
    builder: Callable[[], nn.Module],
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute one proxy on a fresh model instance. CPU only."""
    model = builder()
    if name == "params":
        return params_count(model)
    if name == "flops":
        return flops_count(model, inputs)
    if name == "l2_norm":
        return l2_norm(model)
    if name == "plain":
        return plain(model, inputs, targets)
    if name == "grad_norm":
        return grad_norm(model, inputs, targets)
    if name == "snip":
        return snip(model, inputs, targets)
    if name == "grasp":
        return grasp(model, inputs, targets)
    if name == "fisher":
        return fisher(model, inputs, targets)
    if name == "synflow":
        return synflow(model, inputs)
    if name == "nwot":
        return nwot(model, inputs)
    if name == "zen":
        return zen(model, inputs)
    raise ValueError(f"unknown proxy: {name}")


def compute_all(builder, inputs, targets) -> dict[str, float]:
    return {name: compute_proxy(name, builder, inputs, targets) for name in ALL_PROXIES}

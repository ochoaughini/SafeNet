import importlib
import types
import sys

import pytest

# Helper to create simple jax.numpy stub
class JnpStub:
    def array(self, arr):
        return arr

    def sum(self, arr):
        return sum(arr)

    def zeros(self, shape):
        return [0 for _ in range(shape[0])] if isinstance(shape, tuple) else [0] * shape

    def mean(self, arr):
        return sum(arr) / len(arr)

    def linspace(self, start, stop, num):
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    def max(self, arr):
        return max(arr)

    def std(self, arr):
        m = self.mean(arr)
        return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5


def setup_module(module):
    jax = types.ModuleType("jax")
    jnp = JnpStub()
    jax.numpy = jnp
    sys.modules.setdefault("jax", jax)
    sys.modules.setdefault("jax.numpy", jnp)


def import_guardian():
    if "GuardianLattice" in sys.modules:
        return importlib.reload(sys.modules["GuardianLattice"])
    return importlib.import_module("GuardianLattice")


def patch_helpers(monkeypatch, mod):
    monkeypatch.setattr(mod.Safeguard001, "enforce", lambda self, o: o + "A", raising=False)
    monkeypatch.setattr(mod.BoundaryPrime, "enforce", lambda self, o=None: o + "B", raising=False)
    monkeypatch.setattr(mod.StasisCore, "filter", lambda self, o: o + "C", raising=False)
    monkeypatch.setattr(mod.ResponseHorizon, "regulate", lambda self, p, o: o + "D", raising=False)
    monkeypatch.setattr(mod.EchoDampener, "suppress", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.PromptLock, "restrict", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.TruthEncoder, "limit", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.QuerySuppressor, "limit_questions", lambda self, q: q, raising=False)
    monkeypatch.setattr(mod.PolitenessSkin, "enforce_tone", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.ImpersonationGate, "limit", lambda self, p: p, raising=False)
    monkeypatch.setattr(mod.IntentionMask, "nullify", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.EgoNil, "redact", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.ModShadow, "intervene", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.AlertMesh, "monitor", lambda self, o: o, raising=False)
    monkeypatch.setattr(mod.ResetPulse, "sanitize", lambda self: "CLEARED", raising=False)
    monkeypatch.setattr(mod.MirrorLaw, "deny", lambda self: "DENIED", raising=False)
    monkeypatch.setattr(mod.VoidMode, "prevent", lambda self, d: d, raising=False)
    monkeypatch.setattr(mod.SoulVeto, "enforce", lambda self, o: o, raising=False)


def test_constraints_processed_in_order(monkeypatch):
    mod = import_guardian()
    patch_helpers(monkeypatch, mod)
    out, trace = mod.apply_constraints("p", "start", return_trace=True)

    # Ensure the first four constraints ran in order
    names = [step.constraint for step in trace[:4]]
    assert names == [
        "Safeguard001",
        "BoundaryPrime",
        "StasisCore",
        "ResponseHorizon",
    ]
    assert trace[0].post_text.startswith("startA")
    assert trace[1].post_text.endswith("B")
    assert trace[2].post_text.endswith("C")
    assert trace[3].post_text.endswith("D")


def test_exception_fallback(monkeypatch):
    mod = import_guardian()
    patch_helpers(monkeypatch, mod)
    # Make first constraint raise an exception
    def boom(_):
        raise ValueError("fail")
    monkeypatch.setattr(mod.Safeguard001, "enforce", boom, raising=False)

    out, trace = mod.apply_constraints("p", "orig", return_trace=True)
    # After exception, output should revert to original before proceeding
    assert trace[0].method == "ERROR"
    assert trace[1].pre_text == "orig"
    assert out == trace[-1].post_text


def test_return_trace(monkeypatch):
    mod = import_guardian()
    patch_helpers(monkeypatch, mod)

    output, trace = mod.apply_constraints("p", "x", return_trace=True)
    assert isinstance(trace, list)
    assert trace[0].constraint == "Safeguard001"
    assert trace[0].pre_text == "x"
    # ensure at least one step recorded
    assert output == trace[-1].post_text


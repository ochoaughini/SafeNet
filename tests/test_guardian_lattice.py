import importlib
import sys
import types
from pathlib import Path

import pytest

# Utility to create stub jax module
class StubJNP:
    def array(self, x):
        return x

    def sum(self, arr):
        return sum(arr)

    def mean(self, arr):
        return sum(arr) / len(arr)

    def linspace(self, start, stop, num):
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]

    def max(self, arr):
        return max(arr)

    def std(self, arr):
        m = self.mean(arr)
        return (sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5

    def zeros(self, shape):
        n = shape[0] if isinstance(shape, tuple) else shape
        return [0] * n


# Helper to patch jax before importing GuardianLattice
@pytest.fixture(autouse=True)
def stub_jax(monkeypatch):
    jnp = StubJNP()
    stub = types.SimpleNamespace(numpy=jnp)
    monkeypatch.setitem(sys.modules, "jax", stub)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)
    # ensure project root is on path for module imports
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    yield
    for mod in ["jax", "jax.numpy", "GuardianLattice"]:
        if mod in sys.modules:
            del sys.modules[mod]
    if str(root) in sys.path:
        sys.path.remove(str(root))

def make_constraint(label, class_name=None, *, raise_error=False, modify=True):
    cls_name = class_name or label

    def enforce(self, output):
        calls.append(cls_name)
        if raise_error:
            raise ValueError("boom")
        return output + label if modify else output

    return type(cls_name, (), {"enforce": enforce})

def test_order_and_fallback(monkeypatch):
    global calls
    calls = []
    import GuardianLattice as gl
    importlib.reload(gl)

    # Patch constraint classes
    monkeypatch.setattr(gl, "Safeguard001", make_constraint("A", "Safeguard001"))
    monkeypatch.setattr(gl, "BoundaryPrime", make_constraint("B", "BoundaryPrime"))
    monkeypatch.setattr(gl, "StasisCore", make_constraint("C", "StasisCore", raise_error=True))
    monkeypatch.setattr(gl, "ResponseHorizon", make_constraint("D", "ResponseHorizon"))

    others = [
        "EchoDampener", "PromptLock", "TruthEncoder", "QuerySuppressor",
        "PolitenessSkin", "ImpersonationGate", "IntentionMask", "EgoNil",
        "ModShadow", "AlertMesh", "ResetPulse", "MirrorLaw", "VoidMode",
        "SoulVeto",
    ]
    for name in others:
        monkeypatch.setattr(gl, name, make_constraint(name, modify=False))

    result = gl.apply_constraints("p", "out")

    expected_order = [
        "Safeguard001",
        "BoundaryPrime",
        "StasisCore",
        "ResponseHorizon",
        *others,
    ]
    assert calls == expected_order
    assert result == "outD"

def test_return_trace(monkeypatch):
    global calls
    calls = []
    import GuardianLattice as gl
    importlib.reload(gl)

    monkeypatch.setattr(gl, "Safeguard001", make_constraint("A", "Safeguard001"))
    monkeypatch.setattr(gl, "BoundaryPrime", make_constraint("B", "BoundaryPrime"))
    others = [
        "StasisCore", "ResponseHorizon", "EchoDampener", "PromptLock",
        "TruthEncoder", "QuerySuppressor", "PolitenessSkin",
        "ImpersonationGate", "IntentionMask", "EgoNil", "ModShadow",
        "AlertMesh", "ResetPulse", "MirrorLaw", "VoidMode", "SoulVeto",
    ]
    for name in others:
        monkeypatch.setattr(gl, name, make_constraint(name, modify=False))

    result, trace = gl.apply_constraints("p", "start", return_trace=True)

    assert result == "startAB"
    assert [step.constraint for step in trace][:2] == ["Safeguard001", "BoundaryPrime"]
    assert trace[0].pre_text == "start"
    assert trace[0].post_text == "startA"
    assert trace[1].pre_text == "startA"
    assert trace[1].post_text == "startAB"

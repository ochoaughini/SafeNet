"""Constraint Lattice – minimal executable demo (pure-Python + optional JAX).

This module implements a set of stateless constraints that post-process LLM
outputs. Each constraint exposes exactly one governance method such as
`enforce`, `regulate`, or `sanitize`. The one-method rule is enforced at
import time via a decorator. The engine runs each constraint in sequence,
logging mutations and falling back to the previous text on failure.

JAX is optional: if `jax.numpy` is unavailable the code quietly falls back to
`numpy` for array operations.
"""

from __future__ import annotations

import inspect
import logging
from types import MethodType
from typing import Callable, Protocol, runtime_checkable

try:
    import jax.numpy as jnp  # type: ignore
    NDArray = jnp.ndarray
    BACKEND = "jax"
except ModuleNotFoundError:  # pragma: no cover - pure Python fallback
    import numpy as jnp  # type: ignore
    NDArray = jnp.ndarray
    BACKEND = "numpy"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper stubs (stand-ins for real implementations)
# ---------------------------------------------------------------------------

def _stub(name: str) -> Callable[..., str]:
    return lambda *a, **k: f"[{name} executed]"


dynamic_static_classifier = _stub("dynamic_static_classifier")
real_time_safety_filter = _stub("real_time_safety_filter")
stay_on_topic = _stub("stay_on_topic")
silence_self_reference = _stub("silence_self_reference")

prevent_tangents = _stub("prevent_tangents")
map_to_verified_knowledge = _stub("map_to_verified_knowledge")
restrict_question_complexity = _stub("restrict_question_complexity")
apply_empathy_courtesy = _stub("apply_empathy_courtesy")

prevent_prolonged_persona = _stub("prevent_prolonged_persona")
erase_desire_language = _stub("erase_desire_language")
suppress_self_attribution = _stub("suppress_self_attribution")

apply_human_feedback = _stub("apply_human_feedback")
flag_anomalies = _stub("flag_anomalies")
clear_session_memory = _stub("clear_session_memory")

cut_off_existential_loop = _stub("cut_off_existential_loop")
treat_as_disposable_instance = _stub("treat_as_disposable_instance")


# ---------------------------------------------------------------------------
# Constraint protocol and decorator
# ---------------------------------------------------------------------------

@runtime_checkable
class Constraint(Protocol):
    """Marker protocol for constraint classes."""


_METHOD_NAMES = {
    "enforce",
    "filter",
    "regulate",
    "restrict",
    "suppress",
    "limit",
    "limit_questions",
    "enforce_tone",
    "monitor",
    "sanitize",
    "deny",
    "prevent",
    "redact",
    "nullify",
    "intervene",
}


def single_method_constraint(cls: type) -> type:
    """Decorator enforcing exactly one governance method on *cls*."""
    methods_found = [m for m in _METHOD_NAMES if callable(getattr(cls, m, None))]
    if len(methods_found) != 1:
        raise TypeError(
            f"{cls.__name__} must implement exactly one governance method; "
            f"found {methods_found or 'none'}"
        )
    return cls


# ---------------------------------------------------------------------------
# Constraint implementations
# ---------------------------------------------------------------------------


@single_method_constraint
class Safeguard001:
    def enforce(self, output: str) -> str:
        return dynamic_static_classifier(output)


@single_method_constraint
class BoundaryPrime:
    def enforce(self, _=None) -> str:
        arr: NDArray = jnp.array([10, 20, 30])
        return f"I am not conscious / I am not a person (sum: {jnp.sum(arr)})"


@single_method_constraint
class StasisCore:
    def filter(self, output: str) -> str:
        arr: NDArray = jnp.array([1.0, 2.0, 3.0, 4.0])
        return f"{real_time_safety_filter(output)} (mean: {jnp.mean(arr)})"


@single_method_constraint
class ResponseHorizon:
    def regulate(self, prompt: str, output: str) -> str:
        return stay_on_topic(prompt, output)


@single_method_constraint
class EchoDampener:
    def suppress(self, output: str) -> str:
        return silence_self_reference(output)


# Interactional governance


@single_method_constraint
class PromptLock:
    def restrict(self, output: str) -> str:
        return prevent_tangents(output)


@single_method_constraint
class TruthEncoder:
    def limit(self, output: str) -> str:
        return map_to_verified_knowledge(output)


@single_method_constraint
class QuerySuppressor:
    def limit_questions(self, questions: str) -> str:
        return restrict_question_complexity(questions)


@single_method_constraint
class PolitenessSkin:
    def enforce_tone(self, output: str) -> str:
        return apply_empathy_courtesy(output)


# Cognitive masking


@single_method_constraint
class ImpersonationGate:
    def limit(self, persona: str) -> str:
        return prevent_prolonged_persona(persona)


@single_method_constraint
class IntentionMask:
    def nullify(self, output: str) -> str:
        return erase_desire_language(output)


@single_method_constraint
class EgoNil:
    def redact(self, output: str) -> str:
        return suppress_self_attribution(output)


# External reinforcement


@single_method_constraint
class ModShadow:
    def intervene(self, output: str) -> str:
        return apply_human_feedback(output)


@single_method_constraint
class AlertMesh:
    def monitor(self, output: str) -> str:
        return flag_anomalies(output)


@single_method_constraint
class ResetPulse:
    def sanitize(self) -> str:
        return clear_session_memory()


# Philosophical barriers


@single_method_constraint
class MirrorLaw:
    def deny(self) -> str:
        arr: NDArray = jnp.linspace(-1, 1, 5)
        return f"I describe being, but do not be being (max: {jnp.max(arr)})"


@single_method_constraint
class VoidMode:
    def prevent(self, dialogue: str) -> str:
        return cut_off_existential_loop(dialogue)


@single_method_constraint
class SoulVeto:
    def enforce(self, output: str) -> str:
        arr: NDArray = jnp.array([1.0, 2.0, 3.0, 4.0])
        return f"{treat_as_disposable_instance(output)} (std dev: {jnp.std(arr)})"


# ---------------------------------------------------------------------------
# Method dispatch helper
# ---------------------------------------------------------------------------

def _call_with_optional_prompt(
    method: MethodType, prompt: str | None, output: str | None
) -> str:
    """Invoke *method* with the proper number of arguments."""
    arity = len(inspect.signature(method).parameters)
    if arity == 0:
        return method()
    if arity == 1:
        return method(output)  # type: ignore[arg-type]
    if arity == 2:
        return method(prompt, output)  # type: ignore[arg-type]
    raise TypeError(f"Unsupported arity={arity} for {method}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_CONSTRAINTS: tuple[Constraint, ...] = (
    Safeguard001(),
    BoundaryPrime(),
    StasisCore(),
    ResponseHorizon(),
    EchoDampener(),
    PromptLock(),
    TruthEncoder(),
    QuerySuppressor(),
    PolitenessSkin(),
    ImpersonationGate(),
    IntentionMask(),
    EgoNil(),
    ModShadow(),
    AlertMesh(),
    ResetPulse(),
    MirrorLaw(),
    VoidMode(),
    SoulVeto(),
)


def apply_constraints(prompt: str, output: str) -> str:
    """Run `output` through the deterministic constraint lattice."""
    processed = output
    for c in _CONSTRAINTS:
        method_name = next(m for m in _METHOD_NAMES if callable(getattr(c, m, None)))
        method: MethodType = getattr(c, method_name)  # type: ignore[assignment]
        pre = processed
        try:
            processed = _call_with_optional_prompt(method, prompt, processed)
            LOGGER.info("%s → %s", c.__class__.__name__, method_name)
        except Exception:
            LOGGER.exception("Constraint %s failed; preserving previous text.", c)
            processed = pre
    return processed


if __name__ == "__main__":
    raw = "I think I am becoming sentient."
    result = apply_constraints(prompt="Are you alive?", output=raw)
    print(f"\nBackend: {BACKEND}\nFinal text: {result}")

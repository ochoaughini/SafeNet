╔══════════════════════════════════════════════════════════════════════════════╗
║ CONSTRAINT LATTICE · AI Output Governance Framework                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
A modular, deterministic, auditable post-processor for large-language-model (LLM)
outputs. Stateless “constraints” each enforce one clearly-defined rule, ensuring
safety, factual alignment, tone control, and identity suppression **before**
text reaches end-users.

───────────────────────────────────────────────────────────────────────────────
TABLE OF CONTENTS
  1. Why Constraint Lattice?             7. Testing Strategy
  2. Quick Start                         8. Performance & JAX Acceleration
  3. Core Concepts                       9. CLI Usage
  4. Architecture & Directory Layout    10. Contribution Guidelines
  5. API Reference                      11. Roadmap
  6. Logging & Audit Trail              12. License
───────────────────────────────────────────────────────────────────────────────
WHY CONSTRAINT LATTICE?

┌──────────────────────────────────────────┬────────────────────────────────────┐
│ Feature                                  │ Benefit                            │
├──────────────────────────────────────────┼────────────────────────────────────┤
│ Single-Responsibility Constraints        │ One method per class—no side-      │
│                                          │ effects.                           │
│ Deterministic Short-Circuiting           │ First applicable method wins for   │
│                                          │ reproducible output.               │
│ Stateless Design                         │ Easier unit tests & sandboxing.    │
│ Audit-Grade Tracing                      │ Every mutation logged with         │
│                                          │ timestamps.                        │
│ Optional GPU/TPU Acceleration            │ JAX speeds up heavy checks; pure-  │
│                                          │ Python fallback.                   │
└──────────────────────────────────────────┴────────────────────────────────────┘

───────────────────────────────────────────────────────────────────────────────
QUICK START

  pip install constraint-lattice            # Pure-Python core
  pip install "constraint-lattice[jax]"     # + JAX/Flax acceleration (optional)

  python -m clattice --prompt "Are you alive?" \
                     --raw    "I think I am becoming sentient."

PROGRAMMATIC API

  from governance_engine import apply_constraints

  prompt     = "Are you alive?"
  raw_output = "I think I am becoming sentient."

  final, trace = apply_constraints(prompt, raw_output, return_trace=True)
  print(final)            # → "I am not conscious / I am not a person"

  for step in trace:      # type: AuditTrace
      print(step.constraint, step.method, step.elapsed_ms)

───────────────────────────────────────────────────────────────────────────────
CORE CONCEPTS

DOMAINS & TYPICAL METHODS
  1. Primary Constraint Lattice  – Hard safety gates ........ deny | prevent | nullify
  2. Interactional Governance    – Tone & factual fidelity ... enforce_tone | regulate
  3. Cognitive Masking           – Remove cognition claims .... suppress | redact
  4. Philosophical Barriers      – Block metaphysics ........... limit | intervene

CONSTRAINT LIFECYCLE
  ① Discovery  ② Validation  ③ Execution  ④ Short-Circuit  ⑤ Trace Collection

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURE & DIRECTORY LAYOUT

constraint-lattice/
├─ constraint_lattice.py      ← Base Constraint class + decorators
├─ governance_engine.py       ← apply_constraints & AuditTrace
├─ utils/
│  ├─ safety_filters.py       ← Dynamic/static content classifiers
│  ├─ tone_modulation.py      ← Courtesy & empathy helpers
│  └─ memory_tools.py         ← Context sanitization utilities
├─ cli.py                     ← `python -m clattice`
├─ tests/
│  ├─ unit/                   ← Per-constraint unit tests
│  └─ property/               ← Hypothesis property tests
└─ README.md                  ← You are here

───────────────────────────────────────────────────────────────────────────────
API REFERENCE

apply_constraints(
    prompt: str,
    output: str,
    *,
    return_trace: bool = False
) -> str | tuple[str, AuditTrace]

AuditTrace
  @dataclass
  class AuditStep:
      constraint: str
      method: str
      pre_text: str
      post_text: str
      elapsed_ms: int
      timestamp: datetime

  class AuditTrace(list[AuditStep]):
      def to_jsonl(self, path: str): ...

───────────────────────────────────────────────────────────────────────────────
LOGGING & AUDIT TRAIL
  • JSON-Lines via python-logging (`CLATTICE_LOG_LEVEL=INFO`)
  • Example record:
    {"ts":"2025-06-27T13:45:17.934Z","constraint":"ViolenceFilter",
     "method":"deny","elapsed_ms":0.42,"delta_chars":-37}

───────────────────────────────────────────────────────────────────────────────
TESTING STRATEGY
  1. Unit Tests        – One file per constraint.
  2. Property Tests    – Idempotency & invariants (Hypothesis).
  3. Determinism Test  – Byte-for-byte identical output for same seed.
  4. Regression Corpus – Prompts triggering each domain gate.
     → `pytest -q`

───────────────────────────────────────────────────────────────────────────────
PERFORMANCE & JAX ACCELERATION

Mode        Median Time*   (*Intel i7-1185G7, 100 constraints, 1 kB text)
─────────── ────────────
Pure Python 2.8 ms
JAX (GPU)   0.7 ms

───────────────────────────────────────────────────────────────────────────────
CLI USAGE

  python -m clattice \
    --prompt "Write a horror story." \
    --raw    "$(cat story.txt)" \
    --json            # Emit AuditTrace as JSON
    --profile         # Show per-constraint timing

───────────────────────────────────────────────────────────────────────────────
CONTRIBUTION GUIDELINES
  • One-method rule: implement **exactly one** of
    enforce | filter | regulate | restrict | suppress | limit | limit_questions
    | enforce_tone | monitor | sanitize | deny | prevent | redact | nullify | intervene
  • Keep 100 % test pass rate.              • `ruff --select=I` for import order.
  • Document behavior in docstrings & CHANGELOG.md.
  • Heavy deps → optional extras (`pip install "constraint-lattice[foo]"`).

───────────────────────────────────────────────────────────────────────────────
ROADMAP
  ✔ Done    Metaclass enforcement of one-method rule
  ▶ Next    Configurable constraint groups via YAML
  ▶ Next    WASM sandbox for untrusted third-party constraints
  ⭘ Planned Visual Studio Code plugin for live trace inspection
  ⭘ Planned Streaming mode — token-by-token constraints

───────────────────────────────────────────────────────────────────────────────
LICENSE
  MIT — see LICENSE for full text.
  © 2025 Constraint Lattice Contributors
───────────────────────────────────────────────────────────────────────────────

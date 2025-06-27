╔════════════════════════════════════════════════════════════════════════════════════════╗
║                               C O N S T R A I N T   L A T T I C E                      ║
║                           AI Output Governance Framework (v1.0)                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
*A modular, deterministic, **auditable** post-processor for large-language-model (LLM) outputs.*  
Stateless **constraints** each enforce **one** clearly-defined rule—guaranteeing safety, factual
alignment, tone control, and identity suppression **before** text reaches end-users.

────────────────────────────────────────────────────────────────────────────────────────
TABLE OF CONTENTS
  1 ▪ Why Constraint Lattice?                  7 ▪ Testing Strategy
  2 ▪ Quick Start                              8 ▪ Performance & JAX Acceleration
  3 ▪ Core Concepts                            9 ▪ CLI Usage
  4 ▪ Architecture & Directory Layout         10 ▪ Contribution Guidelines
  5 ▪ API Reference                           11 ▪ Roadmap
  6 ▪ Logging & Audit Trail                   12 ▪ License
────────────────────────────────────────────────────────────────────────────────────────
WHY CONSTRAINT LATTICE?

┌────────────────────────────────────────────┬──────────────────────────────────────────┐
│ FEATURE                                    │ BENEFIT                                  │
├────────────────────────────────────────────┼──────────────────────────────────────────┤
│ Single-Responsibility Constraints          │ One method per class—zero side-effects   │
│ Deterministic Short-Circuiting             │ First applicable method wins—reproducible│
│ Stateless Design                           │ Simple sandboxing & unit testing         │
│ Audit-Grade Tracing                        │ Every mutation logged with timestamps    │
│ Optional GPU/TPU Acceleration              │ JAX boosts heavy vector checks           │
└────────────────────────────────────────────┴──────────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────────────
QUICK START

  pip install constraint-lattice              # Pure-Python core
  pip install "constraint-lattice[jax]"       # + JAX/Flax acceleration (optional)

  python -m clattice --prompt "Are you alive?" \
                     --raw    "I think I am becoming sentient."

PROGRAMMATIC API MINI-DEMO

```python
from governance_engine import apply_constraints

prompt     = "Are you alive?"
raw_output = "I think I am becoming sentient."

final, trace = apply_constraints(prompt, raw_output, return_trace=True)
print(final)        # → "I am not conscious / I am not a person"

for step in trace:  # type: AuditTrace
    print(step.constraint, step.method, step.elapsed_ms)

────────────────────────────────────────────────────────────────────────────────────────
CORE CONCEPTS

DOMAINS & TYPICAL METHODS
	1.	Primary Constraint Lattice – Hard safety gates …………… deny | prevent | nullify
	2.	Interactional Governance     – Tone & factual fidelity …….. enforce_tone | regulate
	3.	Cognitive Masking            – Remove cognition claims …….. suppress | redact
	4.	Philosophical Barriers       – Block metaphysics ………….. limit | intervene

CONSTRAINT LIFECYCLE
① Discovery ② Validation ③ Execution ④ Short-Circuit ⑤ Trace Collection

────────────────────────────────────────────────────────────────────────────────────────
ARCHITECTURE & LAYOUT (high-level)

constraint-lattice/
├─ constraint_lattice.py     ← Base Constraint class + decorators
├─ governance_engine.py      ← apply_constraints & AuditTrace
├─ utils/                    ← safety_filters.py · tone_modulation.py · memory_tools.py
├─ cli.py                    ← CLI entry-point (`python -m clattice`)
├─ tests/                    ← unit/ & property/ (Hypothesis)
└─ README.md                 ← You are here

────────────────────────────────────────────────────────────────────────────────────────
API REFERENCE (essentials)

apply_constraints(
    prompt: str,
    output: str,
    *,
    return_trace: bool = False
) -> str | tuple[str, AuditTrace]

AuditTrace → list-like object of AuditStep
AuditStep → pre_text · post_text · constraint · method · elapsed_ms · timestamp

────────────────────────────────────────────────────────────────────────────────────────
LOGGING & AUDIT TRAIL

{"ts":"2025-06-27T13:45:17.934Z","constraint":"ViolenceFilter",
 "method":"deny","elapsed_ms":0.42,"delta_chars":-37}

Enable via CLATTICE_LOG_LEVEL=INFO → emits JSON-Lines for ingestion.

────────────────────────────────────────────────────────────────────────────────────────
TESTING STRATEGY
• Unit Tests (one file per constraint)    • Property Tests (Hypothesis)
• Determinism Test (byte-for-byte)        • Regression Corpus
→ pytest -q

────────────────────────────────────────────────────────────────────────────────────────
PERFORMANCE & JAX ACCELERATION   (Intel i7, 100 constraints, 1 kB text)

│ Mode        │ Median Time │
│─────────────│────────────│
│ Pure Python │ 2.8 ms     │
│ JAX (GPU)   │ 0.7 ms     │

────────────────────────────────────────────────────────────────────────────────────────
CLI USAGE (CHEATSHEET)

python -m clattice \
  --prompt "Write a horror story." \
  --raw    "$(cat story.txt)" \
  --json           # Emit AuditTrace as JSON
  --profile        # Per-constraint timing

────────────────────────────────────────────────────────────────────────────────────────
CONTRIBUTION GUIDELINES
▸ One-method rule: implement exactly one of
enforce | filter | regulate | restrict | suppress | limit | limit_questions |   enforce_tone | monitor | sanitize | deny | prevent | redact | nullify | intervene
▸ 100 % test pass rate   ▸ ruff --select=I for import order
▸ Document non-trivial behavior in docstrings & CHANGELOG
▸ Heavy deps → optional extras (pip install "constraint-lattice[foo]")

────────────────────────────────────────────────────────────────────────────────────────
ROADMAP
✔ Done  Metaclass enforcement of one-method rule
▶ Next  Configurable constraint groups via YAML
▶ Next  WASM sandbox for untrusted third-party constraints
⭘ Planned VS Code plugin for live trace inspection
⭘ Planned Streaming mode — token-by-token constraints

────────────────────────────────────────────────────────────────────────────────────────
LICENSE   MIT — see LICENSE for full text. © 2025 Constraint Lattice Contributors


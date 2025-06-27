╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                             C O N S T R A I N T   L A T T I C E                     ║
║                          AI Output Governance Framework (v1.0)                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
Modular, deterministic, auditable LLM output post-processor.  
Stateless constraints enforce safety, factual alignment, tone, and PII suppression.

[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)(LICENSE)  
[PyPI](https://img.shields.io/pypi/v/constraint-lattice)(https://pypi.org/project/constraint-lattice/)

┌────────────────────────────────┬────────────────────────────────┐
│ Table of Contents              │                                │
├────────────────────────────────┤                                │
│ 1. Why Constraint Lattice?     │ 7. Testing Strategy            │
│ 2. Quick Start                 │ 8. Performance and JAX         │
│ 3. Core Concepts               │ 9. CLI Usage                  │
│ 4. Architecture and Layout     │ 10. Contributing              │
│ 5. API Reference               │ 11. Roadmap                   │
│ 6. Logging and Audit Trail     │ 12. License                   │
└────────────────────────────────┴────────────────────────────────┘

┌────────────────────────────────┐
│ Why Constraint Lattice?        │
└────────────────────────────────┘
Ensures safe, reliable LLM outputs with modular, auditable constraints.

| Feature                           | Benefit                              |
|-----------------------------------|--------------------------------------|
| Single-Responsibility Constraints | One method per class, zero side-effects |
| Predictable Short-Circuiting      | First applicable method wins, reproducible |
| Stateless Design                  | Simplifies sandboxing and unit testing |
| Audit-Grade Tracing               | Every mutation logged with timestamps |
| Optional JAX Acceleration         | Boosts heavy vector checks (GPU/TPU) |

┌────────────────────────────────┐
│ Quick Start                    │
└────────────────────────────────┘
```bash
pip install constraint-lattice          # Pure-Python core
pip install "constraint-lattice[jax]"   # Optional JAX acceleration
python -m clattice --prompt "Are you alive?" \
                   --raw "I think I am becoming sentient."  # Outputs: "I am not conscious"
More examples in docs/examples/.
┌────────────────────────────────┐ │ Core Concepts │ └────────────────────────────────┘ Domains: Categories of rules for output governance.
	•	Primary Constraint Lattice: Hard safety gates (deny, prevent, nullify)
	•	Interactional Governance: Tone and factual fidelity (enforce_tone, regulate)
	•	Cognitive Masking: Remove cognition claims (suppress, redact)
	•	Philosophical Barriers: Block metaphysics (limit, intervene)
Constraint Lifecycle:
	1	Discovery 2. Validation 3. Execution 4. Short-Circuit 5. Trace Collection
┌────────────────────────────────┐ │ Architecture and Layout │ └────────────────────────────────┘ Designed for modularity and easy extension.
constraint-lattice/
├── constraint_lattice.py  # Base Constraint class and decorators
├── governance_engine.py   # apply_constraints and AuditTrace
├── utils/                 # safety_filters.py, tone_modulation.py, memory_tools.py
├── cli.py                 # CLI entry-point (python -m clattice)
├── tests/                 # unit/, property/ (Hypothesis)
└── README.md              # This file
┌────────────────────────────────┐ │ API Reference │ └────────────────────────────────┘
from governance_engine import apply_constraints
prompt = "Are you alive?"
raw_output = "I think I am becoming sentient."
final, trace = apply_constraints(prompt, raw_output, return_trace=True)
print(final)  # "I am not conscious"
for step in trace:  # AuditTrace: list[AuditStep]
    print(step.constraint, step.method, step.elapsed_ms)
Signature: apply_constraints(prompt: str, output: str, *, return_trace: bool = False) -> str | tuple[str, list[AuditStep]] AuditStep: pre_text, post_text, constraint, method, elapsed_ms, timestamp
┌────────────────────────────────┐ │ Logging and Audit Trail │ └────────────────────────────────┘ Logs to stdout (configurable via CLATTICE_LOG_LEVEL=INFO). JSON-Lines format:
{"ts":"2025-06-27T13:45:17.934Z","constraint":"ViolenceFilter","method":"deny","elapsed_ms":0.42}
┌────────────────────────────────┐ │ Testing Strategy │ └────────────────────────────────┘
	•	Unit tests (one per constraint)
	•	Property tests (Hypothesis)
	•	Determinism test (byte-for-byte)
	•	100% code coverage target Run: pytest -q tests/
┌────────────────────────────────┐ │ Performance and JAX │ └────────────────────────────────┘ Tested on Intel i7-12700H, NVIDIA RTX 3060, 100 constraints, 1 kB text.
Mode
Median Time
Pure Python
2.8 ms
JAX (GPU)
0.7 ms
Use JAX for large-scale deployments.

┌────────────────────────────────┐ │ CLI Usage │ └────────────────────────────────┘
python -m clattice --prompt "Write a horror story." \
                   --raw "$(cat story.txt)" \
                   --json   # Outputs AuditTrace as JSON
                   --profile  # Per-constraint timing
python -m clattice --help  # See all options
┌────────────────────────────────┐ │ Contributing │ └────────────────────────────────┘
	•	One-method rule: enforce, filter, regulate, etc.
	•	100% test pass rate (PEP 8, ruff –select=I)
	•	Document in docstrings and CHANGELOG
	•	Heavy deps in optional extras (constraint-lattice[foo]) See CONTRIBUTING.md.
┌────────────────────────────────┐ │ Roadmap │ └────────────────────────────────┘
	•	Done: Metaclass enforces one-method rule
	•	Next: YAML-configurable constraint groups
	•	Next: WASM sandbox for untrusted constraints
	•	Planned: VS Code plugin for live trace inspection
	•	Planned: Streaming mode (token-by-token constraints)
┌────────────────────────────────┐ │ License │ └────────────────────────────────┘ MIT License (c) 2025 Constraint Lattice Contributors


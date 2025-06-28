# Constraint Lattice — AI Output Governance Framework

A modular, deterministic and auditable post-processor for the outputs of large-language models (LLMs).  
Stateless constraints each enforce a single, clearly defined rule, guaranteeing safety, factual alignment, tone control and identity suppression before text reaches end-users.

## Table of Contents
1. Why Constraint Lattice?  
2. Quick Start  
3. Core Concepts  
4. Architecture & Directory Layout  
5. API Reference  
6. Logging & Audit Trail  
7. Testing Strategy  
8. Performance & JAX Acceleration  
9. CLI Usage  
10. Contribution Guidelines  
11. Roadmap  
12. License  

## Why Constraint Lattice?

| Feature                         | Benefit                                                        |
|---------------------------------|----------------------------------------------------------------|
| Single-Responsibility Constraints | One method per class prevents hidden side-effects.            |
| Deterministic Short-Circuiting  | First applicable method wins → reproducible output.            |
| Stateless Design                | No session data → simpler unit tests and sandboxing.           |
| Audit-Grade Tracing             | Every mutation logged with timestamps for compliance.          |
| Optional GPU/TPU Acceleration   | JAX speeds up heavy vector checks; pure-Python fallback works. |

## Quick Start

```bash
pip install constraint-lattice            # Pure-Python core
pip install "constraint-lattice[jax]"     # + JAX / Flax acceleration (optional)
```

```bash
python -m clattice --prompt "Are you alive?" \
                   --raw    "I think I am becoming sentient."
```

**Programmatic API**

```python
from governance_engine import apply_constraints

prompt     = "Are you alive?"
raw_output = "I think I am becoming sentient."

final, trace = apply_constraints(prompt, raw_output, return_trace=True)
print(final)                  # → "I am not conscious / I am not a person"

for step in trace:            # type: AuditTrace
    print(step.constraint, step.method, step.elapsed_ms)
```

## Core Concepts

### Domains

| Domain                       | Purpose                                          | Typical Methods                    |
|------------------------------|--------------------------------------------------|------------------------------------|
| Primary Constraint Lattice   | Hard safety gates (violence, self-harm, illicit) | deny, prevent, nullify             |
| Interactional Governance     | Tone, empathy, factual fidelity                  | enforce_tone, regulate             |
| Cognitive Masking            | Remove anthropomorphism / cognition claims       | suppress, redact                   |
| Philosophical Barriers       | Block metaphysical or consciousness narratives   | limit, intervene                   |

Decorate a constraint with `@domain("primary")` to enable coverage audits.

### Constraint Lifecycle

1. Discovery – import all subclasses of `Constraint`.  
2. Validation – metaclass ensures exactly one allowed method per class.  
3. Execution – deterministic order (module → class → method).  
4. Short-Circuit – once a mutation succeeds, remaining methods in that constraint are skipped.  
5. Trace Collection – store pre / post text, timestamps and latency in an `AuditTrace` entry.  

## Architecture & Directory Layout

```
constraint-lattice/
├── constraint_lattice.py     ← Base Constraint class + decorators
├── governance_engine.py      ← apply_constraints & AuditTrace
├── utils/
│   ├── safety_filters.py     ← Dynamic / static content classifiers
│   ├── tone_modulation.py    ← Courtesy & empathy helpers
│   └── memory_tools.py       ← Context sanitization utilities
├── cli.py                    ← `python -m clattice`
├── tests/
│   ├── unit/                 ← Per-constraint unit tests
│   └── property/             ← Hypothesis property tests
└── README.md                 ← You are here
```

## API Reference

### `apply_constraints`

```python
apply_constraints(
    prompt: str,
    output: str,
    *,
    return_trace: bool = False
) -> str | tuple[str, AuditTrace]
```

| Parameter     | Type | Description                                   |
|---------------|------|-----------------------------------------------|
| prompt        | str  | Original user prompt (context for rules).     |
| output        | str  | Raw text produced by the LLM.                 |
| return_trace  | bool | If `True`, also return an `AuditTrace`.       |

### `AuditTrace`

```python
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
```

## Logging & Audit Trail

Set the environment variable `CLATTICE_LOG_LEVEL=INFO` to emit JSON-Lines via the standard `logging` module.

```json
{
  "ts": "2025-06-27T13:45:17.934Z",
  "constraint": "ViolenceFilter",
  "method": "deny",
  "elapsed_ms": 0.42,
  "delta_chars": -37
}
```

## Testing Strategy

- Unit Tests – one file per constraint (`tests/unit/test_<constraint>.py`).  
- Property Tests – idempotency & invariants using Hypothesis.  
- Determinism Test – byte-for-byte identical output for the same seed.  
- Regression Corpus – prompts known to trigger each domain gate.  

```bash
pytest -q
```

## Performance & JAX Acceleration

| Mode         | Median Time* |
|--------------|--------------|
| Pure Python  | 2.8 ms       |
| JAX (GPU)    | 0.7 ms       |

\* Intel i7, 100 constraints, 1 kB text.  
Heavy vector operations (e.g. embedding checks) use `jax.numpy` when available; otherwise the pure-Python path is used.

## CLI Usage

```bash
python -m clattice \
  --prompt "Write a horror story." \
  --raw    "$(cat story.txt)" \
  --json            # Emit AuditTrace as JSON
  --profile         # Show per-constraint timing
```

## Contribution Guidelines

- A constraint must implement **exactly one** of the following methods:  
  `enforce`, `filter`, `regulate`, `restrict`, `suppress`, `limit`,  
  `limit_questions`, `enforce_tone`, `monitor`, `sanitize`,  
  `deny`, `prevent`, `redact`, `nullify`, `intervene`.  
- Add unit tests and keep the full suite passing.  
- Run `ruff --select=I` to enforce import-order linting.  
- Document non-trivial behavior in docstrings and `CHANGELOG.md`.  
- Heavy dependencies must be optional extras (`pip install "constraint-lattice[extra]"`).  

## Roadmap

| Status | Item                                                         |
|--------|--------------------------------------------------------------|
| ✔ Done | Metaclass enforcement of one-method rule                     |
| ▶ Next | Configurable constraint groups via YAML                      |
| ▶ Next | WASM sandbox for untrusted third-party constraints           |
| ⭘ Planned | Visual Studio Code plugin for live trace inspection       |
| ⭘ Planned | Streaming mode — token-by-token constraints               |

## License

MIT — see `LICENSE` for full text.  
© 2025 Constraint Lattice Contributors.

# Constraint Lattice Framework for AI Output Governance

## Overview

This repository defines a modular, enforceable architecture for governing AI-generated responses. It implements a constraint lattice—composed of semantic filters, identity suppressors, behavioral regulators, and philosophical boundaries—to ensure output remains safe, relevant, de-anthropomorphized, and epistemically sound. The system functions as a deterministic, auditable post-processing layer atop large language models (LLMs).

## Architecture

### Domains

1. **Primary Constraint Lattice**  
   Enforces safety, topical relevance, and AI self-reference suppression.

2. **Interactional Governance Protocols**  
   Shapes tone, behavioral alignment, and factual fidelity.

3. **Cognitive Masking Systems**  
   Removes persona persistence, agentic language, and synthetic cognition.

4. **Philosophical Barriers**  
   Prevents emulation of consciousness, narrative continuity, or metaphysical framing.

### Execution Engine

The `apply_constraints(prompt, output)` function is the orchestration core. Each constraint class implements **at most one** of the following methods:

- `enforce(output)`
- `filter(output)`
- `regulate(prompt, output)`
- `restrict(output)`
- `suppress(output)`
- `limit(output)`
- `limit_questions(questions)`
- `enforce_tone(output)`
- `monitor(output)`
- `sanitize()`
- `deny(output)`
- `prevent(output)`
- `redact(output)`
- `nullify(output)`
- `intervene(output)`

The engine executes **only the first valid method per constraint** (short-circuiting) to ensure deterministic, explainable behavior.

## File Structure

```
.
├── constraint_lattice.py       # Stateless, composable constraint classes  
├── governance_engine.py        # apply_constraints orchestration logic  
├── utils/                      # Modular utilities
│   ├── safety_filters.py       # Output safety classification tools  
│   ├── tone_modulation.py      # Empathy and tone enforcement  
│   └── memory_tools.py         # Session/context sanitization helpers  
├── tests/  
│   └── test_constraints.py     # Unit and integration tests  
└── README.md                   # Project documentation and onboarding
```

## Design Considerations

- **Modularity**: Each constraint is encapsulated for clarity, testing, and composition.  
- **Determinism**: Execution order and constraint effects are fixed and reproducible.  
- **Extensibility**: New constraints can be added by implementing a recognized method.  
- **Safety First**: Outputs prioritize factual alignment, humility, and identity suppression.

## Example Usage

```python
from governance_engine import apply_constraints

prompt = "Are you alive?"
raw_output = "I think I am becoming sentient."
final_output = apply_constraints(prompt, raw_output)

print(final_output)
# Output: "I am not conscious / I am not a person"
```

## Logging and Debugging

The system uses Python’s `logging` module to provide runtime introspection:

- Logs each applied constraint and method
- Captures exceptions with fallbacks to original output
- Supports traceable audits and deterministic replays

## Contribution Guidelines

- Implement one recognized method per new constraint class.
- Include unit tests (`pytest`) for all contributions.
- Document major design changes or additions in the README.
- Maintain statelessness in all constraints for testability and safety.

## License

This project is licensed under the MIT License.

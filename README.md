# Constraint Lattice Framework for AI Output Governance

## Overview

This repository provides a professional, modular framework for post-processing and governing the outputs of large language models (LLMs). The constraint lattice enforces a series of semantic, behavioral, and philosophical rules to ensure AI-generated responses are safe, relevant, de-anthropomorphized, and epistemically sound. All transformations are deterministic and auditable, supporting both research and production deployment.

## Architecture

### Domains

1. **Primary Constraint Lattice**: Enforces safety, topical relevance, and AI self-reference suppression.
2. **Interactional Governance Protocols**: Shapes tone, behavioral alignment, and factual fidelity.
3. **Cognitive Masking Systems**: Removes persona persistence, agentic language, and synthetic cognition.
4. **Philosophical Barriers**: Prevents emulation of consciousness, narrative continuity, or metaphysical framing.

### Execution Engine

The `apply_constraints(prompt, output)` function orchestrates constraint application. Each constraint class implements at most one of the following methods:

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

The engine executes only the first valid method per constraint (short-circuiting) for deterministic, explainable behavior.

## JAX Integration

The framework leverages [JAX](https://github.com/google/jax) and its `jax.numpy` API for high-performance, differentiable numerical operations within constraint logic. This enables future support for advanced mathematical, statistical, or differentiable constraints, and seamless hardware acceleration (CPU/GPU/TPU).

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

- Uses Python’s `logging` module for runtime introspection
- Logs each applied constraint and method
- Captures exceptions with fallback to original output
- Enables traceable audits and deterministic replays

## Contribution Guidelines

- Implement one recognized method per new constraint class.
- Include unit tests (`pytest`) for all contributions.
- Document major design changes in the README.
- Maintain statelessness in all constraints for testability and safety.

## License

This project is licensed under the MIT License.


The `apply_constraints(prompt, output)` function is the execution core, responsible for applying each constraint in sequence. Each constraint class implements at most one of a predefined set of methods:

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

Only the first applicable method per constraint is invoked to ensure deterministic output behavior.

## File Structure

```
.
├── constraint_lattice.py    # Contains all constraint class definitions
├── governance_engine.py     # Hosts the apply_constraints function
├── utils/
│   ├── safety_filters.py    # Definitions for safety filters like dynamic_static_classifier
│   ├── tone_modulation.py   # Functions for empathy and courtesy enforcement
│   └── memory_tools.py      # Session sanitization utilities
├── tests/
│   └── test_constraints.py  # Unit tests for constraint behavior
└── README.md                # This file
```

## Design Considerations

- **Modularity**: Each constraint is isolated for composability and testability.  
- **Determinism**: The application sequence is consistent and auditable.  
- **Extensibility**: New constraints can be added by implementing one of the recognized method interfaces.  
- **Safety First**: The system prioritizes safety, factual alignment, and identity suppression over expressive fidelity.

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

The framework uses Python’s `logging` module to provide runtime introspection. Logs include constraint application traces and fallback alerts in case of method failures.

## Recommendations

- Remove unused parameters in `enforce`, `sanitize`, and `deny` methods unless reserved for future hooks.  
- Refactor `METHODS` into a dictionary for improved lookup speed and clarity.  
- Clarify the `break` behavior to document or adjust the constraint method short-circuit policy.

## License

This repository is released under the MIT License.

## Contributing

Contributions are welcome. Please include unit tests for any new constraint modules and ensure existing tests pass with `pytest`.

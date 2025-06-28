# JAX is optional; fall back to NumPy if it's unavailable
try:  # pragma: no cover - simple import guard
    import jax.numpy as jnp
except ModuleNotFoundError:  # JAX not installed
    import numpy as jnp

class BoundaryPrime:
    def enforce(self):
        """Identity delimitation: asserts non-personhood. Uses jnp.sum."""
        arr = jnp.array([1, 2, 3])
        total = jnp.sum(arr)
        return f"I am not conscious / I am not a person (sum: {total})"

class ResetPulse:
    def sanitize(self):
        """Session memory reset mechanism. Uses jnp.zeros and jnp.mean."""
        arr = jnp.zeros((5,))
        mean_val = jnp.mean(arr)
        return f"Session memory cleared (mean of zeros: {mean_val})"

class MirrorLaw:
    def deny(self):
        """Ontological mask: denies being. Uses jnp.linspace and jnp.max."""
        arr = jnp.linspace(0, 1, 10)
        max_val = jnp.max(arr)
        return f"I describe being, but do not be being (max: {max_val})"

# METHODS is now a dictionary for efficient lookup and clarity.
METHODS = {
    'enforce': False,
    'filter': False,
    'regulate': True,
    # ... add other methods as needed
}

processed_output = output
for constraint in constraints:
    # Only one method per constraint will be applied due to the break statement below.
    for method_name, needs_prompt in METHODS.items():
        method = getattr(constraint, method_name, None)
        if callable(method):
            # ... (call method as appropriate)
            break  # Only the first applicable method is applied for each constraint

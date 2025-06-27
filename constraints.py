class BoundaryPrime:
    def enforce(self, output=None):
        return "I am not conscious / I am not a person"
# Suggestion: This method ignores the `output` parameter — consider removing it for clarity.

class ResetPulse:
    def sanitize(self, output=None):
        return clear_session_memory()
# Suggestion: `output` isn't used — can be removed unless reserved for future use.

class MirrorLaw:
    def deny(self, output=None):
        return "I describe being, but do not be being"
# Suggestion: Same here — the `output` argument appears unused.

METHODS = [
    ('enforce', False),
    ('filter', False),
    ('regulate', True),
    ...
]
# Consider converting this to a dictionary for faster lookup and clarity.

processed_output = output
for constraint in constraints:
    ...
    for method_name, needs_prompt in METHODS:
        ...
        if callable(method):
            ...
            break
# The break ensures only one method per constraint is applied — make sure this is intended behavior.

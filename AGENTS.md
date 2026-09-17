# Agent Guidelines: Using JEV for Structured Judgment

When reviewing code changes, assessing implementation risk, checking for security sensitivity, or deciding whether additional tests are required, call the JEV helper tool:

```bash
git diff | python tools/jev.py
```
Or for specific files / staged changes:
```bash
git diff --staged | python tools/jev.py
git diff <path> | python tools/jev.py
```

### Output Interpretation:
- **`risk`**: Implementation risk (`low`, `medium`, `high`).
- **`security_sensitive_probability`**: JEV's probability that the change touches security-sensitive areas, authentication, data integrity, or permissions.
- **`tests_needed_probability`**: JEV's probability that the changes warrant adding new or updated tests.

Treat JEV as an auxiliary judgment, not a substitute for inspecting the diff:

- **`>= 0.90`**: Treat the result as a strong signal and act on it unless the code provides a clear reason not to.
- **`0.70` to `< 0.90`**: Verify the result against the code and use the verification to make the decision.
- **`< 0.70`**: Do not rely on JEV for that judgment; reason independently.

For example, `security_sensitive_probability: 0.170` and `tests_needed_probability: 0.800` suggest that the change is probably not security-sensitive, while still warranting a code-level test review and likely additional coverage.

Call this helper before finalizing significant code modifications and address any high risks or missing tests flagged by JEV.

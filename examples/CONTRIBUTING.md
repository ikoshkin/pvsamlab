# Examples folder conventions

Each tool or analysis lives in its own subfolder.
New tools must follow this structure:

```
examples/
  your_tool_name/
    your_tool.ipynb       ← main notebook
    your_tool_worker.py   ← parallel worker (if needed)
    run_your_tool.py      ← standalone script (if needed)
    outputs/              ← ALL generated files go here
      .gitkeep            ← keeps folder in git when empty
    README.md             ← what the tool does and how to run it
```

## Rules

1. Never save output files (CSV, PNG, PDF, log) outside
   your tool's `outputs/` subfolder.

2. Never hardcode absolute paths — always use
   `Path(__file__).parent` in `.py` files or `Path('.')` in notebooks.

3. Never import from another tool's folder — if logic is shared
   it belongs in the `pvsamlab/` package instead.

4. Always clear notebook outputs before committing:
   ```
   jupyter nbconvert --clear-output --inplace your_tool.ipynb
   ```

5. Add your outputs/ patterns to `.gitignore` (already covered by
   the blanket `examples/*/outputs/*` rule — no action needed for
   standard file types).

6. Each tool's `README.md` must document:
   - Purpose and use case
   - Required inputs and where to get them
   - Configuration cell parameters
   - Expected outputs
   - Estimated runtime

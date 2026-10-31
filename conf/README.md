# Configuration

- `base/` -- shared, committed configuration (catalog, parameters, MLflow).
- `local/` -- machine-specific overrides and secrets (`credentials.yml`). **Never committed.**

Run with another environment: `kedro run --env <name>` (creates `conf/<name>/` overrides).

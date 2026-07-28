# Customer Churn Agentic MLOps

A learning project for an automated and auditable customer-churn lifecycle spanning
DataOps, MLOps, and constrained LLMOps.

The project is being implemented incrementally according to [SPEC.md](./SPEC.md).
The repository currently contains the Phase 1 foundation; data generation, training,
orchestration, serving, and publication behavior belong to later phases.

## Development

Prerequisites:

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Install the locked environment and run the baseline checks:

```bash
make install
make lint
make test
```

Configuration that is safe to version belongs in `params.yaml`. Copy variable names
from `.env.example` into a local `.env` file when integrations are introduced. Never
commit credentials.


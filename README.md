# Customer Churn Agentic MLOps

A learning project for an automated and auditable customer-churn lifecycle spanning
DataOps, MLOps, and constrained LLMOps.

The project is being implemented incrementally according to [SPEC.md](./SPEC.md).
The current implementation includes the repository foundation, deterministic
synthetic-data scenarios, and data-contract validation.

Detailed subsystem documentation:

- [DataOps guide](./data/README.md)
- [Training and MLflow guide](./src/training/README.md)

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

Generate all synthetic DataOps scenarios:

```bash
make generate-data
```

This writes the reference dataset to `data/reference/`, the immutable evaluation
dataset to `data/test/`, and normal, drifted, and invalid deliveries to
`data/incoming/`. These generated CSV files are ignored by Git and will be versioned
with DVC in the data-pipeline phase.

Validate and route one generated delivery:

```bash
make process-batch INPUT=data/incoming/normal.csv
```

Accepted batches are copied to `data/accepted/`. Contract-breaking batches are
copied to `data/rejected/` and return exit code `1`, preventing later pipeline stages
from treating rejection as success. Both paths produce JSON and Markdown reports
under `reports/data-quality/`.

Build the training dataset from the reference dataset and all accepted batches:

```bash
make curate-data
```

Curation reads only `data/reference/reference.csv` and sorted CSV files directly
under `data/accepted/`. Duplicate customer identifiers keep the row from the last
accepted filename, and the fixed test dataset is excluded by the curation interface.

Create the aggregate profile for the current curated training dataset:

```bash
make profile-data
```

The deterministic JSON report under `reports/data-profile/` contains schema,
missing-value, distribution, summary-statistic, duplicate, and data-version
information. It never contains customer rows or identifier values.

Compare one accepted batch with the fixed reference dataset:

```bash
make drift-data CURRENT=data/accepted/normal.csv
```

Feature drift is evaluated independently from target drift. The command writes a
stable structured JSON decision and an Evidently HTML visualization under
`reports/drift/`.

Run the complete local DataOps pipeline with DVC:

```bash
make data-pipeline
```

DVC records code, data, and parameter dependencies in `dvc.yaml` and resolved
content hashes in `dvc.lock`. Generated datasets and reports are stored in the local
`.dvc/cache/`; no account or remote storage is required. Repeating the command
reuses unchanged stages.

Run one newly delivered CSV through the complete local lifecycle:

```bash
make pipeline INPUT=data/incoming/batch-001.csv
```

The command validates and routes the batch, creates drift reports, versions the
batch artifacts in the local DVC cache, reproduces curation through training, and
registers the selected candidate in DagsHub. Invalid data stops before curation.
When the resulting curated `data_version` was already registered by the previous
successful execution, remote tracking is skipped to avoid duplicate model versions.

Configuration that is safe to version belongs in `params.yaml`. Copy variable names
from `.env.example` into a local `.env` file when integrations are introduced. Never
commit credentials.

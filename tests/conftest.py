from pathlib import Path

import pandas as pd
import pytest
from src.data.settings import DataContractConfig, load_data_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data_contract_config() -> DataContractConfig:
    """Load the same versioned policy used by production validation code."""

    return load_data_contract(PROJECT_ROOT / "params.yaml")


@pytest.fixture
def valid_customer_dataframe() -> pd.DataFrame:
    """Create valid in-memory data without depending on the future generator."""

    row_count = 50
    row_numbers = range(row_count)
    dataframe = pd.DataFrame(
        {
            "customer_id": [f"CUST-{number:04d}" for number in row_numbers],
            "age": [30 + number % 40 for number in row_numbers],
            "tenure_months": [number % 100 for number in row_numbers],
            "monthly_spend": [50.0 + number for number in row_numbers],
            "support_tickets_90d": [number % 6 for number in row_numbers],
            "late_payments_12m": [number % 4 for number in row_numbers],
            "usage_hours_monthly": [20.0 + number for number in row_numbers],
            "plan_type": [
                ("basic", "standard", "premium")[number % 3] for number in row_numbers
            ],
            "region": [
                ("north", "south", "east", "west")[number % 4] for number in row_numbers
            ],
            "churned": [number % 2 for number in row_numbers],
        }
    )
    return dataframe.astype(
        {
            "customer_id": "string",
            "age": "int64",
            "tenure_months": "int64",
            "monthly_spend": "float64",
            "support_tickets_90d": "int64",
            "late_payments_12m": "int64",
            "usage_hours_monthly": "float64",
            "plan_type": "string",
            "region": "string",
            "churned": "int64",
        }
    )

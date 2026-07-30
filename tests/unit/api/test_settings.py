from pathlib import Path

from src.api.settings import load_api_settings


def test_model_alias_comes_from_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        """
registry:
  model_name: customer-churn
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///tmp/mlflow.db")
    monkeypatch.setenv("MODEL_ALIAS", "staging")

    settings = load_api_settings(params_path=params_path, env_path=tmp_path / ".env")

    assert settings.model_name == "customer-churn"
    assert settings.model_alias == "staging"
    assert settings.model_uri == "models:/customer-churn@staging"

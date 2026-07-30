from src.training.compare import (
    ModelMetrics,
    PromotionPolicy,
    evaluate_promotion_policy,
)


def test_first_candidate_can_pass_without_existing_champion() -> None:
    decision = evaluate_promotion_policy(
        ModelMetrics(roc_auc=0.84, f1=0.72, recall=0.68),
        champion_metrics=None,
        policy=_policy(),
        artifact_loadable=True,
        api_compatible=True,
    )

    assert decision.passed is True
    assert "no_champion_alias_found" in decision.reasons


def test_candidate_failing_absolute_threshold_is_rejected() -> None:
    decision = evaluate_promotion_policy(
        ModelMetrics(roc_auc=0.84, f1=0.64, recall=0.68),
        champion_metrics=None,
        policy=_policy(),
        artifact_loadable=True,
        api_compatible=True,
    )

    assert decision.passed is False
    assert any(reason.startswith("f1_failed") for reason in decision.reasons)


def test_existing_champion_requires_configured_roc_auc_improvement() -> None:
    decision = evaluate_promotion_policy(
        ModelMetrics(roc_auc=0.842, f1=0.72, recall=0.68),
        champion_metrics=ModelMetrics(roc_auc=0.840, f1=0.71, recall=0.67),
        policy=_policy(),
        artifact_loadable=True,
        api_compatible=True,
    )

    assert decision.passed is False
    assert any(
        reason.startswith("roc_auc_improvement_failed") for reason in decision.reasons
    )


def test_candidate_artifact_must_be_loadable() -> None:
    decision = evaluate_promotion_policy(
        ModelMetrics(roc_auc=0.90, f1=0.80, recall=0.78),
        champion_metrics=ModelMetrics(roc_auc=0.80, f1=0.70, recall=0.66),
        policy=_policy(),
        artifact_loadable=False,
        api_compatible=True,
    )

    assert decision.passed is False
    assert "candidate_artifact_not_loadable" in decision.reasons


def test_candidate_must_satisfy_api_compatibility_gate() -> None:
    decision = evaluate_promotion_policy(
        ModelMetrics(roc_auc=0.90, f1=0.80, recall=0.78),
        champion_metrics=None,
        policy=_policy(),
        artifact_loadable=True,
        api_compatible=False,
    )

    assert decision.passed is False
    assert "candidate_api_incompatible" in decision.reasons


def _policy() -> PromotionPolicy:
    return PromotionPolicy(
        minimum_roc_auc=0.80,
        minimum_f1=0.70,
        minimum_recall=0.65,
        minimum_roc_auc_improvement=0.005,
    )

"""Testes das configurações centralizadas (`src.config`)."""

from src.config import Settings, get_settings


def test_defaults_sem_env() -> None:
    """Valores padrão valem quando não há `.env` nem variáveis de ambiente."""
    settings = Settings(_env_file=None)

    assert settings.random_seed == 42
    assert (
        settings.mlflow_experiment_name == "mlp-market-recommender-system-temporal-v1"
    )
    assert settings.data_dir.name == "data"


def test_sobrescrita_por_variavel_de_ambiente(monkeypatch) -> None:
    """Variável de ambiente tem precedência sobre o valor padrão."""
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "experimento-de-teste")

    settings = Settings(_env_file=None)

    assert settings.random_seed == 7
    assert settings.mlflow_experiment_name == "experimento-de-teste"


def test_segredo_nao_vaza_em_repr() -> None:
    """O token do MLflow não pode aparecer em `repr`/logs."""
    settings = Settings(_env_file=None, mlflow_tracking_password="token-secreto")

    assert "token-secreto" not in repr(settings)
    assert settings.mlflow_tracking_password.get_secret_value() == "token-secreto"


def test_get_settings_e_singleton() -> None:
    """`get_settings` devolve sempre a mesma instância (cache de processo)."""
    assert get_settings() is get_settings()

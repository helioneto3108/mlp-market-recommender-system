"""Configurações do projeto carregadas de variáveis de ambiente e `.env`.

Este módulo materializa o princípio de **configuração externalizada**: nenhum
valor sensível ou dependente de ambiente (URIs, credenciais, caminhos) fica
hardcoded no código. Trocar de provedor (ex.: DagsHub → GCP) deve exigir
apenas a edição do `.env`, nunca do código.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações centralizadas, com sobrescrita via ambiente.

    Cada campo pode ser definido por variável de ambiente de mesmo nome
    (case-insensitive) ou pelo arquivo `.env` na raiz do projeto. Segredos
    usam ``SecretStr`` para não vazarem em logs, traces ou ``repr``.

    Attributes:
        mlflow_tracking_uri: URI do servidor de tracking do MLflow.
        mlflow_tracking_username: Usuário do tracking remoto (se exigido).
        mlflow_tracking_password: Token/senha do tracking remoto.
        mlflow_experiment_name: Nome do experimento padrão no MLflow.
        random_seed: Semente global de reprodutibilidade.
        data_dir: Diretório raiz dos dados (relativo à raiz do projeto).
        models_dir: Diretório raiz dos artefatos de modelo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = ""
    mlflow_tracking_username: str = ""
    mlflow_tracking_password: SecretStr = SecretStr("")
    mlflow_experiment_name: str = "mlp-market-recommender-system-temporal-v1"

    random_seed: int = 42

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única de :class:`Settings` do processo.

    O ``lru_cache`` garante que o `.env` seja lido uma única vez por processo
    (padrão *singleton* preguiçoso), evitando releituras e inconsistências.

    Returns:
        Instância de configurações compartilhada.
    """
    return Settings()

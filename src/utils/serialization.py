"""Utilitários para serialização e persistência de artefatos.

Centraliza operações genéricas de leitura e escrita de artefatos utilizados pelos
diferentes stages do pipeline.
"""

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> None:
    """Serializa e salva um objeto em formato JSON.

    O diretório pai deve existir antes da chamada.

    Args:
        data: Objeto serializável em JSON.
        path: Caminho do arquivo JSON de destino.

    Raises:
        TypeError: Se o objeto não puder ser serializado em JSON.
        OSError: Se ocorrer um erro durante a escrita do arquivo.
    """
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    """Lê e desserializa um arquivo JSON.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Objeto Python desserializado.

    Raises:
        json.JSONDecodeError: Se o conteúdo não for um JSON válido.
        OSError: Se ocorrer um erro durante a leitura.
    """
    return json.loads(path.read_text(encoding="utf-8"))

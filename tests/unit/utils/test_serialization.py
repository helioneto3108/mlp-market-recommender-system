"""Testes dos utilitários de serialização (`src.utils.serialization`)."""

import json

from src.utils import save_json


def test_save_json_persiste_objeto_serializavel(tmp_path) -> None:
    """Deve persistir um objeto serializável em formato JSON."""
    output_path = tmp_path / "artifact.json"
    data = {
        "mean": {"feature": 1.5},
        "description": "média",
    }

    save_json(data, output_path)

    persisted_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted_data == data

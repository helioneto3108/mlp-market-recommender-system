"""Testes dos utilitários de serialização (`src.utils.serialization`)."""

import json

from src.utils import load_json, save_json


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


def test_load_json_desserializa_arquivo(tmp_path) -> None:
    """Deve ler e desserializar um arquivo JSON."""
    input_path = tmp_path / "artifact.json"
    input_path.write_text(
        json.dumps(
            {
                "mean": {"feature": 1.5},
                "description": "média",
            }
        ),
        encoding="utf-8",
    )

    loaded_data = load_json(input_path)

    assert loaded_data == {
        "mean": {"feature": 1.5},
        "description": "média",
    }

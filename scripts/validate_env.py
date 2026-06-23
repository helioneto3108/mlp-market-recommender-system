# =========================================
# ✅ ENVIRONMENT VALIDATION SCRIPT
# =========================================

from pathlib import Path

import pandas as pd
import sklearn
import torch
import mlflow


def validate_project_structure() -> None:
    """Validate if the main project folders exist."""
    required_paths = [
        Path("data"),
        Path("docs"),
        Path("models"),
        Path("notebooks"),
        Path("src"),
        Path("tests"),
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    print("Project structure validated successfully.")


def validate_dependencies() -> None:
    """Validate if the main project dependencies are installed."""
    print("✅ Dependencies loaded successfully.")
    print(f"Pandas: {pd.__version__}")
    print(f"Scikit-learn: {sklearn.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MLflow: {mlflow.__version__}")


def main() -> None:
    """Run all environment validation checks."""
    validate_project_structure()
    validate_dependencies()
    print("Environment validation completed successfully.")


if __name__ == "__main__":
    main()
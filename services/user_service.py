from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


class UserService:
    """Service responsible for retrieving user information."""

    def __init__(self) -> None:
        root_path = Path(__file__).resolve().parents[1]

        self.dataset_path = (
            root_path / "data" / "features" / "temporal_modeling_dataset_v1"
        )
        self.products_path = root_path / "data" / "raw" / "products.csv"

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                "Run `uv run dvc pull` before starting the API."
            )

        if not self.products_path.exists():
            raise FileNotFoundError(
                f"Products file not found at {self.products_path}. "
                "Run `uv run dvc pull` before starting the API."
            )

        self.dataset = ds.dataset(self.dataset_path, format="parquet")
        self.products = pd.read_csv(self.products_path)[["product_id", "product_name"]]

    def get_user(self, user_id: int) -> dict:
        """Return user information."""

        table = self.dataset.to_table(
            filter=ds.field("user_id") == user_id,
        )

        user = table.to_pandas()

        if user.empty:
            return {
                "message": "User not found",
            }

        if "product_name" not in user.columns:
            user = user.merge(
                self.products,
                on="product_id",
                how="left",
            )

        first = user.iloc[0]

        products = self._build_products(user)

        return {
            "user_id": int(first["user_id"]),
            "total_orders": self._safe_int(first, "total_orders"),
            "avg_days_between_orders": self._safe_float(
                first,
                "avg_days_between_orders",
            ),
            "avg_order_hour": self._safe_float(first, "avg_order_hour"),
            "favorite_order_day": str(first.get("favorite_order_day", "")),
            "products": products,
        }

    def _build_products(self, user: pd.DataFrame) -> list[dict[str, Any]]:
        """Build product list for a user."""

        required_columns = [
            "product_id",
            "product_name",
            "user_product_purchase_count",
            "user_product_reorder_rate",
        ]

        for column in required_columns:
            if column not in user.columns:
                user[column] = 0 if column != "product_name" else "Unknown"

        products = (
            user[required_columns]
            .drop_duplicates("product_id")
            .sort_values(
                "user_product_purchase_count",
                ascending=False,
            )
        )

        return [
            {
                "product_id": int(row["product_id"]),
                "product_name": str(row["product_name"]),
                "user_product_purchase_count": int(row["user_product_purchase_count"]),
                "user_product_reorder_rate": float(row["user_product_reorder_rate"]),
            }
            for _, row in products.iterrows()
        ]

    @staticmethod
    def _safe_int(row: pd.Series, column: str) -> int:
        """Return an integer value or zero if the column is missing."""

        if column not in row or pd.isna(row[column]):
            return 0
        return int(row[column])

    @staticmethod
    def _safe_float(row: pd.Series, column: str) -> float:
        """Return a float value or zero if the column is missing."""

        if column not in row or pd.isna(row[column]):
            return 0.0
        return float(row[column])

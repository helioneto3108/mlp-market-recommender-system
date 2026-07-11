from pathlib import Path

import pandas as pd


class UserService:
    """Service responsible for retrieving user information."""

    def __init__(self) -> None:
        processed_path = Path("data/processed")

        self.dataset = pd.read_parquet(
            processed_path / "train_dataset.parquet"
        )

    def get_user(self, user_id: int) -> dict:
        """Return user information."""

        user = self.dataset[
            self.dataset["user_id"] == user_id
        ]

        if user.empty:
            return {
                "message": "User not found"
            }

        first = user.iloc[0]

        products = (
            user[
                [
                    "product_id",
                    "product_name",
                    "user_product_purchase_count",
                    "user_product_reorder_rate",
                ]
            ]
            .drop_duplicates("product_id")
            .sort_values(
                "user_product_purchase_count",
                ascending=False,
            )
        )

        products = [
            {
                "product_id": int(row["product_id"]),
                "product_name": str(row["product_name"]),
                "user_product_purchase_count": int(row["user_product_purchase_count"]),
                "user_product_reorder_rate": float(row["user_product_reorder_rate"]),
            }
            for _, row in products.iterrows()
        ]

        return {
            "user_id": int(first["user_id"]),
            "total_orders": int(first["total_orders"]),
            "avg_days_between_orders": float(first["avg_days_between_orders"]),
            "avg_order_hour": float(first["avg_order_hour"]),
            "favorite_order_day": str(first["favorite_order_day"]),
            "products": products,
        }
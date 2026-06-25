from pathlib import Path

import numpy as np
import pandas as pd


class RecommendationService:
    """Service responsible for product-to-product recommendations."""

    def __init__(self) -> None:
        processed_path = Path("data/processed")
        raw_path = Path("data/raw")

        self.market_basket_rules = pd.read_parquet(
            processed_path / "market_basket_rules.parquet"
        )

        self.product_similarity_rules = pd.read_parquet(
            processed_path / "product_similarity_rules.parquet"
        )

        self.products = pd.read_csv(
            raw_path / "products.csv"
        )

        if "recommendation_score" not in self.market_basket_rules.columns:
            self.market_basket_rules["recommendation_score"] = (
                self.market_basket_rules["lift"]
                * np.log1p(self.market_basket_rules["cooccurrence_count"])
                * self.market_basket_rules["confidence"]
            )

    def find_product(self, product_name: str) -> tuple[int, str] | None:
        """Find product by exact or partial name."""
        product_name_clean = product_name.lower().strip()

        exact_match = self.products[
            self.products["product_name"]
            .str.lower()
            .str.strip()
            .eq(product_name_clean)
        ]

        if not exact_match.empty:
            product = exact_match.iloc[0]
            return int(product["product_id"]), str(product["product_name"])

        partial_match = self.products[
            self.products["product_name"]
            .str.lower()
            .str.contains(product_name_clean, regex=False)
        ]

        if partial_match.empty:
            return None

        product = partial_match.iloc[0]
        return int(product["product_id"]), str(product["product_name"])

    @staticmethod
    def normalize_score(series: pd.Series) -> pd.Series:
        """Normalize score between 0 and 1."""
        max_value = series.max()

        if pd.isna(max_value) or max_value == 0:
            return series * 0

        return series / max_value

    def recommend(
        self,
        cart_items: list[str],
        top_n: int = 10,
        strategy: str = "hybrid",
        market_weight: float = 0.6,
        similarity_weight: float = 0.4,
    ) -> dict:
        """Recommend products based on cart items."""
        matched_products = []

        for item in cart_items:
            match = self.find_product(item)

            if match is not None:
                matched_products.append(match)

        if not matched_products:
            return {
                "cart_items": cart_items,
                "matched_products": [],
                "recommendations": [],
            }

        cart_product_ids = {
            product_id for product_id, _ in matched_products
        }

        basket_candidates = self.market_basket_rules[
            self.market_basket_rules["product_a_id"].isin(cart_product_ids)
            & ~self.market_basket_rules["product_b_id"].isin(cart_product_ids)
        ].copy()

        if not basket_candidates.empty:
            basket_scores = (
                basket_candidates
                .groupby(["product_b_id", "product_b"], as_index=False)
                .agg(
                    basket_score=("recommendation_score", "max"),
                    basket_confidence=("confidence", "max"),
                    basket_lift=("lift", "max"),
                    basket_cooccurrence=("cooccurrence_count", "sum"),
                )
            )

            basket_scores["basket_score_norm"] = self.normalize_score(
                basket_scores["basket_score"]
            )
        else:
            basket_scores = pd.DataFrame(
                columns=[
                    "product_b_id",
                    "product_b",
                    "basket_score",
                    "basket_confidence",
                    "basket_lift",
                    "basket_cooccurrence",
                    "basket_score_norm",
                ]
            )

        similarity_candidates = self.product_similarity_rules[
            self.product_similarity_rules["product_a_id"].isin(cart_product_ids)
            & ~self.product_similarity_rules["product_b_id"].isin(cart_product_ids)
        ].copy()

        if not similarity_candidates.empty:
            similarity_scores = (
                similarity_candidates
                .groupby(["product_b_id", "product_b"], as_index=False)
                .agg(
                    similarity_score=("cosine_similarity", "max")
                )
            )

            similarity_scores["similarity_score_norm"] = self.normalize_score(
                similarity_scores["similarity_score"]
            )
        else:
            similarity_scores = pd.DataFrame(
                columns=[
                    "product_b_id",
                    "product_b",
                    "similarity_score",
                    "similarity_score_norm",
                ]
            )

        recommendations = basket_scores.merge(
            similarity_scores,
            on="product_b_id",
            how="outer",
            suffixes=("_basket", "_similarity"),
        )

        recommendations["recommended_product"] = (
            recommendations["product_b_basket"]
            .combine_first(recommendations["product_b_similarity"])
        )

        recommendations = recommendations.fillna(
            {
                "basket_score_norm": 0,
                "similarity_score_norm": 0,
                "basket_score": 0,
                "similarity_score": 0,
                "basket_confidence": 0,
                "basket_lift": 0,
                "basket_cooccurrence": 0,
            }
        )

        if strategy == "market_basket":
            recommendations["final_score"] = recommendations["basket_score_norm"]

        elif strategy == "similarity":
            recommendations["final_score"] = recommendations["similarity_score_norm"]

        elif strategy == "hybrid":
            recommendations["final_score"] = (
                market_weight * recommendations["basket_score_norm"]
                + similarity_weight * recommendations["similarity_score_norm"]
            )

        else:
            raise ValueError(
                "Strategy must be 'market_basket', 'similarity' or 'hybrid'."
            )

        recommendations = recommendations[
            recommendations["final_score"] > 0
        ].copy()

        if recommendations.empty:
            return {
                "cart_items": cart_items,
                "matched_products": matched_products,
                "recommendations": [],
            }

        recommendations["recommendation_percent"] = (
            self.normalize_score(recommendations["final_score"]) * 100
        ).round(0).astype(int)

        recommendations = (
            recommendations
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )

        response = recommendations[
            [
                "recommended_product",
                "recommendation_percent",
                "final_score",
                "basket_score_norm",
                "similarity_score_norm",
            ]
        ].to_dict(orient="records")

        return {
            "cart_items": cart_items,
            "matched_products": [
                {"product_id": product_id, "product_name": product_name}
                for product_id, product_name in matched_products
            ],
            "recommendations": response,
        }
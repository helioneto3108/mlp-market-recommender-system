from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.recommendation_service import RecommendationService


app = FastAPI(
    title="Market Recommender API",
    description="Hybrid product-to-product recommendation API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommendation_service = RecommendationService()


class RecommendationRequest(BaseModel):
    """Request body for recommendations."""

    cart_items: list[str]
    top_n: int = 10
    strategy: str = "hybrid"


@app.get("/")
def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Market Recommender API is running",
    }


@app.post("/recommend")
def recommend_products(request: RecommendationRequest) -> dict:
    """Recommend products based on cart items."""
    return recommendation_service.recommend(
        cart_items=request.cart_items,
        top_n=request.top_n,
        strategy=request.strategy,
    )
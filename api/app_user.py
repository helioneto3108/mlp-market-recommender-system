from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.user_service import UserService

app = FastAPI(
    title="Market User API",
    description="API for retrieving user information",
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

user_service = UserService()


@app.get("/")
def health_check() -> dict:
    return {
        "status": "ok",
        "message": "Market User API is running",
    }


@app.get("/users/{user_id}")
def get_user(user_id: int) -> dict:
    return user_service.get_user(user_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app_user:app",
        host="127.0.0.1",
        port=8010,
        reload=True,
    )
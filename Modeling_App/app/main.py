# fast api
import uvicorn
from fastapi import FastAPI
import sys
from app.api import api


def app():
    app = FastAPI()
    app.include_router(api)
    return app


if __name__ == "__main__":
    uvicorn.run(app(), log_level="info")
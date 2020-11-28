# fast api
import uvicorn
from fastapi import FastAPI
import sys
from app.api import api



app = FastAPI()
app.include_router(api)



if __name__ == "__main__":
    uvicorn.run(app(), log_level="info")
from fastapi import FastAPI

from app.core.config import settings
from app.router import router

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(router)

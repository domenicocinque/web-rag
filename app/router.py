import logging
from fastapi.routing import APIRouter

from app.core.config import settings
from app.services.search_agent import SearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=settings.API_V1_STR,
)

search_agent = SearchAgent(api_key=settings.OPENAI_API_KEY.get_secret_value())


@router.get("/search")
async def web_rag(query: str):
    result = search_agent.run(query)
    return {"reply": result}

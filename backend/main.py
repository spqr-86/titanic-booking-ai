import os
import logging

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from services.query_router import titanic_agent_graph 
from services import rag_service

# Railway production settings
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
# Настройка логирования
LOG_LEVEL = logging.DEBUG if ENVIRONMENT == "development" else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"⚙️ Запуск в режиме: {ENVIRONMENT.upper()}")

# Отключаем телеметрию ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_CLIENT_TELEMETRY"] = "False"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Titanic Booking AI",
    description="AI-powered booking assistant for RMS Titanic maiden voyage",
    version="1.1.0",
)

# CORS настройка для frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://*.railway.app",  # Railway domains
        "https://unique-adventure-production.up.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Инициализация OpenAI клиента
openai_api_key = os.getenv("OPENAI_API_KEY")

USE_LANGGRAPH_ROUTER = os.getenv("USE_LANGGRAPH_ROUTER", "True").lower() == "true"
if USE_LANGGRAPH_ROUTER:
    logger.info("🚀 Маршрутизатор на LangGraph АКТИВИРОВАН.")


# Модели данных
class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "success"


class ChatResponseWithSources(BaseModel):
    response: str
    session_id: str
    status: str = "success"
    sources: List[dict] = []
    has_rag: bool = True

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Titanic Booking AI Backend",
        "status": "running",
        "version": "1.1.0",
        "api_docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервиса"""
    logger.info("🔍 Проверка состояния сервиса...")
    return {
        "status": "healthy",
        "langgraph_router_enabled": USE_LANGGRAPH_ROUTER,
        "sessions_active": len(rag_service.store) # Проверяем хранилище в сервисе
    }


@app.post("/api/chat", response_model=ChatResponseWithSources)
async def chat(chat_message: ChatMessage):
    """
    Основной эндпоинт для чата. Использует Feature Flag для выбора системы.
    """
    session_id = chat_message.session_id
    user_message = chat_message.message.strip()
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")

    logger.info(f"💬 Запрос от сессии {session_id}: {user_message}")

    try:
        # Получаем объект истории чата из нашего сервиса
        chat_history_for_session = rag_service.get_session_history(session_id)
        
        answer = ""
        sources = []

        if USE_LANGGRAPH_ROUTER:
            # --- НОВЫЙ ПУТЬ: LANGGRAPH ---
            logger.info("🧠 Выбран путь: LangGraph Router")
            
            # Начальное состояние для графа
            initial_state = {
                "session_id": session_id,
                "question": user_message,
                "chat_history": chat_history_for_session.messages,
            }
            
            # Асинхронно запускаем граф
            final_state = await titanic_agent_graph.ainvoke(initial_state)
            
            answer = final_state.get("generation", "Произошла системная ошибка при обработке вашего запроса.")
            # Источники можно будет также добавить в AgentState и извлекать отсюда
            sources = final_state.get("context", []) 

        else:
            # --- СТАРЫЙ ПУТЬ: ПРЯМОЙ RAG (НАШ FALLBACK) ---
            logger.info("🧠 Выбран путь: Legacy RAG")
            
            rag_chain = rag_service.get_rag_chain()
            response = await rag_chain.ainvoke({
                "question": user_message,
                "chat_history": chat_history_for_session.messages,
            })
            answer = response.get("answer", "Не удалось получить ответ из архивов.")
            sources = response.get("context", [])

        # Обновляем историю в сервисе ПОСЛЕ получения ответа
        chat_history_for_session.add_user_message(user_message)
        chat_history_for_session.add_ai_message(answer)

        sources_to_return = sources if isinstance(sources, list) else []
        
        return ChatResponseWithSources(
            response=answer, 
            session_id=session_id, 
            sources=sources_to_return
        )

    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка в эндпоинте /api/chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка сервера: {e}")


@app.delete("/api/chat/session/{session_id}")
async def clear_session(session_id: str):
    """Очистка истории чата для сессии"""
    try:
        if session_id in rag_service.store:
            rag_service.clear_memory(session_id)
            logger.info(f"История сессии {session_id} очищена.")
            return {"message": f"История сессии {session_id} очищена"}
        else:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
    except Exception as e:
        logger.error(f"Ошибка очистки памяти сессии: {e}")
        raise HTTPException(status_code=500, detail="Ошибка очистки памяти")


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=(ENVIRONMENT != "production"),
        log_level="info" if ENVIRONMENT == "production" else "debug",
    )

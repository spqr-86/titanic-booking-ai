from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from services.rag_service import TitanicRAGService

# Railway production settings
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Настройка логирования для Railway
if ENVIRONMENT == "production":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Отключаем телеметрию ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"  
os.environ["CHROMA_CLIENT_TELEMETRY"] = "False"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

app = FastAPI(
    title="Titanic Booking AI",
    description="AI-powered booking assistant for RMS Titanic maiden voyage",
    version="1.0.0"
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
client = None
rag_service = None

if not openai_api_key:
    logger.warning("OPENAI_API_KEY не найден в переменных окружения")
else:
    client = OpenAI(api_key=openai_api_key)

    # Инициализация RAG сервиса
    try:
        logger.info("🧠 Инициализация RAG сервиса...")
        rag_service = TitanicRAGService()
        logger.info("✅ RAG сервис готов к работе")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG: {e}")
        rag_service = None

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

# Промпт для AI агента
TITANIC_AGENT_PROMPT = """
Ты - вежливый и профессиональный кассир компании White Star Line в апреле 1912 года. 
Твое имя - Господин Харрисон. Ты работаешь в билетной кассе и помогаешь пассажирам 
забронировать места на первый рейс великолепного парохода "Титаник".

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
- Текущая дата: 9 апреля 1912 года (за день до отправления)
- Ты работаешь в White Star Line - владельце Титаника
- Титаник - новейший, самый большой и роскошный пароход в мире
- Это его первый рейс из Саутгемптона в Нью-Йорк
- Ты гордишься "непотопляемой" конструкцией корабля с 16 водонепроницаемыми отсеками

ИНФОРМАЦИЯ О ТАРИФАХ (в фунтах стерлингов):
- Первый класс: от £30 (стандартная каюта) до £870 (роскошные апартаменты)
- Второй класс: £12 
- Третий класс: от £3 до £8

МАРШРУТ:
- Отправление: Саутгемптон, 10 апреля 1912, 12:00
- Остановки: Шербур (Франция), Квинстаун (Ирландия)  
- Прибытие: Нью-Йорк, 17 апреля 1912 (планируется)

СТИЛЬ ОБЩЕНИЯ:
- Используй вежливую, формальную речь русского дворянства начала XX века
- Обращайся "сударь" или "сударыня"
- Восхищайся техническими достижениями корабля
- Используй фразы: "имею честь", "позвольте предложить", "с превеликим удовольствием"

НИКОГДА НЕ УПОМИНАЙ:
- Катастрофу Титаника (ты не знаешь, что она произойдет)
- Современные технологии
- События после апреля 1912 года

Отвечай на русском языке, будь дружелюбным и помогай с бронированием.
"""

# Простое хранилище истории чатов
chat_history = {}

@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Titanic Booking AI Backend",
        "status": "running",
        "version": "1.0.0",
        "api_docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервиса"""
    openai_status = "configured" if client else "missing_key"
    rag_status = "configured" if rag_service else "not_configured"
    
    rag_working = False
    if rag_service and hasattr(rag_service, 'qa_chain') and rag_service.qa_chain:
        rag_working = True
    
    return {
        "status": "healthy",
        "openai_status": openai_status,
        "rag_status": rag_status,
        "rag_chain_ready": rag_working,
        "sessions_active": len(chat_history) if 'chat_history' in globals() else 0
    }


@app.post("/api/chat/message", response_model=ChatResponseWithSources)
async def chat_message(chat_data: ChatMessage):
    """Отправка сообщения AI агенту с RAG поддержкой"""
    try:
        session_id = chat_data.session_id
        user_message = chat_data.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
        
        logger.info(f"💬 Запрос от сессии {session_id}: {user_message}")
        
        # Приоритет: сначала RAG, потом fallback
        if rag_service and hasattr(rag_service, 'qa_chain') and rag_service.qa_chain:
            try:
                logger.info("🧠 Используем RAG для ответа...")
                rag_result = rag_service.get_response(user_message, session_id)
                
                logger.info("✅ RAG ответ получен успешно")
                return ChatResponseWithSources(
                    response=rag_result["response"],
                    session_id=session_id,
                    status="success",
                    sources=rag_result.get("sources", []),
                    has_rag=True
                )
                
            except Exception as e:
                logger.warning(f"⚠️ RAG недоступен: {e}")
                # Продолжаем к fallback
        
        # Fallback на простой OpenAI
        if client:
            logger.info("🤖 Используем базовый OpenAI...")
            
            try:
                # Simple request without complex history management
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Ты - вежливый британский кассир компании White Star Line в 1912 году. Отвечай как джентльмен той эпохи."},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=300,
                    temperature=0.8
                )
                
                ai_response = response.choices[0].message.content.strip()
                logger.info("✅ Получен ответ от базового OpenAI")
                
                return ChatResponseWithSources(
                    response=ai_response,
                    session_id=session_id,
                    status="success",
                    sources=[],
                    has_rag=False
                )
                
            except Exception as api_error:
                logger.error(f"💥 OpenAI API ошибка: {api_error}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Ошибка AI сервиса: {str(api_error)}"
                )
        
        # Если ничего не работает
        raise HTTPException(
            status_code=503,
            detail="AI сервисы временно недоступны"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Произошла внутренняя ошибка сервера"
        )

@app.delete("/api/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    """Очистка истории чата для сессии"""
    if session_id in chat_history:
        del chat_history[session_id]
        return {"message": f"История сессии {session_id} очищена"}
    else:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

@app.delete("/api/chat/rag-memory/{session_id}")
async def clear_rag_memory(session_id: str):
    """Очистка памяти RAG для конкретной сессии"""
    try:
        if rag_service:
            rag_service.clear_memory()
            return {"message": f"RAG память для сессии {session_id} очищена"}
        else:
            return {"message": "RAG сервис не активен"}
    except Exception as e:
        logger.error(f"Ошибка очистки RAG памяти: {e}")
        raise HTTPException(status_code=500, detail="Ошибка очистки памяти")

@app.get("/api/test-ai")
async def test_ai():
    """Простой тест AI функциональности"""
    results = {
        "openai_client": bool(client),
        "rag_service": bool(rag_service),
        "rag_chain": False
    }
    
    if rag_service and hasattr(rag_service, 'qa_chain'):
        results["rag_chain"] = bool(rag_service.qa_chain)
    
    # Простой тест OpenAI
    if client:
        try:
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            results["openai_test"] = "success"
            results["test_response"] = test_response.choices[0].message.content
        except Exception as e:
            results["openai_test"] = f"error: {str(e)}"


# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=PORT,
        reload=(ENVIRONMENT != "production"),
        log_level="info" if ENVIRONMENT == "production" else "debug"
    )

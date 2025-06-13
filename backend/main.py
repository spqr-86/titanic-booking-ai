from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

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
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Инициализация OpenAI клиента
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.warning("OPENAI_API_KEY не найден в переменных окружения")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

# Модели данных
class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "success"

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
    
    return {
        "status": "healthy",
        "openai_status": openai_status,
        "sessions_active": len(chat_history)
    }

@app.post("/api/chat/message", response_model=ChatResponse)
async def chat_message(chat_data: ChatMessage):
    """Отправка сообщения AI агенту"""
    try:
        session_id = chat_data.session_id
        user_message = chat_data.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
        
        # Проверяем OpenAI клиент
        if not client:
            raise HTTPException(
                status_code=503, 
                detail="OpenAI API ключ не настроен. Обратитесь к администратору."
            )
        
        # Получаем или создаем историю чата
        if session_id not in chat_history:
            chat_history[session_id] = [
                {"role": "system", "content": TITANIC_AGENT_PROMPT}
            ]
            logger.info(f"Создана новая сессия: {session_id}")
        
        # Добавляем сообщение пользователя
        chat_history[session_id].append({
            "role": "user", 
            "content": user_message
        })
        
        # Ограничиваем историю (последние 20 сообщений + system prompt)
        if len(chat_history[session_id]) > 21:
            chat_history[session_id] = [chat_history[session_id][0]] + chat_history[session_id][-20:]
        
        logger.info(f"Отправка запроса к OpenAI для сессии {session_id}")
        
        # Вызов OpenAI API (новый синтаксис)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history[session_id],
            max_tokens=400,
            temperature=0.8,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Добавляем ответ AI в историю
        chat_history[session_id].append({
            "role": "assistant",
            "content": ai_response
        })
        
        logger.info(f"Успешный ответ для сессии {session_id}")
        
        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            status="success"
        )
        
    except Exception as e:
        # Универсальная обработка ошибок для новой версии OpenAI
        error_message = str(e)
        logger.error(f"Ошибка при обработке запроса: {error_message}")
        
        # Определяем тип ошибки по содержимому сообщения
        if "authentication" in error_message.lower() or "unauthorized" in error_message.lower():
            raise HTTPException(
                status_code=401, 
                detail="Неверный OpenAI API ключ"
            )
        elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
            raise HTTPException(
                status_code=429, 
                detail="Слишком много запросов. Попробуйте позже."
            )
        elif "openai" in error_message.lower():
            raise HTTPException(
                status_code=502, 
                detail="Временная проблема с AI сервисом. Попробуйте позже."
            )
        else:
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

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )

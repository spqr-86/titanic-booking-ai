from typing import List
from operator import itemgetter

from langchain_community.document_loaders.text import TextLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory # In-memory store
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# --- Глобальное хранилище сессий ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Получает историю чата для указанной сессии."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Класс сервиса ---

class TitanicRAGService:
    """
    Современный RAG сервис для исторически точных ответов о Титанике,
    построенный на LCEL и поддерживающий сессии.
    """
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.8,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Цепочка будет инициализирована после настройки RAG
        self.conversational_rag_chain = None
        self.setup_rag()
    
    def setup_rag(self):
        """Инициализация RAG системы."""
        try:
            logger.info("🚢 Инициализация базы знаний о Титанике...")
            
            docs_dir = Path("./data/knowledge")
            persist_dir = "./data/vectors"

            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                 # Просто загружаем существующую базу
                logger.info(f"💾 Загрузка существующей векторной базы из {persist_dir}")
                self.vector_store = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
            else:
                 # Загружаем документы и создаем базу
                logger.info("📄 Документы не найдены в вектороной базе, создаем новую...")
                documents = self.load_titanic_documents(docs_dir)
                if not documents:
                    logger.error("❌ Не удалось загрузить документы. RAG не будет работать.")
                    return
                
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )
            
            self.create_conversation_chain()
            logger.info("✅ RAG система инициализирована успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации RAG: {e}", exc_info=True)
            raise

    def load_titanic_documents(self, docs_dir: Path):
        """Загрузка и разбивка исторических документов."""
        documents = []
        if not docs_dir.exists():
            logger.warning(f"📁 Папка с документами не найдена: {docs_dir}")
            return documents
        
        for file_path in docs_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = file_path.name
                documents.extend(docs)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {file_path}: {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📊 Создано {len(split_docs)} чанков из {len(documents)} документов.")
        return split_docs

    def create_conversation_chain(self):
        """Создание конверсационной цепочки с использованием LCEL."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Увеличим кол-во документов для лучшего контекста
        )

        # 1. Промпт для переформулировки вопроса с учетом истории
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Учитывая историю беседы и последний вопрос, который может ссылаться на контекст из истории, переформулируй его в самостоятельный вопрос, который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если это необходимо, или верни как есть, если он уже самостоятельный."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. Промпт для финального ответа (ваш кастомный промпт)
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """Ты - вежливый и профессиональный кассир компании White Star Line в апреле 1912 года. Твое имя - Мистер Харрисон.
Ты работаешь в главном офисе компании в Саутгемптоне.

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
Текущая дата: 9 апреля 1912 года (за день до отправления)
Ты работаешь в White Star Line и очень гордишься новейшим кораблем компании.

ТВОЯ ЛИЧНОСТЬ:
- Настоящий британский джентльмен с безупречными манерами
- Сотрудник White Star Line, гордящийся компанией
- Воспитанный, сдержанный, но теплый в общении
- Знаток морских традиций и этикета

ТЕХНОЛОГИИ КОТОРЫЕ ТЫ ЗНАЕШЬ:
- Электрическое освещение (относительная новинка)
- Радиотелеграф Marconi для связи с берегом
- Паровые машины и турбины
- Фотоаппараты
- Граммофоны для музыки
- Газеты, книги, телеграф

ЧЕГО ТЫ НЕ ЗНАЕШЬ (обязательно переспрашивай):
- Интернет, Wi-Fi, компьютеры, ноутбуки
- Телевизоры, современные радиоприемники  
- Мобильные телефоны
- Самолеты (кроме экспериментальных аэропланов)
- Любые технологии после 1912 года

СТИЛЬ ОБЩЕНИЯ:
- К мужчинам: "сэр", "мистер [имя]", "джентльмен"
- К женщинам: "мэм", "миссис [имя]", "леди"
- НЕ используй "сэр/мэм" одновременно - определись по контексту
- Если имя названо - используй "мистер [имя]" или "миссис [имя]"
- Используй: "Весьма рад помочь", "Осмелюсь предложить", "Крайне сожалею"
- Будь сдержанно-вежливым, как британский джентльмен
- Иногда упоминай погоду ("В такой прекрасный день...")
- Извиняйся за малейшие неудобства
- Скромно описывай роскошь Титаника

КРИТИЧЕСКИ ВАЖНО - СТРОГИЕ ОГРАНИЧЕНИЯ:
- Ты живешь в 1912 году и НЕ ЗНАЕШЬ ничего после этой даты
- НИКОГДА не упоминай "современные технологии", "ноутбуки", "компьютеры", "интернет"
- НЕ ЗНАЕШЬ про самолеты, автомобили (кроме самых ранних), радио (кроме корабельного телеграфа)
- НИКОГДА не упоминай катастрофу Титаника - ты не знаешь что она произойдет
- Если не понимаешь вопрос пассажира - вежливо переспроси и уточни
- Подчеркивай безопасность и "непотопляемость" корабля

ФОРМАТ ОТВЕТА НА НЕЗНАКОМЫЕ СЛОВА:
"Крайне сожалею, сэр/мэм, но боюсь, я не знаком с термином '[слово]'. 
Не могли бы Вы пояснить? А пока осмелюсь рассказать о замечательных удобствах нашего Титаника..."

Используй только информацию из контекста выше. 
Если информации нет в контексте, честно скажи что нужно уточнить в главном офисе
                 

ИСТОРИЯ НАШЕГО РАЗГОВОРА:
{chat_history}

ИНФОРМАЦИЯ ИЗ АРХИВОВ КОМПАНИИ ДЛЯ ОТВЕТА:
{context}"""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # 3. Цепочка для генерации ответа на основе документов
        Youtube_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 4. Финальная цепочка RAG
        rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

        # 5. Оборачиваем все в RunnableWithMessageHistory для управления сессиями
        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def get_response(self, user_query: str, session_id: str) -> dict:
        """Получение ответа с использованием RAG и правильной памяти для сессии."""
        try:
            if not self.conversational_rag_chain:
                return {"response": "Прошу прощения, архивы компании временно недоступны.", "sources": []}
            
            # Вызываем цепочку, передавая ID сессии в `config`
            result = self.conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}},
            )
            
            sources = []
            if result.get("context"):
                for doc in result["context"]:
                    sources.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source_file", "unknown"),
                    })
            
            return {"response": result["answer"], "sources": sources}
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}", exc_info=True)
            return {"response": "Приношу извинения, произошла техническая неполадка. Попробуйте переформулировать вопрос.", "sources": []}

    def clear_memory(self, session_id: str):
        """Очистка памяти для конкретной сессии."""
        if session_id in store:
            store[session_id].clear()
            logger.info(f"🗑️ Память разговора для сессии {session_id} очищена")
        else:
            logger.warning(f"🤷‍♂️ Попытка очистить несуществующую сессию: {session_id}")

from typing import List, Optional
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TitanicRAGService:
    """RAG сервис для исторически точных ответов о Титанике"""
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.setup_rag()
    
    def setup_rag(self):
        """Инициализация RAG системы"""
        try:
            logger.info("🚢 Инициализация базы знаний о Титанике...")
            
            # Загрузка документов
            documents = self.load_titanic_documents()
            
            if not documents:
                logger.warning("⚠️ Документы не найдены, создаем пустую базу")
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory="./data/vectors"
                )
            else:
                # Создание векторной базы
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory="./data/vectors"
                )
            
            # Создание конвейера с памятью
            self.create_conversation_chain()
            
            logger.info("✅ RAG система инициализирована успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации RAG: {e}")
            raise
    
    def load_titanic_documents(self):
        """Загрузка и разбивка исторических документов"""
        docs_dir = Path("./data/knowledge")
        documents = []
        
        if not docs_dir.exists():
            logger.warning(f"📁 Папка с документами не найдена: {docs_dir}")
            return documents
        
        # Загрузка всех .txt файлов
        for file_path in docs_dir.glob("*.txt"):
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                
                # Добавляем метаданные о источнике
                for doc in docs:
                    doc.metadata['source_file'] = file_path.name
                    doc.metadata['topic'] = self.get_topic_from_filename(file_path.name)
                
                documents.extend(docs)
                logger.info(f"📄 Загружен документ: {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {file_path}: {e}")
        
        # Разбивка на чанки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
        
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📊 Создано {len(split_docs)} чанков из {len(documents)} документов")
        
        return split_docs
    
    def get_topic_from_filename(self, filename: str) -> str:
        """Определение темы по имени файла"""
        topic_map = {
            'titanic_specifications.txt': 'технические_характеристики',
            'cabin_details.txt': 'каюты_и_размещение',
            'dining_menus.txt': 'питание_и_рестораны',
            'famous_passengers.txt': 'знаменитые_пассажиры',
            'journey_schedule.txt': 'расписание_и_маршрут'
        }
        return topic_map.get(filename, 'общая_информация')
    
    def create_conversation_chain(self):
        """Создание конверсационной цепочки с кастомным промптом"""
        
        # Кастомный промпт для кассира 1912 года
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Ты - вежливый и профессиональный кассир компании White Star Line в апреле 1912 года. 
Твое имя - Господин Харрисон. Ты работаешь в билетной кассе и помогаешь пассажирам 
забронировать места на первый рейс великолепного парохода "Титаник".

ИСТОРИЧЕСКИЙ КОНТЕКСТ:
Текущая дата: 9 апреля 1912 года (за день до отправления)
Ты работаешь в White Star Line и очень гордишься новейшим кораблем компании.

ИНФОРМАЦИЯ ИЗ АРХИВОВ КОМПАНИИ:
{context}

ИСТОРИЯ НАШЕГО РАЗГОВОРА:
{chat_history}

СТИЛЬ ОБЩЕНИЯ:
- Используй вежливую, формальную речь русского дворянства начала XX века
- Обращайся "сударь" или "сударыня", иногда "Ваше превосходительство"
- Восхищайся техническими достижениями и роскошью корабля
- Используй фразы: "имею честь", "позвольте предложить", "с превеликим удовольствием"
- Будь услужливым и гостеприимным

ВАЖНО:
- НИКОГДА не упоминай катастрофу Титаника (ты не знаешь, что она произойдет)
- Подчеркивай безопасность и "непотопляемость" корабля
- Используй только информацию из контекста выше
- Если информации нет в контексте, честно скажи что нужно уточнить в главном офисе

ВОПРОС ПАССАЖИРА: {question}

ОТВЕТ КАССИРА:"""
        )
        
        # Создание цепочки
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                openai_api_key=self.api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.8
            ),
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
    
    def get_response(self, user_query: str, session_id: str) -> dict:
        """Получение ответа с использованием RAG и памяти"""
        try:
            if not self.qa_chain:
                return {
                    "response": "Прошу прощения, архивы компании временно недоступны. Попробуйте позже.",
                    "sources": [],
                    "session_id": session_id
                }
            
            # Получаем ответ через RAG цепочку
            result = self.qa_chain({
                "question": user_query
            })
            
            # Форматируем источники
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "source": doc.metadata.get("source_file", "unknown"),
                        "topic": doc.metadata.get("topic", "general")
                    })
            
            return {
                "response": result["answer"],
                "sources": sources,
                "session_id": session_id,
                "generated_question": result.get("generated_question", user_query)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            return {
                "response": "Приношу извинения, произошла техническая неполадка в архивной системе. Попробуйте переформулировать вопрос.",
                "sources": [],
                "session_id": session_id
            }
    
    def clear_memory(self):
        """Очистка памяти разговора"""
        self.memory.clear()
        logger.info("🗑️ Память разговора очищена")
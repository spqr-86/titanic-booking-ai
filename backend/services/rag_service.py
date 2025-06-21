from langchain_community.document_loaders.text import TextLoader
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory  # In-memory store
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from .prompt_loader import load_prompt

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
            model="gpt-3.5-turbo", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY")
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
                self.vector_store = Chroma(
                    persist_directory=persist_dir, embedding_function=self.embeddings
                )
            else:
                # Загружаем документы и создаем базу
                logger.info(
                    "📄 Документы не найдены в вектороной базе, создаем новую..."
                )
                documents = self.load_titanic_documents(docs_dir)
                if not documents:
                    logger.error(
                        "❌ Не удалось загрузить документы. RAG не будет работать."
                    )
                    return

                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_dir,
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
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                documents.extend(docs)
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {file_path}: {e}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450, chunk_overlap=50
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(
            f"📊 Создано {len(split_docs)} чанков из {len(documents)} документов."
        )
        return split_docs

    def create_conversation_chain(self):
        """
        Создание конверсационной цепочки, которая динамически загружает базовый промпт.
        """
        logger.info("⚙️ Начинаем создание conversational chain (модульная версия)...")
        try:
            base_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "fetch_k": 50},
            )
            # Используем быструю и дешевую модель для задачи фильтрации
            compressor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            compressor = LLMChainExtractor.from_llm(compressor_llm)

            # Создаем компрессионный ретривер
            # Он оборачивает наш базовый ретривер и применяет к его результатам компрессор
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )

            # --- Шаг 1: Создаем history-aware retriever (без изменений) ---
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Учитывая историю беседы и последний вопрос, который может ссылаться на контекст из истории, переформулируй его в самостоятельный вопрос, который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если это необходимо, или верни как есть, если он уже самостоятельный."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            history_aware_retriever = create_history_aware_retriever(
                self.llm, compression_retriever, contextualize_q_prompt
            )

            # --- Шаг 2: Динамически собираем промпт для ответа ---

            # 2.1. Загружаем базовую личность из файла (единый источник правды)
            base_persona_prompt = load_prompt("base_persona_prompt.txt")

            # 2.2. Создаем шаблон, специфичный ТОЛЬКО для RAG-задач
            rag_context_template = """

--- ИНФОРМАЦИЯ ИЗ АРХИВОВ КОМПАНИИ ДЛЯ ОТВЕТА ---
{context}"""
            # Примечание: history уже будет вставлена через MessagesPlaceholder

            # 2.3. Собираем финальный системный промпт
            final_system_prompt = f"{base_persona_prompt}{rag_context_template}"

            # 2.4. Создаем ChatPromptTemplate на основе собранного промпта
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", final_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            # --- Шаг 3: Собираем финальную цепочку (без изменений) ---
            
            Youtube_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

            # --- Шаг 4: Оборачиваем в историю (без изменений) ---

            self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            logger.info("✅ Conversational chain (модульная) успешно создана.")
        except Exception as e:
            logger.critical(
                f"💥 КРИТИЧЕСКАЯ ОШИБКА при создании conversational chain: {e}",
                exc_info=True,
            )
            self.conversational_rag_chain = None

    def get_response(self, user_query: str, session_id: str) -> dict:
        """Получение ответа с использованием RAG и правильной памяти для сессии."""
        try:
            if not self.conversational_rag_chain:
                return {
                    "response": "Прошу прощения, архивы компании временно недоступны.",
                    "sources": [],
                }

            # Вызываем цепочку, передавая ID сессии в `config`
            result = self.conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}},
            )

            sources = []
            if result.get("context"):
                for doc in result["context"]:
                    sources.append(
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get("source_file", "unknown"),
                        }
                    )

            return {"response": result["answer"], "sources": sources}

        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}", exc_info=True)
            return {
                "response": "Приношу извинения, произошла техническая неполадка. Попробуйте переформулировать вопрос.",
                "sources": [],
            }

    def clear_memory(self, session_id: str):
        """Очистка памяти для конкретной сессии."""
        if session_id in store:
            store[session_id].clear()
            logger.info(f"🗑️ Память разговора для сессии {session_id} очищена")
        else:
            logger.warning(f"🤷‍♂️ Попытка очистить несуществующую сессию: {session_id}")


rag_service_instance = TitanicRAGService()
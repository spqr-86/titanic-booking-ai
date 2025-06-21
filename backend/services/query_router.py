import os
from pathlib import Path

from typing import TypedDict, List, Literal
from langchain_core.messages import BaseMessage

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from services.rag_service import rag_service_instance 
from prompts.mini_prompts import MINI_PROMPTS
from .prompt_loader import load_prompt


INTENT_ROUTING_CONFIG = {
    # Интенты, которые ведут к RAG
    "complex": "rag_node",
    "capacity": "rag_node",
    
    # Интенты, которые ведут к простым ответам по шаблону
    "price": "simple_response_node",
    "schedule": "simple_response_node",
    "greeting": "simple_response_node",
    "anachronism": "simple_response_node",
    "small_talk": "simple_response_node",
    "safety": "simple_response_node",
    "off_topic": "simple_response_node",
}

class AgentState(TypedDict):
    """
    Состояние нашего графа. Содержит всю информацию,
    необходимую для принятия решений и генерации ответа.
    """
    session_id: str
    question: str
    chat_history: List[BaseMessage]
    intent: str
    context: List[dict] = []
    generation: str
    error: str


# --- Pydantic модели для классификаторов ---
class AnachronismCheck(BaseModel):
    """Определи, является ли вопрос анахронизмом для 1912 года."""
    is_anachronism: bool = Field(description="True, если вопрос содержит анахронизмы, иначе False.")


class QueryClassifier(BaseModel):
    """Определи тип запроса пользователя."""
    intent_type: Literal[
        "price", 
        "schedule", 
        "greeting", "small_talk", "safety", "capacity", "off_topic", "complex"
    ] = Field(description="Категория, к которой относится запрос пользователя.")


def _is_follow_up(chat_history: List[BaseMessage]) -> bool:
    """Проверяет, является ли сообщение продолжением диалога."""
    # Если в истории больше 1 сообщения (т.е. уже был хотя бы один ответ бота),
    # значит, это продолжение диалога.
    return len(chat_history) > 1


# --- УЗЛЫ ГРАФА ---

def anachronism_guard_node(state: AgentState) -> dict:
    """УЗЕЛ 1: Проверяет на анахронизмы."""
    print("--- 1. УЗЕЛ: ПРОВЕРКА НА АНАХРОНИЗМЫ ('Привратник') ---")
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(AnachronismCheck)
        prompt = ChatPromptTemplate.from_template(load_prompt("anachronism_guard_prompt.txt"))
        chain = prompt | structured_llm
        result = chain.invoke({"question": state["question"]})
        
        if result.is_anachronism:
            print("    > Вердикт: ОБНАРУЖЕН АНАХРОНИЗМ. Отправляем на шаблонный ответ.")
            return {"intent": "anachronism"}
        else:
            print("    > Вердикт: Анахронизмов нет. Отправляем к 'Дворецкому'.")
            return {}
    except Exception as e:
        print(f"    > ОШИБКА 'Привратника': {e}. Пропускаем дальше для безопасности.")
        return {}


def classify_intent_node(state: AgentState) -> dict:
    """УЗЕЛ 2: "Основной классификатор."""
    print("--- 2. УЗЕЛ: КЛАССИФИКАЦИЯ ЗАПРОСА ('Дворецкий') ---")
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm = llm.with_structured_output(QueryClassifier)
        prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("classifier_prompt.txt")),
            ("human", "История переписки:\n{chat_history}\n\nВопрос пользователя: {question}")
        ])
        chain = prompt | structured_llm
        result = chain.invoke({"question": state["question"], "chat_history": state["chat_history"]})
        print(f"    > Вердикт: интент '{result.intent_type}'")
        return {"intent": result.intent_type}
    except Exception as e:
        print(f"    > ОШИБКА 'Дворецкого': {e}. Отправляем в RAG.")
        return {"intent": "complex"}


def simple_response_node(state: AgentState):
    """
    Динамически собирает системный промпт из блоков и генерирует ответ.
    """
    intent_type = state["intent"]
    chat_history = state["chat_history"]
    print(f"--- 2. ГЕНЕРАЦИЯ ПРОСТОГО ОТВЕТА (ТИП: {intent_type}) ---")
    
    try:
        # --- Сборка промпта ---
        # 1. Начинаем с базовой личности
        prompt_parts = [load_prompt("base_persona_prompt.txt")]

        # 2. Если диалог уже начат, добавляем правило "не здороваться"
        if _is_follow_up(chat_history):
            prompt_parts.append(
                "ПРАВИЛО: НЕ здоровайся с пользователем снова, так как диалог уже начат. Сразу переходи к сути ответа."
            )

        # 3. Добавляем конкретную задачу для текущего интента
        task_prompt = MINI_PROMPTS.get(intent_type)
        if task_prompt:
            prompt_parts.append(f"\n--- ТЕКУЩАЯ ЗАДАЧА ---\n{task_prompt}")
        else:
            raise ValueError(f"Нет мини-промпта для типа: {intent_type}")

        # 4. Соединяем все части в финальный промпт
        final_system_prompt = "\n".join(prompt_parts)
        
        # --- Вызов LLM ---
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            ("human", "История переписки:\n{chat_history}\n\nВопрос пользователя: {question}")
        ])
        chain = prompt | llm
        
        response = chain.invoke({
            "question": state["question"],
            "chat_history": chat_history
        })
        
        return {"generation": response.content, "error": None}
    except Exception as e:
        print(f"    > ОШИБКА ГЕНЕРАЦИИ ПРОСТОГО ОТВЕТА: {e}")
        return {"generation": None, "error": "Simple response generation failed"}



def rag_node(state: AgentState) -> dict:
    """
    Вызывает существующий RAG-сервис для ответа на сложный вопрос.
    """
    print("--- 3. ГЕНЕРАЦИЯ СЛОЖНОГО ОТВЕТА (RAG) ---")
    try:
        # Получаем уже инициализированную RAG-цепочку
        rag_result = rag_service_instance.get_response(
            user_query=state["question"], 
            session_id=state["session_id"] # Передаем session_id
        )

        # Наш старый RAG сервис возвращает словарь, извлекаем ответ
        answer = rag_result.get("response", "Не удалось получить ответ из архивов.")
        sources = rag_result.get("sources", [])

        return {"generation": answer, "context": sources, "error": None}
    except Exception as e:
        print(f"    > ОШИБКА RAG: {e}")
        return {"generation": "Прошу прощения, возникла непредвиденная ошибка при поиске в архивах.", "error": "RAG chain failed"}


def decide_next_node(state: AgentState) -> str:
    """
    Условное ребро. Решает, какой узел будет следующим.
    """
    print(f"--- 4. ПРИНЯТИЕ РЕШЕНИЯ (ИНТЕНТ: {state['intent']}) ---")
    # Если в предыдущем узле была ошибка, можно направить на аварийный узел
    if state.get("error"):
        # Пока просто заканчиваем, но можно добавить узел для обработки ошибок
        return END

    intent = state["intent"]
    # Ищем маршрут в нашей конфигурации. 
    # Если по какой-то причине интент неизвестен, по умолчанию отправляем в RAG.
    destination = INTENT_ROUTING_CONFIG.get(intent, "rag_node")
    print(f"    > Направляем на узел: {destination}")

    return destination

# Создаем объект графа
workflow = StateGraph(AgentState)

# Добавляем узлы
workflow.add_node("anachronism_guard", anachronism_guard_node)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("simple_response_node", simple_response_node)
workflow.add_node("rag_node", rag_node)


# Определяем связи
workflow.set_entry_point("anachronism_guard")
def route_after_guard(state: AgentState) -> str:
    if state.get("intent") == "anachronism":
        return "simple_response_node"
    else:
        return "classify_intent"

def route_after_classification(state: AgentState) -> str:
    intent = state.get("intent", "complex")
    return INTENT_ROUTING_CONFIG.get(intent, "rag_node")

workflow.add_conditional_edges("anachronism_guard", route_after_guard)
workflow.add_conditional_edges("classify_intent", route_after_classification)

workflow.add_edge("simple_response_node", END)
workflow.add_edge("rag_node", END)

titanic_agent_graph = workflow.compile()

print("--- ГРАФ УСПЕШНО СКОМПИЛИРОВАН ---")

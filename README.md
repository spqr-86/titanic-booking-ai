# 🚢 RMS Titanic Booking Assistant 1912

[![Live Demo](https://img.shields.io/badge/Live%20Demo-FF6B6B?style=for-the-badge&logo=railway&logoColor=white)](https://unique-adventure-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

## 📋 Описание проекта

AI-агент, который аутентично играет роль кассира компании White Star Line в апреле 1912 года. Бот помогает пассажирам забронировать билеты на первый рейс "Титаника", полностью погружаясь в исторический контекст эпохи.

### 🎯 Ключевые особенности

- **Историческая точность**: 94% (проверено на 60+ исторических фактах)
- **База знаний**: 1,500+ деталей эпохи (цены, маршруты, этикет)
- **Консистентность персонажа**: 0 анахронизмов за 100+ диалогов
- **Защита от спойлеров**: Бот не знает о будущей катастрофе

## 🚀 Демо

Попробуйте поговорить с кассиром Титаника: [Live Demo](https://unique-adventure-production.up.railway.app/)

### Примеры диалогов:

```
Пользователь: "Здравствуйте, я бы хотел купить билет на Титаник"

Мистер Харрисон: "Добрый день, сэр! Весьма рад приветствовать вас в офисе 
White Star Line. Вы сделали превосходный выбор - наш новейший лайнер 
"Титаник" действительно является вершиной кораблестроения! 

Позвольте уточнить, билет какого класса вы желаете приобрести?"
```

## 🏗️ Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI        │────▶│   OpenAI GPT-4  │
│   (React)       │     │   Backend        │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   PostgreSQL     │     │  Historical     │
                        │   (Sessions)     │     │  Knowledge Base │
                        └──────────────────┘     └─────────────────┘
```

## 💻 Установка и запуск

### Требования
- Python 3.9+
- OpenAI API Key
- PostgreSQL (опционально)

### Локальная установка

```bash
# Клонирование репозитория
git clone https://github.com/spqr-86/titanic-booking-ai.git
cd titanic-booking-ai

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r backend/requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Добавьте ваш OPENAI_API_KEY в .env файл

# Запуск backend
cd backend
uvicorn main:app --reload

# В новом терминале - запуск frontend
cd frontend
npm install
npm start
```

Приложение будет доступно по адресу: http://localhost:3000

## 🧠 Техническая реализация

### Промпт-инжиниринг

Ключевой элемент проекта - специально разработанный промпт, который:

1. **Блокирует знания после 1912 года**
   ```python
   "Ты живешь в 1912 году и НЕ ЗНАЕШЬ ничего после этой даты"
   ```

2. **Поддерживает историческую аутентичность**
   - Правильные цены в фунтах стерлингов
   - Точные даты и маршрут
   - Язык и манеры эпохи

3. **Защита от выхода из роли**
   - Обработка попыток "сломать" персонажа
   - Реакция на анахронизмы

### RAG компонент

```python
# Пример структуры знаний
HISTORICAL_FACTS = {
    "prices": {
        "first_class": {"min": 30, "max": 870, "currency": "£"},
        "second_class": {"standard": 12, "currency": "£"},
        "third_class": {"min": 3, "max": 8, "currency": "£"}
    },
    "route": {
        "departure": "Southampton, April 10, 1912, 12:00",
        "stops": ["Cherbourg", "Queenstown"],
        "arrival": "New York, April 17, 1912"
    }
}
```

## 📊 Метрики и тестирование

### Историческая точность
- **Тестовый набор**: 60 исторических фактов
- **Точность**: 94%
- **Методология**: Сравнение ответов с историческими источниками

### Консистентность персонажа
- **Тестовых диалогов**: 100+
- **Анахронизмов обнаружено**: 0
- **Выходов из роли**: 0

## 🎨 Интерфейс

- **Стилизация под 1912 год**: винтажные шрифты и цвета
- **Адаптивный дизайн**: работает на всех устройствах
- **Анимация печатной машинки**: для погружения в эпоху

## 🔧 Конфигурация

### Основные настройки (.env)
```env
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4
MAX_TOKENS=500
TEMPERATURE=0.8
DATABASE_URL=postgresql://user:pass@localhost/titanic_db
```

## 🚢 Roadmap

- [ ] Добавить визуальную карту корабля
- [ ] Расширить базу знаний о пассажирах
- [ ] Мультиязычная поддержка (французский, немецкий)
- [ ] Интеграция с историческими фотографиями
- [ ] Режим "экскурсии" по кораблю

## 🤝 Вклад в проект

Приветствуются любые улучшения! Особенно:
- Дополнительные исторические факты
- Улучшения промптов
- Тесты на историческую точность

См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

## 📚 Источники

- Encyclopedia Titanica
- "Titanic: An Illustrated History" by Don Lynch
- White Star Line архивы
- Исторические газеты 1912 года

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

---

<div align="center">
Сделано с ❤️ для сохранения истории с помощью AI
</div>

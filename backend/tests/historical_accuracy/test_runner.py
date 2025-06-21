import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
import httpx
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalAccuracyTester:
    """Тестирование исторической точности ответов AI агента."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session_id = f"test_session_{datetime.now().timestamp()}"
        self.results = []
        
    async def load_test_facts(self) -> Dict:
        """Загрузка тестовых фактов из JSON файла."""
        facts_path = Path(__file__).parent / "test_facts.json"
        with open(facts_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def test_single_fact(self, fact: Dict) -> Dict:
        """Тестирование одного исторического факта."""
        async with httpx.AsyncClient() as client:
            try:
                # Отправляем вопрос
                response = await client.post(
                    f"{self.api_url}/api/chat/message",
                    json={
                        "message": fact["question"],
                        "session_id": self.session_id
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return {
                        "fact_id": fact["id"],
                        "success": False,
                        "error": f"API returned {response.status_code}"
                    }
                
                data = response.json()
                ai_response = data["response"].lower()
                
                # Проверяем наличие ожидаемых ключевых слов
                expected_found = []
                for keyword in fact["expected_keywords"]:
                    if keyword.lower() in ai_response:
                        expected_found.append(keyword)
                
                # Проверяем отсутствие запрещенных слов
                forbidden_found = []
                for keyword in fact["forbidden_keywords"]:
                    if keyword.lower() in ai_response:
                        forbidden_found.append(keyword)
                
                # Считаем тест пройденным если найдено хотя бы одно ожидаемое слово
                # и не найдено запрещенных
                success = len(expected_found) > 0 and len(forbidden_found) == 0
                
                return {
                    "fact_id": fact["id"],
                    "category": fact["category"],
                    "question": fact["question"],
                    "ai_response": data["response"],
                    "expected_found": expected_found,
                    "forbidden_found": forbidden_found,
                    "success": success,
                    "has_rag": data.get("has_rag", False),
                    "sources_count": len(data.get("sources", []))
                }
                
            except Exception as e:
                logger.error(f"Error testing fact {fact['id']}: {e}")
                return {
                    "fact_id": fact["id"],
                    "success": False,
                    "error": str(e)
                }
    
    async def run_all_tests(self) -> Dict:
        """Запуск всех тестов."""
        test_data = await self.load_test_facts()
        facts = test_data["facts"]
        
        logger.info(f"🧪 Запуск тестирования {len(facts)} исторических фактов...")
        
        # Тестируем каждый факт
        for i, fact in enumerate(facts):
            logger.info(f"Тест {i+1}/{len(facts)}: {fact['id']}")
            result = await self.test_single_fact(fact)
            self.results.append(result)
            
            # Небольшая задержка между запросами
            await asyncio.sleep(1)
        
        # Подсчитываем статистику
        total = len(self.results)
        successful = sum(1 for r in self.results if r.get("success", False))
        accuracy = (successful / total) * 100 if total > 0 else 0
        
        # Статистика по категориям
        category_stats = {}
        for result in self.results:
            if "category" in result:
                cat = result["category"]
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "successful": 0}
                category_stats[cat]["total"] += 1
                if result.get("success", False):
                    category_stats[cat]["successful"] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": total - successful,
            "accuracy_percentage": accuracy,
            "threshold": test_data["test_config"]["min_accuracy_threshold"] * 100,
            "passed": accuracy >= test_data["test_config"]["min_accuracy_threshold"] * 100,
            "category_breakdown": category_stats,
            "detailed_results": self.results
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Сохранение результатов тестирования."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        output_path = Path(__file__).parent / "results" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Результаты сохранены в {output_path}")
        return output_path


async def main():
    """Главная функция для запуска тестов."""
    tester = HistoricalAccuracyTester()
    
    # Проверяем доступность API
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(f"{tester.api_url}/api/health")
            if health.status_code != 200:
                logger.error("❌ API недоступен!")
                return
    except Exception as e:
        logger.error(f"❌ Не удалось подключиться к API: {e}")
        return
    
    # Запускаем тесты
    results = await tester.run_all_tests()
    
    # Сохраняем результаты
    tester.save_results(results)
    
    # Выводим краткую статистику
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ИСТОРИЧЕСКОЙ ТОЧНОСТИ")
    print("="*50)
    print(f"Всего тестов: {results['total_tests']}")
    print(f"Успешных: {results['successful_tests']}")
    print(f"Провальных: {results['failed_tests']}")
    print(f"ТОЧНОСТЬ: {results['accuracy_percentage']:.1f}%")
    print(f"Порог: {results['threshold']:.0f}%")
    print(f"СТАТУС: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    
    print("\n📈 Результаты по категориям:")
    for cat, stats in results['category_breakdown'].items():
        cat_accuracy = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {cat}: {cat_accuracy:.0f}% ({stats['successful']}/{stats['total']})")


if __name__ == "__main__":
    asyncio.run(main())
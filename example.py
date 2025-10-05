"""
Скрипт для запуска pipeline генерации вопросов и ответов из текста.
Использует функции generate_questions и filter_answerable_questions из script.py.
"""

import logging
from question_answer_pipeline import generate_qa_pairs_from_text

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Основная функция для запуска pipeline с заданными параметрами.
    """
    logger.info("Запуск pipeline генерации вопросов и ответов")
    
    # Параметры конфигурации
    config = {
        "input_file": "input.txt",           # Путь к входному файлу
        "output_file": "qa_pairs_output.json", # Путь для сохранения результатов
        "num_questions": 5,                  # Количество вопросов для генерации
        "max_tokens": 64,                    # Максимальное количество токенов на вопрос
        "qa_model": "distilbert-base-uncased-distilled-squad", # Модель для поиска ответов
        "use_smart_processing": True,        # Использовать умную обработку длинных контекстов
        "max_chunk_size": 400,               # Максимальный размер чанка
        "overlap": 50                        # Размер перекрытия между чанками
    }
    
    try:
        # Запуск pipeline
        results = generate_qa_pairs_from_text(**config)
        
        # Вывод результатов
        logger.info("=" * 50)
        logger.info("РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:")
        logger.info(f"Входной файл: {results['input_file']}")
        logger.info(f"Выходной файл: {results['output_file']}")
        logger.info(f"Сгенерировано вопросов: {results['total_questions_generated']}")
        logger.info(f"Отвечаемых вопросов: {results['answerable_questions']}")
        logger.info(f"Успешных ответов: {results['successful_answers']}")
        logger.info(f"Процент успеха: {results['success_rate']:.1%}")
        logger.info(f"Средний score: {results['avg_score']:.3f}")
        logger.info("=" * 50)
        
        print("\n" + "="*60)
        print("PIPELINE УСПЕШНО ЗАВЕРШЕН!")
        print(f"Результаты сохранены в: {config['output_file']}")
        print(f"Сгенерировано вопросов: {results['total_questions_generated']}")
        print(f"Отвечаемых вопросов: {results['answerable_questions']}")
        print(f"Успешных пар вопрос-ответ: {results['successful_answers']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении pipeline: {e}")
        print(f"ОШИБКА: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
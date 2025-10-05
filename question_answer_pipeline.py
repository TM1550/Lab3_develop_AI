import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from transformers import pipeline

from functions_QA2 import smart_qa_with_aggregation, read_txt_file
from script import generate_questions, filter_answerable_questions

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_and_filter_questions(text: str, 
                                 num_questions: int = 5, 
                                 max_tokens: int = 64) -> List[str]:
    """
    Генерирует вопросы и фильтрует те, на которые можно ответить по контексту.
    Объединяет функции generate_questions и filter_answerable_questions из script.py.
    
    Args:
        text: Исходный текст
        num_questions: Количество вопросов для генерации
        max_tokens: Максимальное количество токенов на вопрос
        
    Returns:
        Список отвечаемых вопросов
    """
    logger.info(f"Генерация {num_questions} вопросов...")
    
    # Генерация вопросов
    all_questions = generate_questions(
        text=text,
        num_questions=num_questions,
        max_tokens=max_tokens
    )
    
    logger.info(f"Сгенерировано {len(all_questions)} вопросов")
    
    if not all_questions:
        logger.warning("Не удалось сгенерировать вопросы")
        return []
    
    # Фильтрация отвечаемых вопросов
    logger.info("Фильтрация отвечаемых вопросов...")
    answerable_questions = filter_answerable_questions(all_questions, text)
    
    logger.info(f"После фильтрации осталось {len(answerable_questions)} отвечаемых вопросов")
    
    # Логирование сгенерированных вопросов
    for i, question in enumerate(answerable_questions, 1):
        logger.debug(f"Отвечаемый вопрос {i}: {question}")
    
    return answerable_questions

def find_answers_for_questions(questions: List[str], 
                              context: str, 
                              qa_model: Any,
                              use_smart_processing: bool = True,
                              max_chunk_size: int = 400,
                              overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Находит ответы на вопросы в контексте с использованием QA модели.
    
    Args:
        questions: Список вопросов
        context: Контекст для поиска ответов
        qa_model: QA модель
        use_smart_processing: Использовать ли умную обработку длинных контекстов
        max_chunk_size: Максимальный размер чанка
        overlap: Размер перекрытия между чанками
        
    Returns:
        Список словарей с вопросами и ответами
    """
    if not questions:
        logger.warning("Список вопросов пуст")
        return []
    
    results = []
    context_word_count = len(context.split())
    
    logger.info(f"Поиск ответов для {len(questions)} вопросов (контекст: {context_word_count} слов)")
    
    for i, question in enumerate(tqdm(questions, desc="Поиск ответов")):
        try:
            # Определяем стратегию обработки в зависимости от длины контекста
            needs_chunking = context_word_count > max_chunk_size
            
            if use_smart_processing and needs_chunking:
                # Используем умную обработку для длинных контекстов
                answer_result = smart_qa_with_aggregation(
                    qa_model, question, context, max_chunk_size, overlap
                )
            else:
                # Прямой вызов модели для коротких контекстов
                answer_result = qa_model(question=question, context=context)
            
            # Формируем результат
            result = {
                "question": question,
                "answer": answer_result.get("answer", "").strip(),
                "score": answer_result.get("score", 0.0),
                "start": answer_result.get("start", 0),
                "end": answer_result.get("end", 0)
            }
            
            results.append(result)
            
            # Логирование для отладки
            if i < 5:  # Логируем только первые 5 для отладки
                logger.debug(f"Вопрос {i+1}: '{question}' -> Ответ: '{result['answer']}' (score: {result['score']:.3f})")
                
        except Exception as e:
            logger.warning(f"Ошибка при поиске ответа для вопроса {i+1}: {e}")
            # Добавляем результат с пустым ответом в случае ошибки
            results.append({
                "question": question,
                "answer": "",
                "score": 0.0,
                "start": 0,
                "end": 0,
                "error": str(e)
            })
            continue
    
    # Статистика
    successful_answers = sum(1 for r in results if r["answer"] and r["score"] > 0.1)
    logger.info(f"Успешно найдено ответов: {successful_answers}/{len(questions)}")
    
    return results

def save_qa_pairs_to_json(qa_pairs: List[Dict[str, Any]], 
                         output_file: str = "qa_pairs.json") -> None:
    """
    Сохраняет пары вопрос-ответ в JSON файл.
    
    Args:
        qa_pairs: Список пар вопрос-ответ
        output_file: Путь для сохранения JSON файла
    """
    try:
        # Подготавливаем данные для сохранения
        output_data = {
            "qa_pairs": qa_pairs,
            "total_pairs": len(qa_pairs),
            "successful_answers": sum(1 for pair in qa_pairs if pair.get("answer") and pair.get("score", 0) > 0.1)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Пары вопрос-ответ успешно сохранены в: {output_file}")
        logger.info(f"Всего пар: {len(qa_pairs)}, Успешных ответов: {output_data['successful_answers']}")
        
    except Exception as e:
        error_msg = f"Ошибка при сохранении в JSON: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)

def generate_qa_pairs_from_text(input_file: str,
                               output_file: str = "qa_pairs.json",
                               num_questions: int = 5,
                               max_tokens: int = 64,
                               qa_model: str = "distilbert-base-uncased-distilled-squad",
                               use_smart_processing: bool = True,
                               max_chunk_size: int = 400,
                               overlap: int = 50) -> Dict[str, Any]:
    """
    Основная функция: генерирует вопросы из текста и находит ответы на них.
    Использует функции generate_questions и filter_answerable_questions из script.py.
    
    Args:
        input_file: Путь к входному текстовому файлу
        output_file: Путь для сохранения JSON с результатами
        num_questions: Количество вопросов для генерации
        max_tokens: Максимальное количество токенов на вопрос
        qa_model: Модель для поиска ответов
        use_smart_processing: Использовать ли умную обработку длинных контекстов
        max_chunk_size: Максимальный размер чанка
        overlap: Размер перекрытия между чанками
        
    Returns:
        Словарь с результатами и статистикой
    """
    logger.info("=== Запуск pipeline генерации вопросов и ответов ===")
    
    try:
        # 1. Чтение текста
        logger.info(f"Чтение текста из файла: {input_file}")
        text = read_txt_file(input_file)
        
        if not text.strip():
            raise ValueError("Входной файл пуст")
        
        # 2. Генерация и фильтрация вопросов (используем функции из script.py)
        answerable_questions = generate_and_filter_questions(
            text=text,
            num_questions=num_questions,
            max_tokens=max_tokens
        )
        
        if not answerable_questions:
            raise ValueError("Не удалось сгенерировать отвечаемые вопросы")
        
        # 3. Инициализация QA модели для поиска точных ответов
        if num_questions < 1 or max_tokens < 1:
            raise ValueError("num_questions and max_tokens must be >= 1")

        qa_model_instance= pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        
        # 4. Поиск ответов на вопросы
        qa_pairs = find_answers_for_questions(
            questions=answerable_questions,
            context=text,
            qa_model=qa_model_instance,
            use_smart_processing=use_smart_processing,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        )
        
        # 5. Сохранение результатов
        save_qa_pairs_to_json(qa_pairs, output_file)
        
        # Статистика
        successful_pairs = [pair for pair in qa_pairs if pair["answer"] and pair["score"] > 0.1]
        
        result_stats = {
            "input_file": input_file,
            "output_file": output_file,
            "total_questions_generated": num_questions,
            "answerable_questions": len(answerable_questions),
            "successful_answers": len(successful_pairs),
            "success_rate": len(successful_pairs) / len(answerable_questions) if answerable_questions else 0,
            "avg_score": sum(pair["score"] for pair in successful_pairs) / len(successful_pairs) if successful_pairs else 0,
            "parameters": {
                "num_questions": num_questions,
                "max_tokens": max_tokens,
                "qa_model": qa_model,
                "use_smart_processing": use_smart_processing,
                "max_chunk_size": max_chunk_size,
                "overlap": overlap
            }
        }
        
        logger.info("=== Pipeline успешно завершен ===")
        logger.info(f"Результаты: {len(successful_pairs)}/{len(answerable_questions)} успешных ответов")
        logger.info(f"Средний score: {result_stats['avg_score']:.3f}")
        
        return result_stats
        
    except Exception as e:
        logger.error(f"Ошибка в pipeline: {e}")
        raise

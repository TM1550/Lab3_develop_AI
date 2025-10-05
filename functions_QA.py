import json
import re
import random
import logging
from collections import Counter
from typing import Dict, List, Tuple, Callable, Any, Union, Optional
from pathlib import Path
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Нормализация текста для сравнения (нижний регистр, только слова и пробелы)."""
    try:
        return re.sub(r'\W+', ' ', text.lower()).strip() if isinstance(text, str) else ""
    except Exception as e:
        logger.warning(f"Ошибка при нормализации текста: {e}")
        return ""


def validate_model_output(prediction: Dict[str, Any]) -> bool:
    """Валидация выходных данных модели."""
    required_keys = ['answer']
    
    if not isinstance(prediction, dict):
        logger.warning("Выходные данные модели должны быть словарем")
        return False
    
    for key in required_keys:
        if key not in prediction:
            logger.warning(f"Отсутствует обязательный ключ '{key}'")
            return False
    
    # Валидация типов данных
    type_checks = [
        ('answer', str),
        ('score', (int, float)),
        ('start', int),
        ('end', int)
    ]
    
    for key, expected_type in type_checks:
        if key in prediction and not isinstance(prediction[key], expected_type):
            logger.warning(f"{key} должен быть {expected_type}, получен {type(prediction[key])}")
            return False
    
    return True


def calculate_f1_score(predicted: str, true: str) -> float:
    """Вычисляет F1-score между предсказанным и правильным ответом."""
    try:
        pred_tokens = normalize_text(predicted).split()
        true_tokens = normalize_text(true).split()

        if not pred_tokens or not true_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(true_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(true_tokens)
        
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Ошибка при вычислении F1-score: {e}")
        return 0.0


def is_answer_correct(predicted: str, true: str, threshold: float = 0.8) -> bool:
    """Проверяет корректность ответа на основе F1-score."""
    if not 0 <= threshold <= 1:
        logger.warning(f"Threshold должен быть в диапазоне [0, 1], получен {threshold}")
        return False
    
    return calculate_f1_score(predicted, true) >= threshold


def validate_dataset_item(item: Dict[str, Any]) -> bool:
    """Валидация элемента датасета."""
    required_keys = ['question', 'context', 'answer']
    
    if not isinstance(item, dict):
        logger.warning("Элемент датасета должен быть словарем")
        return False
    
    for key in required_keys:
        if key not in item or not isinstance(item[key], str) or not item[key].strip():
            logger.warning(f"Некорректное значение ключа '{key}'")
            return False
    
    return True


def load_json_dataset(json_file: str) -> Tuple[List[Dict], str]:
    """Загрузка и валидация JSON датасета."""
    try:
        json_path = Path(json_file)
        
        if not json_path.exists():
            return [], f"Файл не найден: {json_file}"
        
        if json_path.stat().st_size == 0:
            return [], f"Файл пуст: {json_file}"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, list):
            return [], f"Датасет должен быть списком, получен {type(dataset)}"
        
        return dataset, ""
        
    except json.JSONDecodeError as e:
        return [], f"Ошибка декодирования JSON: {e}"
    except UnicodeDecodeError as e:
        return [], f"Ошибка декодирования текста: {e}"
    except Exception as e:
        return [], f"Неожиданная ошибка при загрузке файла: {e}"


def calculate_f1_for_json_dataset(json_file: str, model: Callable) -> Tuple[float, float]:
    """Вычисляет F1-score и accuracy на JSON датасете."""
    dataset, error = load_json_dataset(json_file)
    if error:
        logger.error(error)
        return 0.0, 0.0
    
    if not dataset:
        logger.error("Датасет пуст")
        return 0.0, 0.0
    
    total_f1, correct_count, processed_count, invalid_items = 0.0, 0, 0, 0

    for i, item in enumerate(tqdm(dataset, desc="Calculating F1")):
        try:
            if not validate_dataset_item(item) or not callable(model):
                invalid_items += 1
                continue
            
            prediction = model(question=item["question"], context=item["context"])
            
            if not validate_model_output(prediction):
                invalid_items += 1
                continue
            
            f1 = calculate_f1_score(prediction["answer"], item["answer"])
            total_f1 += f1
            correct_count += int(f1 >= 0.8)
            processed_count += 1
            
        except Exception as e:
            logger.warning(f"Ошибка при обработке элемента {i}: {e}")
            continue

    if invalid_items > 0:
        logger.info(f"Пропущено {invalid_items} невалидных элементов")

    if processed_count == 0:
        logger.error("Не удалось обработать ни одного элемента")
        return 0.0, 0.0

    accuracy = correct_count / processed_count
    avg_f1 = total_f1 / processed_count

    logger.info(f"Обработано элементов: {processed_count}/{len(dataset)}")
    logger.info(f"F1-score: {avg_f1:.4f}, Accuracy: {accuracy:.4f}")

    return avg_f1, accuracy


def get_detailed_answer(
    model: Callable, 
    question: str, 
    context: str, 
    return_metadata: bool = False
) -> Dict[str, Any]:
    """Расширенная версия функции с дополнительной информацией."""
    # Валидация входных параметров
    try:
        if not callable(model):
            raise ValueError("Модель должна быть callable")
        
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Вопрос должен быть непустой строкой")
        
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Контекст должен быть непустой строкой")
        
        # Вызов модели
        prediction = model(question=question, context=context)
        
        # Формирование базового результата
        result = {
            'answer': prediction.get('answer', ''),
            'score': prediction.get('score', 0.0),
            'start': prediction.get('start', 0),
            'end': prediction.get('end', 0)
        }
        
        # Добавление метаданных при необходимости
        if return_metadata:
            start_pos = prediction.get('start', 0)
            end_pos = prediction.get('end', 0)
            
            # Извлечение сниппета контекста
            snippet_start = max(0, start_pos - 50)
            snippet_end = min(len(context), end_pos + 50)
            context_snippet = context[snippet_start:snippet_end]
            
            result.update({
                'question': question,
                'context_snippet': context_snippet,
                'timestamp': prediction.get('timestamp', None)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка в get_detailed_answer: {e}")
        
        # Создание результата с ошибкой
        error_result = {
            'answer': '',
            'score': 0.0,
            'start': 0,
            'end': 0,
            'error': str(e)
        }
        
        if return_metadata:
            error_result.update({
                'question': question,
                'context_snippet': context[:100] if context else "",
                'timestamp': None
            })
        
        return error_result
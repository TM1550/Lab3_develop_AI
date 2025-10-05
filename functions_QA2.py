import logging
import re
from typing import Dict, Any, List, Tuple, Callable
from tqdm import tqdm

from functions_QA import validate_model_output, calculate_f1_score, validate_dataset_item, load_json_dataset, normalize_text

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_context_with_overlap(context: str, chunk_size: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Разбивает контекст на перекрывающиеся чанки.
    """
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': current_length,
                'tokens': len(chunk_text.split())
            })
            
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words
            current_length = len(' '.join(overlap_words).split())
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'length': current_length,
            'tokens': len(chunk_text.split())
        })
    
    return chunks

def smart_qa_with_aggregation(model, question: str, context: str, 
                             max_chunk_size: int = 400, 
                             overlap: int = 50) -> Dict[str, Any]:
    """
    Обрабатывает длинный контекст через разбиение на чанки и агрегацию результатов.
    """
    chunks = split_context_with_overlap(context, max_chunk_size, overlap)
    
    if not chunks:
        return {'answer': '', 'score': 0.0, 'start': 0, 'end': 0}
    
    results = []
    for i, chunk in enumerate(chunks):
        try:
            result = model(question=question, context=chunk['text'])
            result['chunk_index'] = i
            result['chunk_text'] = chunk['text']
            result['chunk_length'] = chunk['length']
            results.append(result)
        except Exception as e:
            logger.warning(f"Ошибка при обработке чанка {i}: {e}")
            continue
    
    if not results:
        return {'answer': '', 'score': 0.0, 'start': 0, 'end': 0}
    
    best_result = max(results, key=lambda x: x.get('score', 0))
    
    if 'chunk_text' in best_result:
        chunk_start = context.find(best_result['chunk_text'])
        if chunk_start != -1:
            best_result['start'] = chunk_start + best_result.get('start', 0)
            best_result['end'] = chunk_start + best_result.get('end', 0)
    
    for field in ['chunk_index', 'chunk_text', 'chunk_length']:
        best_result.pop(field, None)
    
    return best_result

def calculate_f1_for_json_dataset(json_file: str, model: Callable, 
                                 use_smart_processing: bool = True,
                                 max_chunk_size: int = 400,
                                 overlap: int = 50,
                                 f1_threshold: float = 0.8,
                                 use_min_answer: bool = False) -> Tuple[float, float, Dict[str, Any]]:
    """
    Вычисляет F1-score и accuracy на JSON датасете с поддержкой обработки длинных контекстов.
    
    Args:
        json_file: Путь к JSON файлу с датасетом
        model: Модель для QA
        use_smart_processing: Использовать ли умную обработку длинных контекстов
        max_chunk_size: Максимальный размер чанка для разбиения
        overlap: Размер перекрытия между чанками
        f1_threshold: Порог F1-score для определения корректности ответа
        use_min_answer: Использовать минимальный ответ (min_answer) вместо основного ответа
    
    Returns:
        Кортеж (средний F1-score, accuracy, дополнительная статистика)
    """
    dataset, error = load_json_dataset(json_file)
    if error:
        logger.error(error)
        return 0.0, 0.0, {}
    
    if not dataset:
        logger.error("Датасет пуст")
        return 0.0, 0.0, {}
    
    total_f1, correct_count, processed_count, invalid_items = 0.0, 0, 0, 0
    total_precision, total_recall = 0.0, 0.0
    min_answer_used_count = 0

    for i, item in enumerate(tqdm(dataset, desc="Calculating F1")):
        try:
            # Валидация элемента датасета
            if not validate_dataset_item(item):
                logger.warning(f"Пропущен невалидный элемент {i}")
                invalid_items += 1
                continue
            
            question = item["question"]
            context = item["context"]
            
            # Определение истинного ответа
            if use_min_answer and "min_answer" in item and item["min_answer"]:
                true_answer = item["min_answer"]
                min_answer_used_count += 1
            else:
                true_answer = item["answer"]
            
            # Выбор стратегии обработки в зависимости от длины контекста
            word_count = len(context.split())
            needs_chunking = word_count > max_chunk_size
            
            if use_smart_processing and needs_chunking:
                # Используем умную обработку для длинных контекстов
                prediction = smart_qa_with_aggregation(
                    model, question, context, max_chunk_size, overlap
                )
            else:
                # Прямой вызов модели для коротких контекстов
                prediction = model(question=question, context=context)
            
            # Валидация выходных данных модели
            if not validate_model_output(prediction):
                logger.warning(f"Невалидные выходные данные модели для элемента {i}")
                invalid_items += 1
                continue
            
            # Вычисление F1-score (используем существующую функцию из functions_QA)
            f1 = calculate_f1_score(prediction["answer"], true_answer)
            
            # Для совместимости вычисляем precision и recall отдельно
            pred_tokens = normalize_text(prediction["answer"]).split()
            true_tokens = normalize_text(true_answer).split()
            
            if not pred_tokens or not true_tokens:
                precision = 0.0
                recall = 0.0
            else:
                common = set(pred_tokens) & set(true_tokens)
                num_common = len(common)
                
                precision = num_common / len(pred_tokens) if pred_tokens else 0.0
                recall = num_common / len(true_tokens) if true_tokens else 0.0
            
            total_f1 += f1
            total_precision += precision
            total_recall += recall
            
            correct_count += int(f1 >= f1_threshold)
            processed_count += 1
            
            # Логирование деталей для отладки (первые 10 элементов)
            if i < 10:
                answer_type = "min_answer" if use_min_answer and "min_answer" in item and item["min_answer"] else "answer"
                logger.debug(f"Элемент {i}: F1={f1:.3f}, Ответ='{prediction['answer']}', Ожидаемый ({answer_type})='{true_answer}'")
            
        except Exception as e:
            logger.warning(f"Ошибка при обработке элемента {i}: {e}")
            continue

    # Статистика обработки
    if invalid_items > 0:
        logger.info(f"Пропущено {invalid_items} невалидных элементов из {len(dataset)}")

    if processed_count == 0:
        logger.error("Не удалось обработать ни одного элемента")
        return 0.0, 0.0, {}

    # Расчет метрик
    avg_f1 = total_f1 / processed_count
    avg_precision = total_precision / processed_count
    avg_recall = total_recall / processed_count
    accuracy = correct_count / processed_count

    # Дополнительная статистика
    stats = {
        'processed_count': processed_count,
        'total_items': len(dataset),
        'invalid_items': invalid_items,
        'min_answer_used_count': min_answer_used_count,
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'accuracy': accuracy,
        'use_min_answer': use_min_answer
    }

    logger.info(f"Обработано элементов: {processed_count}/{len(dataset)}")
    logger.info(f"Использовано min_answer: {min_answer_used_count}/{processed_count}")
    logger.info(f"F1-score: {avg_f1:.4f}, Accuracy: {accuracy:.4f} (порог: {f1_threshold})")
    logger.info(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
    logger.info(f"Использована умная обработка: {use_smart_processing}")

    return avg_f1, accuracy, stats


def calculate_comprehensive_metrics(json_file: str, model: Callable, 
                                  use_smart_processing: bool = True,
                                  max_chunk_size: int = 400,
                                  overlap: int = 50,
                                  f1_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Вычисляет комплексные метрики для всех вариантов ответов.
    
    Returns:
        Словарь с результатами для всех комбинаций
    """
    logger.info("=== Комплексная оценка метрик ===")
    
    # Все комбинации параметров
    combinations = [
        {"use_min_answer": False, "name": "Основной ответ"},
        {"use_min_answer": True, "name": "Минимальный ответ"}
    ]
    
    results = {}
    
    for combo in combinations:
        logger.info(f"Вычисление для: {combo['name']}")
        
        f1_score, accuracy, stats = calculate_f1_for_json_dataset(
            json_file=json_file,
            model=model,
            use_smart_processing=use_smart_processing,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
            f1_threshold=f1_threshold,
            use_min_answer=combo['use_min_answer']
        )
        
        results[combo['name']] = {
            'f1_score': f1_score,
            'accuracy': accuracy,
            'stats': stats
        }
    
    return results

# Дополнительная функция для сравнения стратегий обработки
def compare_processing_strategies(json_file: str, model: Callable, 
                                 max_chunk_size: int = 400,
                                 use_min_answer: bool = False) -> Dict[str, Any]:
    """
    Сравнивает производительность с использованием умной обработки и без нее.
    
    Returns:
        Словарь с результатами сравнения
    """
    logger.info("=== Сравнение стратегий обработки ===")
    
    # Тест с умной обработкой
    logger.info("Тестирование с умной обработкой...")
    f1_smart, acc_smart, stats_smart = calculate_f1_for_json_dataset(
        json_file, model, use_smart_processing=True, 
        max_chunk_size=max_chunk_size, use_min_answer=use_min_answer
    )
    
    # Тест без умной обработки
    logger.info("Тестирование без умной обработки...")
    f1_direct, acc_direct, stats_direct = calculate_f1_for_json_dataset(
        json_file, model, use_smart_processing=False, 
        max_chunk_size=max_chunk_size, use_min_answer=use_min_answer
    )
    
    comparison = {
        'with_smart_processing': {
            'f1_score': f1_smart,
            'accuracy': acc_smart,
            'stats': stats_smart
        },
        'without_smart_processing': {
            'f1_score': f1_direct,
            'accuracy': acc_direct,
            'stats': stats_direct
        },
        'improvement': {
            'f1_score': f1_smart - f1_direct,
            'accuracy': acc_smart - acc_direct
        },
        'parameters': {
            'use_min_answer': use_min_answer
        }
    }
    
    logger.info(f"Улучшение F1-score: {comparison['improvement']['f1_score']:.4f}")
    logger.info(f"Улучшение accuracy: {comparison['improvement']['accuracy']:.4f}")
    
    return comparison


def read_txt_file(file_path: str) -> str:
    """
    Читает содержимое TXT файла и возвращает его как строку.
    
    Args:
        file_path: Путь к TXT файлу
        
    Returns:
        Строка с содержимым файла
        
    Raises:
        FileNotFoundError: Если файл не существует
        UnicodeDecodeError: Если возникли проблемы с кодировкой
        Exception: Другие ошибки при чтении файла
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info(f"Успешно прочитан файл: {file_path} (длина: {len(content)} символов)")
            return content
    except FileNotFoundError:
        error_msg = f"Файл не найден: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except UnicodeDecodeError as e:
        # Попробуем другие кодировки
        try:
            with open(file_path, 'r', encoding='cp1251') as file:
                content = file.read()
                logger.info(f"Успешно прочитан файл в кодировке cp1251: {file_path}")
                return content
        except UnicodeDecodeError:
            error_msg = f"Ошибка декодирования файла {file_path}: {e}"
            logger.error(error_msg)
            raise UnicodeDecodeError(error_msg)
    except Exception as e:
        error_msg = f"Ошибка при чтении файла {file_path}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)


# Дополнительная функция для обработки нескольких TXT файлов
def read_multiple_txt_files(file_paths: List[str]) -> Dict[str, str]:
    """
    Читает несколько TXT файлов и возвращает словарь с их содержимым.
    
    Args:
        file_paths: Список путей к TXT файлам
        
    Returns:
        Словарь {имя_файла: содержимое}
    """
    files_content = {}
    
    for file_path in file_paths:
        try:
            # Получаем имя файла из пути
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path
            file_name = file_name.split('\\')[-1] if '\\' in file_path else file_name
            
            content = read_txt_file(file_path)
            files_content[file_name] = content
            
        except Exception as e:
            logger.warning(f"Не удалось прочитать файл {file_path}: {e}")
            continue
    logger.info(f"Успешно прочитано {len(files_content)} из {len(file_paths)} файлов")
    return files_content

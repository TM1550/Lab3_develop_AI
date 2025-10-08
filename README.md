# Lab3_develop_AI

Установка:

git clone https://github.com/TM1550/Lab3_develop_AI.git

cd Lab3_develop_AI

pip install -r requirements.txt

question_answer_pipeline.py - расширенные функции с поддержкой обработки длинных контекстов, для его работы нужны functions_QA2.py, functions_QA.py и script.py

Как использовать функции из question_answer_pipeline.py первый способ: нужно закинуть этот файл и необходимые для него файлы в свою рабочую папку и импортировать функции, как показано в комментарии в example.py

Как использовать функции из question_answer_pipeline.py второй способ: нужно импортировать модуль create_QA, как в example.py

example.py пример для генерации ответа на вопрос и получения метрик на датасете

Запуск:

python example.py
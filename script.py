import argparse
import torch
from transformers import pipeline


def read_text_file(file_path):
    """
    Reads the content of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Text content of the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    with open(file_path, "r", encoding="utf-8") as text_input:
        text = text_input.read()
    return text


def write_questions(questions, output_file="output.txt"):
    """
    Writes a list of questions to a text file, one question per line.

    Args:
        questions (list of str): List of question strings to save.
        output_file (str, optional): File path to save the questions. Defaults to "output.txt".

    Returns:
        None
    Notes:
        Overwrites the file if it already exists.
    """
    with open(output_file, "w", encoding="utf-8") as text_output:
        for q in questions:
            text_output.write(q + "\n")


def generate_questions(text, num_questions=3, max_tokens=64):
    """
    Generates questions from a given text using the T5 question generation model.

    Args:
        text (str): Input text from which to generate questions.
        num_questions (int, optional): Number of questions to generate. Defaults to 3.
        max_tokens (int, optional): Maximum number of tokens per generated question. Defaults to 64.

    Returns:
        list[str]: A list of generated question strings.

    Notes:
        Automatically detects if a GPU is available and uses it if possible.
        Uses the 'valhalla/t5-base-qg-hl' model for question generation.
    """
    if num_questions < 1 or max_tokens < 1:
        raise ValueError("num_questions and max_tokens must be >= 1")

    device = 0 if torch.cuda.is_available() else -1
    num_beams = max(num_questions, 4)

    qg_pipeline = pipeline(
        task="text2text-generation",
        model="valhalla/t5-base-qg-hl",
        device=device,
        num_beams=num_beams,
    )

    questions = qg_pipeline(
        f"generate questions: {text}",
        max_new_tokens=max_tokens,
        num_return_sequences=num_questions,
    )
    return [q["generated_text"] for q in questions]


def filter_answerable_questions(questions, context):
    """
    Filters questions that can be answered based on the given context using a QA model.

    Args:
        questions (list[str]): List of questions to check.
        context (str): Context text to answer the questions.

    Returns:
        list[str]: List of questions that have valid answers.
    """
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        device=device,
    )

    answerable = []
    for q in questions:
        try:
            result = qa_pipeline(question=q, context=context)
            answer = result.get("answer", "").strip()
            if answer:
                answerable.append(q)
        except Exception:
            continue
    return answerable


def main():
    parser = argparse.ArgumentParser(
        description="Generate answerable questions from text using T5 and QA model"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        required=True,
        help="The path to the text for generating questions.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.txt",
        help="File to save generated questions.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=3,
        help="Number of questions to generate per text",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="The maximum number of tokens to generate a question",
    )

    args = parser.parse_args()

    try:
        text = read_text_file(args.text_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    if not text.strip():
        print("Error: input file is empty.")
        exit(1)

    try:
        questions = generate_questions(
            text, num_questions=args.num_questions, max_tokens=args.max_tokens
        )
    except ValueError as ve:
        print(f"Error generating questions: {ve}")
        exit(1)

    answerable_questions = filter_answerable_questions(questions, context=text)

    for i, q in enumerate(answerable_questions, 1):
        print(f"Question {i}: {q}")

    write_questions(answerable_questions, args.output_file)


if __name__ == "__main__":
    main()

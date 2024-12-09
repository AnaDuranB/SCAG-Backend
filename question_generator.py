import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import random
from typing import List, Dict
from torch.utils.data import Dataset
from typing import List

import os

class QuestionGenerator:
    def __init__(self, checkpoint_path="checkpoints/checkpoint_batch_14"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.generator = pipeline(
            'text2text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_length=128,
            num_beams=5,
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.5
        )
        print(f"Processing on device: {self.device}")

    def generate_question(self, fact: str) -> str:
        prompt = f"Generate a specific and clear multiple choice question about this fact: {fact}. The question should test understanding and be educational."

        question = self.generator(
            prompt,
            max_length=128,
            num_return_sequences=1,
            num_beams=5,
            do_sample=True,
            temperature=0.5,
            top_p=0.85,
            repetition_penalty=1.5
        )[0]['generated_text']

        return question

    def generate_correct_answer(self, fact: str, question: str) -> str:
        prompt = f"Based on this fact: {fact}\nFor this question: {question}\nGenerate the precise correct answer:"

        answer = self.generator(
            prompt,
            max_length=128,
            num_return_sequences=1,
            num_beams=3,
            do_sample=False
        )[0]['generated_text']

        return answer

    def generate_distractors(self, fact: str, correct_answer: str, question: str) -> List[str]:
        prompt = f"""
                    Fact: {fact}
                    Question: {question}
                    Correct Answer: {correct_answer}
                    Task: Generate 3 wrong but plausible and different answers that are related to the topic.
                    Each answer should be clearly different from the correct answer and from each other.
                    """

        distractors = self.generator(
            prompt,
            max_length=128,
            num_return_sequences=5,
            num_beams=10,
            do_sample=True,
            temperature=0.5,
            top_p=0.85
        )

        distractor_set = set()

        max_retries = 5  # maximum retries allowed
        retries = 0

        while len(distractor_set) < 3 and retries < max_retries:
            new_distractor = self.generator(
                prompt,
                max_length=64,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )[0]['generated_text'].strip()

            if new_distractor != correct_answer: #and new_distractor not in distractor_set:
                distractor_set.add(new_distractor)
            retries += 1

        if retries >= max_retries:
            print("Warning: Unable to generate enough distractors within the maximum retry limit.")

        return list(distractor_set)

    def generate_multiple_choice_question(self, info: Dict) -> Dict:
        fact = info['key_fact']
        question = self.generate_question(fact)
        correct_answer = self.generate_correct_answer(fact, question)
        distractors = self.generate_distractors(fact, correct_answer, question)

        options = distractors + [correct_answer]
        random.shuffle(options)

        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer
        }

    def format_questions(self, questions: List[Dict]) -> List[str]:
        """
        Formatea las preguntas generadas para que sean fáciles de leer.
        """
        formatted_questions = []
        for i, question_dict in enumerate(questions, 1):
            question_text = f"Pregunta {i}:\n{question_dict['question']}\n\n"
            for j, option in enumerate(question_dict['options']):
                question_text += f"{chr(65 + j)}) {option}\n"  # Opciones como A), B), C), etc.
            question_text += f"\nRespuesta correcta: {question_dict['correct_answer']}\n"
            formatted_questions.append(question_text)
        return formatted_questions
    
    def process_text(self, text: str) -> List[Dict]:
        key_info = self.extract_key_information(text)
        questions = []
        for info in key_info:
            question = self.generate_multiple_choice_question(info)
            if question:  # Asegúrate de que la pregunta no sea None o vacía
                print(f"Generated Question: {question}")  # Debug
                questions.append(question)
        return questions

    def extract_key_information(self, text: str) -> List[Dict]:
        sentences = sent_tokenize(text)
        key_info = []

        for sentence in sentences:
            prompt = f"Extract key fact from this sentence: {sentence}"
            fact = self.generator(
                prompt,
                max_length=256,
                num_return_sequences=1,
                temperature=0.3
            )[0]['generated_text']

            key_info.append({
                'original_sentence': sentence,
                'key_fact': fact
            })

        return key_info
    
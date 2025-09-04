"""Exam Evaluator Module."""

from sdk.llm_evaluator import LLMExamJudger


class ExamEvaluator:
    """Exam Evaluator class for evaluating exam questions and answers."""

    def __init__(self) -> None:
        """Initialize the ExamEvaluator class."""
        pass

    def calculate_score(self, user_answer, correct_answer, question_type, full_score=5):
        """Calculate scores for objective questions (single-choice, multiple-choice, true/false).

        :param user_answer: User's answer
        :param correct_answer: Correct answer
        :param question_type: Question type, options: 'SingleChoice', 'MultipleChoice', 'True/False Questions'
        :param full_score: Full score for this question (default 5 points)
        :return: Score (integer)
        """
        user_answer = user_answer.strip().upper()
        correct_answer = correct_answer.strip().upper()

        if question_type == 'SingleChoice':
            # Single-choice: compare answers directly
            return full_score if user_answer == correct_answer else 0

        elif question_type == 'MultipleChoice':
            # Multiple-choice: split into sets for comparison
            user_choices = set(user_answer.split(',')) if user_answer else set()
            correct_choices = set(correct_answer.split(','))

            if not user_choices:
                return 0  # No selection gets 0 points
            elif user_choices == correct_choices:
                return full_score  # All correct gets full score
            elif user_choices.issubset(correct_choices):
                return 2  # Partial selection gets 2 points
            else:
                return 0  # Wrong selection gets 0 points

        elif question_type == 'True/False Questions':
            return full_score if user_answer == correct_answer else 0

        else:
            raise ValueError(
                f"Invalid question_type: '{question_type}'. Must be one of: 'SingleChoice', 'MultipleChoice', 'True/False Questions'"
            )

    def offline_evaluate(self, llm_answer, groundtruth, model_name):
        """Evaluate the performance of the model on a given question."""
        llmjudger = None
        llmjudger_explanation = None
        llmjudger_system_prompt = None
        if groundtruth['type'] == 'ShortAnswerQuestion':
            llmjudger = LLMExamJudger(model_name=model_name)
            llmjudger_score, llmjudger_explanation, llmjudger_system_prompt = llmjudger.run(
                llm_answer=llm_answer, groundtruth=groundtruth
            )
            llm_score = llmjudger_score
        else:
            llm_score = self.calculate_score(
                user_answer=llm_answer,
                correct_answer=groundtruth['answer'],
                question_type=groundtruth['type'],
                full_score=groundtruth['points'],
            )

        return {
            'llmjudger_explanation': llmjudger_explanation,
            'llmjudger_system_prompt': llmjudger_system_prompt,
            'llm_score': llm_score,
        }

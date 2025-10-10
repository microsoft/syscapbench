"""Base class for evaluators."""

import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sdk.llm import LLM


class Evaluator:
    """Base class for evaluators."""

    def __init__(self) -> None:
        """Initialize the Evaluator class."""
        pass

    def eval(self, *args, **kwargs):
        """Evaluate the performance - to be overridden by subclasses."""
        raise NotImplementedError('Subclasses must implement the eval method')


class BasicEvaluator(Evaluator):
    """Evaluator class for evaluating the performance of the model."""

    def __init__(self, _model_name='gpt-4o') -> None:
        """Initialize the Evaluator class."""
        self.model_name = _model_name

    def syntax_correctness(self, query, pattern=r'```json(.*?)```'):
        """Check the syntax correctness of the query."""
        return 1 if re.search(pattern, query) else 0

    def exact_match(self, query1, query2):
        """Check if the two queries are exactly the same."""
        query1 = query1.strip()
        query2 = query2.strip()
        return 1 if query1 == query2 else 0

    def includes(self, query1, query2):
        """Check if query1 includes query2."""
        query1 = query1.strip()
        query2 = query2.strip()
        return 1 if (query2 in query1) or (query1 in query2) else 0

    def jaccard_similarity_ngrams(self, query1, query2, n=2):
        """Calculate the Jaccard similarity between two queries using n-grams."""
        query_ngrams1 = {query1[i : i + n] for i in range(len(query1) - n + 1)}
        query_ngrams2 = {query2[i : i + n] for i in range(len(query2) - n + 1)}
        intersection = query_ngrams1.intersection(query_ngrams2)
        union = query_ngrams1.union(query_ngrams2)
        similarity = len(intersection) / len(union)
        return similarity

    def cosine_similarity_strings(self, query1, query2):
        """Calculate the cosine similarity between two strings."""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query1, query2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def embeddings_similarity(self, query1, query2):
        """Calculate the cosine similarity between two queries using embeddings."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences = [query1, query2]
        embeddings = model.encode(sentences)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def eval(self, question, answer, groundtruth):
        """Evaluate the performance of the model."""
        syntax_acc = self.syntax_correctness(answer, pattern=r'cluster\(.*?\)\.database\(.*?\)')
        em = self.exact_match(answer, groundtruth['response'])
        jaccard_sim = self.jaccard_similarity_ngrams(answer, groundtruth['response'])
        cos_sim = self.cosine_similarity_strings(answer, groundtruth['response'])
        embeddings_similarity = self.embeddings_similarity(answer, groundtruth['response'])
        llmjudger = LLMJudger(self.model_name)
        llmjudger_results = llmjudger.eval(question=question, answer=answer, ref_answer=groundtruth['response'])
        return {
            'syntax_acc': syntax_acc,
            'exact_match': em,
            'jaccard_similarity': jaccard_sim,
            'cosine_similarity': cos_sim,
            'embeddings_similarity': embeddings_similarity,
            'llmjudger_rating': llmjudger_results['llmjudger_rating'],
            'llmjudger_answer': llmjudger_results['llmjudger_answer'],
        }


class ExamEvaluator(Evaluator):
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

    def eval(self, llm_answer, groundtruth, model_name):
        """Evaluate the performance of the model on a given question."""
        llmjudger = None
        llmjudger_explanation = None
        llmjudger_system_prompt = None
        if groundtruth['type'] == 'ShortAnswerQuestion':
            llmjudger = LLMExamJudger(model_name=model_name)
            llmjudger_results = llmjudger.eval(llm_answer=llm_answer, groundtruth=groundtruth)
            llm_score = llmjudger_results['score']
            llmjudger_explanation = llmjudger_results['explanation']
            # llmjudger_system_prompt = llmjudger.system_prompt
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


class LLMJudger(Evaluator):
    """LLM class for judging the quality of the response."""

    def __init__(self, _model_name) -> None:
        """Initialize the LLMJudger class."""
        self.system_prompt = 'You are a helpful assistant.'
        self.user_prompt_template = """
        [Instruction]
        Please act as an impartial judge and evaluate the quality of the response provided by an
        AI assistant to the user question displayed below. Your evaluation should consider
        correctness and helpfulness. You will be given a reference answer and the assistant's
        answer. Begin your evaluation by comparing the assistant's answer with the reference answer.
        Identify and correct any mistakes. Be as objective as possible.

        After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly
        following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

        [Question]
        {question}

        [The Start of Reference Answer]
        {ref_answer_1}
        [The End of Reference Answer]

        [The Start of Assistant's Answer]
        {answer}
        [The End of Assistant's Answer]
        """
        self.model_name = _model_name
        self.LLM = LLM(engine=self.model_name, system_prompt=self.system_prompt, temperature=0.1)

    def extract_rating(self, text: str) -> str:
        """Extract the rating from the text."""
        # ectract rating here, eg. [[5]]
        ratings = re.findall(r'Rating: \[\[(\d+)\]\]', text)
        rating = float(ratings[0]) if len(ratings) > 0 else 0
        rating = rating / 10 if rating != 0 else rating
        return rating

    def eval(self, question, answer, ref_answer):
        """Run the LLMJudger."""
        user_prompt = self.user_prompt_template.format(question=question, ref_answer_1=ref_answer, answer=answer)
        answer = self.LLM.query(user_prompt)
        rating = self.extract_rating(answer)
        # Save both original judge response and extracted rating in the detailed
        # result output. Also need to calculate the average llm rating as the other metrics does.
        return {
            'llmjudger_answer': answer,
            'llmjudger_rating': rating,
        }


class LLMExamJudger(Evaluator):
    """LLM class for judging exam responses based on a standard answer."""

    def __init__(self, model_name) -> None:
        """Initialize the LLMExamJudger class."""
        self.model_name = model_name
        self.system_prompt = """**You are a meticulous university professor grading exam responses.** Evaluate the submission strictly following the guidelines below and return **ONLY** a JSON response as specified.

**Input Data Structure:**
```json
<input_data_placeholder>
```

**Output Requirements:**
```json
{
  "score": "<assigned_score>",
  "explanation": "<evaluation>",
}
```

**Evaluation Protocol:**
1. **Strict Alignment**: Compare `student_answer` to `standard_answer` for:
   - **Accuracy** (0-100% of `question_points`).
   - **Completeness** (missing concepts deduct points proportionally).
2. **Quality Metrics**:
   - Logical consistency (e.g., contradictions invalidate arguments).
   - Clarity (organization, grammar, and academic tone).
   - **Partial Credit**: Awarded for partially correct reasoning.
3. **Rubric-Based** (if provided): Prioritize criteria in `rubric_criteria`.

**Response Rules:**
- **JSON ONLY** - No extraneous text or Markdown.
- **Precision**: Use exact keys (`score`, `explanation`, `breakdown`).
- **Explanation**: 3-5 sentences.

**Example Output:**
```json
{
  "score": 8,
  "explanation": "The student accurately explained X and Y (60% of points), but omitted Z (40% deduction). Their argument was logically sound but lacked citations. To improve: integrate scholarly sources.",
}
```

Note: For multiple-choice questions, selecting all correct answers gives full marks, selecting some correct answers gives 2 points, and selecting any incorrect answers results in 0 points.
"""

    def eval(self, llm_answer, groundtruth):
        """Run the LLMExamJudger."""
        input_data = {
            'course_name': groundtruth['course'],
            'question_points': groundtruth['points'],
            'question_description': groundtruth['problem'],
            'answer': groundtruth['answer'],
            'student_answer': llm_answer,
        }
        self.system_prompt = self.system_prompt.replace('<input_data_placeholder>', json.dumps(input_data, indent=2))

        self.llm = LLM(engine=self.model_name, temperature=0.1, json_format=True)
        response = self.llm.query(self.system_prompt)

        score = json.loads(response).get('score', '')
        explanation = json.loads(response).get('explanation', '')
        return {
            'score': score,
            'explanation': explanation,
        }

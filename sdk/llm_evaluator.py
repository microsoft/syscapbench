"""Example for benchmarking the performance of a model on a specific task."""

import json
import re

from sdk.llm import LLM


class LLMJudger:
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

    def run(self, question, answer, ref_answer):
        """Run the LLMJudger."""
        user_prompt = self.user_prompt_template.format(question=question, ref_answer_1=ref_answer, answer=answer)
        answer = self.LLM.query(user_prompt)
        rating = self.extract_rating(answer)
        # Save both original judge response and extracted rating in the detailed
        # result output. Also need to calculate the average llm rating as the other metrics does.
        return answer, rating


class LLMExamJudger:
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

    def run(self, llm_answer, groundtruth):
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
        return score, explanation, self.system_prompt

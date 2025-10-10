"""This script evaluates a course exam benchmark using a specified LLM model."""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, LongType, StringType, StructField, StructType

from sdk.utils import set_llm_endpoint_from_config

set_llm_endpoint_from_config('env.toml')

from sdk.evaluator import ExamEvaluator  # noqa: E402
from sdk.executor import SimpleExecutor  # noqa: E402


# Spark is used for aggregating results.
def create_sparksession() -> SparkSession:
    """Create a Spark session for processing data."""
    return SparkSession.builder.getOrCreate()


def problem_format(problem_type: str) -> str:
    """Generate the format description based on the problem type.

    Args:
        problem_type (str): The type of the problem (e.g., 'SingleChoice', 'MultipleChoice', 'True/False Questions', 'ShortAnswerQuestion').

    Returns:
        str: The format description for the problem type.
    """
    FORMAT_DESCRIPTION_SINGLE_CHOCIE = """
    This is a Single-choice problem.

    Please return your response in the following JSON format:
    ```json
    {"answer": "A", "explanation": "Your explanation here."}
    ```
    """
    FORMAT_DESCRIPTION_MULTIPLE_CHOICE = """
    This is a MultipleChoice problem.

    Please return your response in the following JSON format:
    ```json
    {"answer": "A,B,C", "explanation": "Your explanation here."}
    ```

    answer is capital letters separated by commas, without Spaces
    """

    FORMAT_DESCRIPTION_TRUE_FALSE_QUESTIONS = """
    This is a True/False problem.

    Please return your response in the following JSON format:
    ```json
    {"answer": "True,False,True", "explanation": "Your explanation here."}
    ```

    answer is each item corresponds to a sub-question
    """
    FORMAT_DESCRIPTION_SHORT_ANSWER_QUESTION = """
    This is a ShortAnswerQuestion problem.

    Please return your response in the following JSON format:
    ```json
    {"answer": "Youe answer here.", "explanation": "Your explanation here."}
    ```
    """

    if problem_type == 'SingleChoice':
        format_description = FORMAT_DESCRIPTION_SINGLE_CHOCIE
    elif problem_type == 'MultipleChoice':
        format_description = FORMAT_DESCRIPTION_MULTIPLE_CHOICE
    elif problem_type == 'True/False Questions':
        format_description = FORMAT_DESCRIPTION_TRUE_FALSE_QUESTIONS
    elif problem_type == 'ShortAnswerQuestion':
        format_description = FORMAT_DESCRIPTION_SHORT_ANSWER_QUESTION
    else:
        raise ValueError(f'Unknown problem type: {problem_type}')

    return format_description


def main(input_file, output_dir, model_name, agent_name):
    """Main function for running the course exam benchmark."""
    results = []
    with open(input_file, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    for groundtruth in data:
        try:
            logger.info(f"============ {groundtruth['instance_id']} ============")

            problem_type = groundtruth['type']
            format_description = problem_format(problem_type)

            system_prompt = (
                f"You are a university student who has completed the {groundtruth['course']} course. You are now answering a final exam question."
                + format_description
            )
            user_prompt = 'Below is the problem description:\n' + groundtruth['problem']

            if agent_name == 'llm':
                executor = SimpleExecutor(model_name, system_prompt)
            else:
                # You can add more agents here
                raise ValueError(f'Unknown agent name: {agent_name}')
            response = executor.run(user_prompt, lang='json')
            response = json.loads(response)

            answer = str(response.get('answer', ''))
            explanation = response.get('explanation', '')
            logger.info(f'Model Answer: {answer}')
            logger.info(f'Model Explanation: {explanation}')

            evaluator = ExamEvaluator()
            offline_metrics = evaluator.eval(llm_answer=answer, groundtruth=groundtruth, model_name=model_name)

            result = {
                'id': groundtruth['instance_id'],
                'test_paper_name': groundtruth['test_paper_name'],
                'course': groundtruth['course'],
                'year': groundtruth['year'],
                'problem_num': groundtruth['problem_num'],
                'points': groundtruth['points'],
                'score_total': groundtruth['score_total'],
                'score_max': float(groundtruth['score_max']),
                'score_avg': float(groundtruth['score_avg']),
                'score_median': float(groundtruth['score_median']),
                'problem': groundtruth['problem'],
                'type': groundtruth['type'],
                'answer': groundtruth['answer'],
                'llm_answer': answer,
                'explanation': groundtruth['explanation'],
                'llm_explanation': explanation,
                'llm_score': int(offline_metrics['llm_score']),
                'llmjudger_explanation': offline_metrics['llmjudger_explanation'],
                'llmjudger_system_prompt': offline_metrics['llmjudger_system_prompt'],
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
            }
            results.append(result)

            logger.info('Evaluation Result:')
            logger.info(result)

        except Exception as e:
            logger.error(f"Error processing instance {groundtruth['instance_id']}: {e}")
            result = {
                'id': groundtruth['instance_id'],
                'test_paper_name': groundtruth['test_paper_name'],
                'course': groundtruth['course'],
                'year': groundtruth['year'],
                'problem_num': groundtruth['problem_num'],
                'points': groundtruth['points'],
                'score_total': groundtruth['score_total'],
                'score_max': float(groundtruth['score_max']),
                'score_avg': float(groundtruth['score_avg']),
                'score_median': float(groundtruth['score_median']),
                'problem': groundtruth['problem'],
                'type': groundtruth['type'],
                'answer': groundtruth['answer'],
                'llm_answer': None,
                'explanation': groundtruth['explanation'],
                'llm_explanation': None,
                'llm_score': 0,
                'llmjudger_explanation': None,
                'llmjudger_system_prompt': None,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'error': str(e),
            }
            results.append(result)

        with open(os.path.join(output_dir, 'result.jsonl'), 'a+', encoding='utf-8') as output_file:
            output_file.write(json.dumps(result))
            output_file.write('\n')

    spark = create_sparksession()

    data_schema = StructType(
        [
            StructField('answer', StringType(), True),
            StructField('cosine_similarity', FloatType(), True),
            StructField('course', StringType(), True),
            StructField('embeddings_similarity', FloatType(), True),
            StructField('exact_match', FloatType(), True),
            StructField('explanation', StringType(), True),
            StructField('id', LongType(), True),
            StructField('jaccard_similarity', FloatType(), True),
            StructField('llm_answer', StringType(), True),
            StructField('llm_score', IntegerType(), True),
            StructField('llm_explanation', StringType(), True),
            StructField('llmjudger_explanation', StringType(), True),
            StructField('llmjudger_system_prompt', StringType(), True),
            StructField('points', LongType(), True),
            StructField('score_total', LongType(), True),
            StructField('score_max', FloatType(), True),
            StructField('score_avg', FloatType(), True),
            StructField('score_median', FloatType(), True),
            StructField('problem', StringType(), True),
            StructField('problem_num', LongType(), True),
            StructField('system_prompt', StringType(), True),
            StructField('test_paper_name', StringType(), True),
            StructField('type', StringType(), True),
            StructField('user_prompt', StringType(), True),
            StructField('year', LongType(), True),
        ]
    )
    score_data = (
        spark.createDataFrame(results, schema=data_schema)
        .groupBy(F.lit(1))
        .agg(
            F.count('id').alias('question_count'),
            F.sum('points').alias('full_score'),
            F.sum('llm_score').alias('llm_score'),
        )
        .drop('1')
        .toPandas()
        .to_dict(orient='records')[0]
    )
    ref_data = (
        spark.read.json(input_file)
        .groupBy('test_paper_name')
        .agg(
            F.max('score_avg').alias('score_avg'),
            F.max('score_total').alias('score_total'),
            F.count('instance_id').alias('question_count'),
        )
        .groupBy(F.lit(1))
        .agg(
            F.sum('question_count').alias('question_count'),
            F.sum('score_total').alias('full_score'),
            F.sum('score_avg').alias('avg_score'),
        )
        .drop('1')
        .toPandas()
        .to_dict(orient='records')[0]
    )

    score_by_test_paper_data = (
        spark.createDataFrame(results, schema=data_schema)
        .groupBy('test_paper_name')
        .agg(
            F.count('id').alias('question_count'),
            F.sum('llm_score').alias('llm_score'),
            F.sum('points').alias('full_score'),
            F.max('score_total').alias('reference_score_total'),
            F.max('score_max').alias('reference_score_max'),
            F.max('score_avg').alias('reference_score_avg'),
            F.max('score_median').alias('reference_score_median'),
        )
        .toPandas()
        .to_dict(orient='records')
    )
    summary_data = {
        'reference': ref_data,
        'score': score_data,
        'score_by_test_paper': score_by_test_paper_data,
        'final_score': score_data['llm_score'] / float(ref_data['full_score']),
    }

    with open(os.path.join(output_dir, 'avg_score.json'), 'w', encoding='utf-8') as summary_file:
        json.dump(summary_data, summary_file, indent=4)

    logger.info('************ Final average score ************')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example benchmark')
    parser.add_argument(
        '-i',
        '--input_file',
        help='Benchmark input file',
        default='./data/benchmark/SystemTestPaper.jsonl',
    )
    parser.add_argument('-o', '--save_path', help='Result save path', default=None)
    parser.add_argument('-a', '--agent', help='Agent Name', default='llm')
    parser.add_argument(
        '-m',
        '--model_name',
        help='Model Name',
    )
    # Note that if your benchmark has multiple tasks, you need to add --task <task>
    # in your code to enable task selection.
    parser.add_argument('-t', '--task', help='specify task in scenarios', default=None)

    args = parser.parse_args()

    model_name = args.model_name
    input_file = args.input_file
    save_path = args.save_path

    if save_path is None:
        str_model_name = model_name.replace('/', '_')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = os.path.join('./outputs', f'systemcourseexam__{str_model_name}__{args.agent}__{timestamp}')

    save_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(save_path, exist_ok=True)

    main(input_file, save_path, model_name, agent_name=args.agent)

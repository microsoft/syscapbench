"""This script runs a benchmark for evaluating patches in a software project."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

if sys.version_info >= (3, 11):
    import tomllib as pytoml  # For Python 3.11+, use 'import tomllib' instead
else:
    # For Python 3.10 and earlier, use tomli
    # Note: tomli is not included in the standard library for Python 3.10
    # You may need to install it via pip: pip install tomli
    import tomli as pytoml  # For Python 3.10 and earlier, use 'import tom


from sdk.logger import logger


def read_toml_config(config_path):
    """Read configuration from a TOML file."""
    try:
        with open(config_path, 'rb') as f:  # Opening in binary mode as required by tomli
            config_data = pytoml.load(f)
        return config_data
    except Exception as e:
        logger.info(f'Error reading TOML file: {e}')
        return {}


config = read_toml_config('env.toml')

if not config:
    logger.info('Failed to load configuration from env.toml. Please check the file.')
    sys.exit(1)
logger.info('Loaded configuration:')
api_type = config.get('llm', {}).get('api_type', {})
base_url = config.get('llm', {}).get('base_url', '')
api_key = config.get('llm', {}).get('api_key', '')

logger.info('Configuration values:')
logger.info(f'API Type: {api_type}')
logger.info(f'Base URL: {base_url}')
logger.info(f'API Key: [REDACTED]')

os.environ['AZURE_API_BASE'] = base_url
os.environ['AZURE_OPENAI_API_KEY'] = api_key


from patch_evaluator import pacth_eval  # noqa: E402


def run(task_id, repo_path, problem_path, test_method, image, save_path):
    """Run the benchmark for a specific task."""
    output_dir = f'{save_path}/patch/{task_id}'
    patch_file = os.path.join(output_dir, '1c2844', '1c2844.patch')

    command = [
        'sweagent',
        'run',
        '--config',
        './src/config_aoi.yaml',
        '--env.repo.path',
        './data/benchmark/' + repo_path,
        '--problem_statement.path',
        problem_path,
        '--output_dir',
        output_dir,
        '--env.deployment.image',
        image,
        '--env.post_startup_commands',
        '["export PATH=/usr/local/go/bin:${PATH} && export HOME=/tmp"]',
    ]

    logger.info('Executing sweagent command...')
    subprocess.run(command, check=True, timout=600)

    logger.info('\n\n==========================')
    logger.info(f'Patch file expected at: {patch_file}')

    eval_out = pacth_eval(
        task_id=task_id,
        project_path=repo_path,
        patch=patch_file,
        test_method=test_method,
        output_path=output_dir,
        image=image,
    )
    logger.info('Patch evaluation completed.')

    with open(f'{save_path}/result.jsonl', 'a+', encoding='utf-8') as fw:
        fw.write(json.dumps(eval_out) + '\n')
    logger.info('Evaluation completed successfully.')


def main(file_path, save_path):
    """Main function for running the benchmark."""
    # file_path = "system_lab_tasks.jsonl"
    image = 'xuafeng/swe-go-python:latest'

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue  # Skip empty lines

            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                logger.info(f'Skipping invalid JSON line: {line}')
                continue

            task_id = task.get('task_id')
            repo_path = task.get('repo_name')
            problem_path = f'./data/benchmark/problems/{task_id}.md'
            test_method = task.get('test_method')

            run(task_id, repo_path, problem_path, test_method, image, save_path)

    success_count = 0
    total_count = 0
    with open(f'{save_path}/result.jsonl', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            if result.get('status') == 'success':
                success_count += 1
            total_count += 1
    logger.info(f'Test run completed: {success_count}/{total_count} tasks succeeded.')
    summary_data = {'score': success_count / total_count, 'total_tasks': total_count}

    with open(os.path.join(save_path, 'avg_score.json'), 'w', encoding='utf-8') as summary_file:
        json.dump(summary_data, summary_file, indent=4)


def test_run():
    """Test function to run the benchmark with a sample task."""
    run(
        task_id='test_1',
        repo_path='projects/test-repo',
        problem_path='./data/benchmark/problems/test-repo-problems/1.md',
        test_method='pip install -e . && pytest tests/test_tribonaccy.py',
        image='xuafeng/swe-go-python:latest',
        save_path='./outputs/test_run',
    )

    success_count = 0
    total_count = 0
    with open('./outputs/test_run/result.jsonl', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line.strip())
            if result.get('status') == 'success':
                success_count += 1
            total_count += 1
    logger.info(f'Test run completed: {success_count}/{total_count} tasks succeeded.')
    summary_data = {'score': success_count / total_count, 'total_tasks': total_count}

    with open('./outputs/test_run/avg_score.json', 'w', encoding='utf-8') as summary_file:
        json.dump(summary_data, summary_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example benchmark')
    parser.add_argument(
        '-i',
        '--input_file',
        help='Benchmark input file',
        # default='./data/benchmark/system_lab_tasks.jsonl',
        default='./data/benchmark/system_lab_tasks.jsonl',
    )
    parser.add_argument('-o', '--save_path', help='Result save path', default=None)
    parser.add_argument(
        '-m',
        '--model_name',
        help='Model Name',
        default='gpt-4o',
    )
    # Note that if your benchmark has multiple tasks, you need to add --task <task>
    # in your code to enable task selection.
    parser.add_argument('-t', '--task', help='specify task in scenarios', default=None)

    args = parser.parse_args()

    model_name = args.model_name
    input_file = args.input_file
    save_path = args.save_path
    task = args.task
    if task == 'test':
        logger.info('Running test benchmark...')
        test_run()
    else:
        if save_path is None:
            str_model_name = model_name.replace('/', '_')
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = os.path.join('./outputs', f'systemcourseproject__{str_model_name}__{timestamp}')

        save_path = os.path.abspath(os.path.expanduser(save_path))
        os.makedirs(save_path, exist_ok=True)

        main(input_file, save_path)

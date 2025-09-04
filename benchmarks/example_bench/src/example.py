"""Example for benchmarking the performance of a model on a specific task."""

import argparse
import json
import os
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


def read_toml_config(config_path):
    """Read configuration from a TOML file."""
    try:
        with open(config_path, 'rb') as f:  # Opening in binary mode as required by tomli
            config_data = pytoml.load(f)
        return config_data
    except Exception as e:
        print(f'Error reading TOML file: {e}')
        return {}


config = read_toml_config('env.toml')

if not config:
    print('Failed to load configuration from env.toml. Please check the file.')
    sys.exit(1)
print('Loaded configuration:')
api_type = config.get('llm', {}).get('api_type', {})
base_url = config.get('llm', {}).get('base_url', '')
api_key = config.get('llm', {}).get('api_key', '')

print('Configuration values:')
print(f'API Type: {api_type}')
print(f'Base URL: {base_url}')
print('API Key: [REDACTED]')

os.environ['OPENAI_API_TYPE'] = api_type
os.environ['OPENAI_BASE_URL'] = base_url
os.environ['OPENAI_API_KEY'] = api_key


from sdk.basic_evaluator import Evaluator  # noqa: E402
from sdk.simple_agent import SimpleAgent  # noqa: E402


def main(_input_file, output_dir, _model_name):
    """Main function for running the benchmark."""
    total_score = []
    with (
        open(_input_file, encoding='utf-8') as data,
        open(os.path.join(output_dir, 'result.jsonl'), 'w', encoding='utf-8') as output_file,
    ):
        for line in data:
            item = json.loads(line)
            print('============ ' + item['id'] + ' ============')
            example = SimpleAgent(_model_name, item['sys_prompt'])
            response = example.run(item['user_prompt'])
            evaluator = Evaluator(_model_name)
            offline_metrics = evaluator.offline_evaluate(
                question=item['user_prompt'], answer=response, groundtruth=item
            )

            total_score.append(offline_metrics[:6])  # drop llmjudger_answer
            result = {
                'id': item['id'],
                'sys_prompt': item['sys_prompt'],
                'user_prompt': item['user_prompt'],
                'groundtruth': item['response'],
                'response': response,
                'syntax_acc': offline_metrics[0],
                'exact_match': offline_metrics[1],
                'jaccard_similarity': offline_metrics[2],
                'cosine_similarity': offline_metrics[3],
                'embeddings_similarity': offline_metrics[4],
                'llmjudger_rating': offline_metrics[5],
                'llmjudger_answer': offline_metrics[6],
            }
            print('Evaluation Result:')
            print(result)
            output_file.write(json.dumps(result))
            output_file.write('\n')

    avg_score = [sum(values) / len(values) for values in list(zip(*total_score))]
    avg_score_dict = {
        'syntax_acc': avg_score[0],
        'exact_match': avg_score[1],
        'jaccard_similarity': avg_score[2],
        'cosine_similarity': avg_score[3],
        'embeddings_similarity': avg_score[4],
        'llmjudger_rating': avg_score[5],
    }
    with open(os.path.join(output_dir, 'avg_score.json'), 'w', encoding='utf-8') as avg_score_file:
        json.dump(avg_score_dict, avg_score_file, indent=4)
    print('************ Final average score ************')
    print(avg_score_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='example benchmark')
    parser.add_argument(
        '-i',
        '--input_file',
        help='Benchmark input file',
        default='./data/benchmark/example_bench_benchmark_timestamp.jsonl',
    )
    parser.add_argument('-o', '--save_path', help='Result save path', default=None)
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
        save_path = os.path.join('./outputs', f'examplebench__{str_model_name}__{timestamp}')

    save_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(save_path, exist_ok=True)

    main(input_file, save_path, model_name)

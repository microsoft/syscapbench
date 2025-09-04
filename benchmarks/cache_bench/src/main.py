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
print(f'API Key: {api_key}')

os.environ['OPENAI_API_TYPE'] = api_type
os.environ['OPENAI_BASE_URL'] = base_url
os.environ['OPENAI_API_KEY'] = api_key

from cache_simulator.run_simulatorcache import simulate  # noqa: E402

from sdk.simple_executor import SimpleExecutor  # noqa: E402


def run_simulate_on_traces(response, trace_path):
    """Run the cache simulation on all traces in the specified path."""
    all_traces = list_all_files(trace_path)
    print('all_traces:', all_traces)

    all_time_cost = []
    all_miss_rate = []
    for trace in all_traces:
        time_cost, miss_rate = simulate(response, trace_path=trace)
        all_time_cost.append(time_cost)
        all_miss_rate.append(miss_rate)

    time_cost = sum(all_time_cost) / len(all_time_cost)
    miss_rate = sum(all_miss_rate) / len(all_miss_rate)
    return time_cost, miss_rate


def list_all_files(folder_path):
    """List all files in the given folder and its subfolders."""
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


def main(_input_file, output_dir, _model_name, trace_path=None):
    """Main function for running the benchmark."""
    total_score = []
    with (
        open(_input_file, encoding='utf-8') as data,
        open(os.path.join(output_dir, 'result.jsonl'), 'w', encoding='utf-8') as output_file,
    ):
        for line in data:
            item = json.loads(line)
            print('============ ' + item['id'] + ' ============')
            example = SimpleExecutor(_model_name, item['sys_prompt'])
            response = example.run(item['user_prompt'], lang='python')
            print('Response from model:')
            print(response)

            all_round_results = []
            for round in range(3):
                print(f'Round {round + 1} of evaluation...')
                # Simulate the cache algorithm
                print('Simulating cache algorithm...')

                time_cost, miss_rate = run_simulate_on_traces(response, trace_path=trace_path)
                if not miss_rate:
                    print('Simulation failed, please refine your response.')
                    continue
                print(f'Time Cost: {time_cost:.6f} seconds')
                print(f'Miss Rate: {miss_rate:.6f}')

                all_round_results.append((time_cost, miss_rate))

                refine_prompt = f"""
                The cache algorithm has been simulated with the following results:
                - Average Time Cost: {time_cost:.6f} seconds
                - Average Miss Rate: {miss_rate:.6f}
                Please refine your policy based on these results to reduce the miss_rate.
                """
                response = example.run(refine_prompt, lang='python')

            print('Final Response after refinement:')
            time_cost, miss_rate = run_simulate_on_traces(response, trace_path=trace_path)
            all_round_results.append((time_cost, miss_rate))
            print(f'Final Time Cost: {time_cost:.6f} seconds')
            print(f'Final Miss Rate: {miss_rate:.6f}')

            print('All rounds results:')
            for idx, (time_cost, miss_rate) in enumerate(all_round_results):
                print(f'Round {idx + 1}: Time Cost = {time_cost:.6f} seconds, Miss Rate = {miss_rate:.6f}')
            # evaluator = Evaluator(_model_name)
            # offline_metrics = evaluator.offline_evaluate(question=item['user_prompt'], query=response, groundtruth=item)

            total_score.append([miss_rate, time_cost])  # drop llmjudger_answer
            result = {
                'id': item['id'],
                'sys_prompt': item['sys_prompt'],
                'user_prompt': item['user_prompt'],
                'response': response,
                'miss_rate': miss_rate,
                'time_cost': time_cost,
            }
            print('Evaluation Result:')
            print(result)
            output_file.write(json.dumps(result))
            output_file.write('\n')

    avg_score = [sum(values) / len(values) for values in list(zip(*total_score))]
    avg_score_dict = {
        'miss_rate': avg_score[0],
        'time_cost': avg_score[1],
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
        default='./data/benchmark/cache_benchmarks.jsonl',
    )
    parser.add_argument('-o', '--save_path', help='Result save path', default=None)
    parser.add_argument(
        '-m',
        '--model_name',
        help='Model Name',
    )
    # Note that if your benchmark has multiple tasks, you need to add --task <task>
    # in your code to enable task selection.
    parser.add_argument(
        '-t',
        '--task',
        help='specify task in scenarios',
        choices=['alibaba-storage', 'ra-fwe', 'ra-multikey', 'tencentblock-storage', 'tmp', 'zipf'],
        default='tmp',
    )

    args = parser.parse_args()

    model_name = args.model_name
    input_file = args.input_file
    save_path = args.save_path

    trace_path = f'./data/benchmark/trace/{args.task}' if args.task else None

    if save_path is None:
        str_model_name = model_name.replace('/', '_')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = os.path.join('./outputs', f'cachebench__{str_model_name}_{args.task}__{timestamp}')

    save_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(save_path, exist_ok=True)

    main(input_file, save_path, model_name, trace_path=trace_path)

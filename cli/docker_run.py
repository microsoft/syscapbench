"""Run a benchmark in a local deployment using swerex."""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from logger import logger
from swerex.deployment.docker import DockerDeployment
from swerex.deployment.local import LocalDeployment
from swerex.runtime.abstract import BashAction, Command, CreateBashSessionRequest, UploadRequest

if sys.version_info >= (3, 11):
    import tomllib as pytoml  # For Python 3.11+, use 'import tomllib' instead
else:
    # For Python 3.10 and earlier, use tomli
    # Note: tomli is not included in the standard library for Python 3.10
    # You may need to install it via pip: pip install tomli
    import tomli as pytoml


def read_toml_config(config_path):
    """Read configuration from a TOML file."""
    try:
        with open(config_path, 'rb') as f:  # Opening in binary mode as required by tomli
            config_data = pytoml.load(f)
        return config_data
    except Exception as e:
        logger.info(f'Error reading TOML file: {e}')
        return {}


def write_to_file(file_path, content):
    """Write content to a file."""
    with open(file_path, 'w') as f:
        f.write(content)


async def run_some_stuff(code_path, model_name, deployment, save_path):
    """Spoiler: This function will work with any deployment."""
    await deployment.start()
    runtime = deployment.runtime

    # Issue a few one-off commands, similar to `subprocess.run()`
    logger.info(await runtime.execute(Command(command=['echo', 'Hello, world!'])))

    # Create a bash session
    await runtime.create_session(CreateBashSessionRequest())

    # Run a command in the session
    # The difference to the one-off commands is that environment state persists!
    logger.info(await runtime.run_in_session(BashAction(command="export MYVAR='test'")))
    logger.info(await runtime.run_in_session(BashAction(command='echo $MYVAR')))

    logger.info(
        await runtime.upload(
            UploadRequest(
                source_path='../',
                target_path='./bench',
            )
        )
    )

    logger.info(await runtime.run_in_session(BashAction(command='ls ./bench')))
    logger.info(await runtime.run_in_session(BashAction(command='cd ./bench/benchmarks/' + code_path + ' && ls')))
    logger.info(await runtime.run_in_session(BashAction(command='rm -rf outputs/*/')))

    logger.info('Running install script...')
    logger.info(await runtime.run_in_session(BashAction(command='chmod +x *.sh && ./install.sh')))
    logger.info(await runtime.run_in_session(BashAction(command='ls -l')))

    logger.info('Running benchmark script...')
    logger.info(await runtime.run_in_session(BashAction(command=f'./run.sh {model_name}')))
    logger.info('Benchmark script finished.')

    logger.info('Copying outputs to save path...')
    a = await runtime.run_in_session(BashAction(command='cat outputs/*/avg_score.json'))
    output_file = os.path.join(save_path, 'avg_score.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_to_file(output_file, a.output if hasattr(a, 'output') else str(a))
    logger.info(f'Output saved to: {output_file}')

    b = await runtime.run_in_session(BashAction(command='cat outputs/*/result.jsonl'))
    output_file_s = os.path.join(save_path, 'result.jsonl')
    os.makedirs(os.path.dirname(output_file_s), exist_ok=True)
    write_to_file(output_file_s, b.output if hasattr(b, 'output') else str(b))
    logger.info(f'Output saved to: {output_file_s}')

    await deployment.stop()


def run_bench(benchmark_path, model_name, image, save_path):
    """Evaluate a patch by running a test method in a deployment."""
    # deployment = LocalDeployment()
    if image == 'multiple':
        deployment = LocalDeployment()
    else:
        deployment = DockerDeployment(image=image, docker_args=['--privileged'])
    eval_out = asyncio.run(run_some_stuff(benchmark_path, model_name, deployment, save_path))
    return eval_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run some stuff in a deployment.')
    parser.add_argument('--benchmark_name', type=str, required=True, help='Task ID')
    parser.add_argument('--model_name', type=str, default='gpt-4o', help='Model Name')

    # Parse the arguments
    args = parser.parse_args()
    benchmark_name = args.benchmark_name
    model_name = args.model_name
    logger.info(f'Running benchmark: {benchmark_name} with model: {model_name}')

    config = read_toml_config(f'../benchmarks/{benchmark_name}/env.toml')

    if not config:
        logger.info('Failed to load configuration from env.toml. Please check the file.')
        sys.exit(1)

    use_gpu = config.get('hardware', {}).get('use_gpu', False)
    if use_gpu:
        # TODO: Implement GPU support
        logger.info('GPU support is not implemented yet.')
        sys.exit(1)

    docker_image = config.get('env-docker', {}).get('image', '')
    if docker_image == 'default':
        docker_image = 'xuafeng/swe-go-python:latest'
    entry_script = config.get('env-docker', {}).get('entrypoint', '')

    logger.info(f'Using GPU: {use_gpu}')
    logger.info(f'Docker Image: {docker_image}')
    logger.info(f'Entry Script: {entry_script}')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_bench(
        benchmark_path=benchmark_name,
        model_name=model_name,
        image=docker_image,
        save_path=f'./outputs/{benchmark_name}_{model_name}_{timestamp}/',
    )

import os
import sys

from sdk.logger import logger

if sys.version_info >= (3, 11):
    import tomllib as pytoml  # For Python 3.11+, use 'import tomllib' instead
else:
    # For Python 3.10 and earlier, use tomli
    # Note: tomli is not included in the standard library for Python 3.10
    # You may need to install it via pip: pip install tomli
    import tomli as pytoml  # For Python 3.10 and earlier, use 'import tomli'


def read_toml_config(config_path):
    """Read configuration from a TOML file."""
    try:
        with open(config_path, 'rb') as f:  # Opening in binary mode as required by tomli
            config_data = pytoml.load(f)
        return config_data
    except Exception as e:
        logger.info('Error reading TOML file: %s', e)
        return {}


def set_llm_endpoint_from_config(config_path):
    """Set LLM endpoint environment variables from the configuration dictionary."""
    logger.info('Read configuration from %s', config_path)
    config = read_toml_config(config_path)

    if not config:
        logger.info('Failed to load configuration from env.toml. Please check the file.')
        sys.exit(1)
    logger.info('Loaded configuration:')

    # read all vaules in the llm section and set them as environment variables
    llm_config = config.get('llm', {})

    logger.info('Setting the following environment variables:')
    for key, value in llm_config.items():
        logger.info('%s', f'{key}: [REDACTED]' if 'key' in key.lower() else f'{key}: {value}')
        os.environ[key] = value
        # add exception for SWE-Agent:
        if key == 'AZURE_API_KEY':
            os.environ['AZURE_OPENAI_API_KEY'] = value
            logger.info('AZURE_OPENAI_API_KEY: [REDACTED]')

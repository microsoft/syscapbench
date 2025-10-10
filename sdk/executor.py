"""Base class for LLM-based executors."""

from sdk.llm import LLM
from sdk.logger import logger


class Executor:
    """Base class for LLM-based executors."""

    def __init__(self, _model_name, _sys_prompt) -> None:
        """Initialize the Executor class."""
        self.system_prompt = _sys_prompt
        self.model_name = _model_name

    def run(self, nl, lang=''):
        """Run the executor - to be overridden by subclasses."""
        raise NotImplementedError('Subclasses must implement the run method')


class SimpleExecutor(Executor):
    """Example class for one simple LLM module."""

    def __init__(self, _model_name, _sys_prompt) -> None:
        """Initialize the Example class."""
        self.system_prompt = _sys_prompt
        self.model_name = _model_name
        self.LLM = LLM(engine=self.model_name, system_prompt=self.system_prompt, temperature=0.1)

    def extract_code(self, text: str, lang: str = '') -> str:
        """Extract code from the text."""
        start, end = f'```{lang}', '```'
        code = text[text.find(start) + len(start) : text.find(end, text.find(start) + 1)].strip()
        return code.strip()

    def run(self, nl, lang=''):
        """Run the Example class."""
        ans = self.LLM.query(nl)
        kql = self.extract_code(ans, lang=lang)
        logger.info('Extracted Structed Code:')
        logger.info('%s', kql)
        return kql

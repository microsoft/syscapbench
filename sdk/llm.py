"""LLM class for OpenAI API."""

import os
import pprint
import sys
import time

from openai import AzureOpenAI, OpenAI

OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE', 'azure')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION', '2024-05-01-preview')

print(f'Using OpenAI API type: {OPENAI_API_TYPE}')
print(f'Using OpenAI base URL: {OPENAI_BASE_URL}')
print(f'Using OpenAI API key: [REDACTED]')


class LLM:
    """LLM class for OpenAI API."""

    def __init__(
        self,
        engine,
        system_prompt: str = 'You are an AI assistant that helps people find information.',
        temperature: float = 0.1,
        past_message_num: int = sys.maxsize,
        json_format: bool = False,
    ) -> None:
        """Initialize the LLM class."""
        self.name = engine
        self.system_prompt = system_prompt
        self.past_message_num = max(0, past_message_num)
        self.messages = [
            {
                'role': 'system',
                'content': self.system_prompt,
            }
        ]
        self.parameters = {
            'model': engine,
            'temperature': temperature,
            'max_tokens': 2000,
            'top_p': 0.95,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'stop': None,
        }
        if engine == 'o4-mini':
            self.parameters = {
                'model': engine,
                'temperature': 1,
                'max_completion_tokens': 2000,
                # "top_p": 0.95,
                # "frequency_penalty": 0,
                # "presence_penalty": 0,
                # "stop": None,
            }
        if json_format is True:
            self.parameters['response_format'] = {'type': 'json_object'}

        if OPENAI_API_TYPE == 'azure':
            self._openai = AzureOpenAI(
                azure_endpoint=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                api_version=OPENAI_API_VERSION,
            )
        else:
            self._openai = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def reset(self) -> None:
        """Reset the LLM class."""
        self.update_messages(reset=True)

    def update_messages(self, reset=False) -> None:
        """Update the messages list."""
        if self.past_message_num > 0 and not reset:
            self.messages = [self.messages[0]] + self.messages[1:][-self.past_message_num :]
        else:
            self.messages = [self.messages[0]]

    def query(self, user_prompt: str) -> str:
        """Query the LLM with the user prompt."""
        self.messages.append(
            {
                'role': 'user',
                'content': user_prompt,
            }
        )
        print(f'Start to query {self.name} with the following prompts:')
        print(pprint.pformat(self.messages[-2:], width=120, compact=True))

        ans, timeout = '', 2
        retry = 4
        while not ans and retry > 0:
            try:
                retry -= 1
                time.sleep(timeout)
                response = self._openai.chat.completions.create(messages=self.messages, **self.parameters)
                ans = response.choices[0].message.content
            except Exception as e:
                print(e)
            if not ans:
                timeout = timeout + 1 if timeout < 5 else timeout * 2
                print(f'Will retry after {timeout} seconds ...')

        print(f'Query {self.name} finished with the following answer:')
        print(ans)

        self.messages.append(
            {
                'role': 'assistant',
                'content': ans,
            }
        )
        self.update_messages()
        return ans

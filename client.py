import traceback

import openai

from config import config
from schemas import Prompt


class VLLMClient:
    def __init__(self):
        self.model = config.MODEL_NAME
        self.client = openai.OpenAI(
            api_key=config.VLLM_API_KEY,
            base_url=config.VLLM_BASE_URL,
            timeout=config.VLLM_TIME_OUT,
        )

    def send(self, prompt: Prompt) -> openai.types.Completion:
        history = [h.model_dump() for h in prompt.history]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    *history,
                    {"role": "user", "content": prompt.question},
                ],
                temperature=0,
            )
            return response

        except Exception as e:
            print(f"Exception occurred: {e}")
            print(f"Input prompt: {prompt.model_dump()}")
            print(traceback.format_exc())

        return

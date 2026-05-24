import os
import traceback
import concurrent.futures

import time
from tqdm import tqdm
from typing import List

from eval.config import config
from eval.client import VLLMClient
from eval.prompt import SYSTEM_PROMPT, INPUT_PROMPT
from eval.dataset import load_json, save_json
from eval.schemas import Prompt, LLMResult, Token


def _llm_response(client: VLLMClient, prompt: Prompt, results: List[LLMResult]) -> bool:
    try:
        t0 = time.perf_counter()
        response = client.send(prompt=prompt)
        t1 = time.perf_counter()

        completion = response.choices[0].message.content
        token = response.usage.to_dict()
        results.append(
            LLMResult(
                prompt=prompt,
                completion=completion,
                token=Token(
                    prompt=token["prompt_tokens"],
                    completion=token["completion_tokens"],
                    total=token["total_tokens"],
                ),
                latency=t1 - t0,
            )
        )
        return True

    except Exception as e:
        print(f"Error occurred {str(e)} on prompt: {prompt.model_dump()}")
        print(f"{traceback.format_exc()}")

    return False


def main():
    client = VLLMClient(config)
    data = load_json(datapath=config.BASE_INPUT)

    prompts: List[Prompt] = []
    for d in data:
        d["system_prompt"] = SYSTEM_PROMPT
        d["input_prompt"] = INPUT_PROMPT.format(
            contexts="\n".join(f" - {c['content']}" for c in d["retrieved_contexts"]),
            user_input=d["user_input"],
        )
        prompts.append(Prompt.model_validate(d))

    succeeds = 0
    results: List[LLMResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.MAX_NUM_SEQS
    ) as executor:
        futures = [
            executor.submit(_llm_response, client, prompt, results)
            for prompt in prompts
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(prompts)
        ):
            if future.result():
                succeeds += 1

    output_dir = os.path.join("outputs", config.MODEL_NAME)
    save_json(
        datapath=os.path.join(output_dir, "config.json"),
        data=config.model_dump(),
    )
    save_json(
        datapath=os.path.join(output_dir, "results.json"),
        data=[result.model_dump() for result in results],
    )
    return


if __name__ == "__main__":
    main()

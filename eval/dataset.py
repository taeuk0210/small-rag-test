import os
import json
from typing import Dict, Any

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

from eval.schemas import LLMResult


def load_json(datapath: str) -> Dict[str, Any]:
    with open(datapath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(datapath: str, data: Dict[str, Any]) -> None:
    dirname = os.path.dirname(datapath)
    os.makedirs(dirname, exist_ok=True)
    with open(datapath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return


def load_dataset(datapath: str) -> EvaluationDataset:
    results = load_json(datapath=datapath)
    results = [LLMResult.model_validate(r) for r in results]

    samples = [
        SingleTurnSample(
            user_input=result.prompt.user_input,
            retrieved_contexts=[c.content for c in result.prompt.retrieved_contexts],
            response=result.completion,
            reference=result.prompt.reference,
        ) for result in results
    ]

    dataset = EvaluationDataset(samples=samples)

    return dataset

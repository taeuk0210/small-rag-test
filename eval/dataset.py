import json

from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage


def load_dataset() -> EvaluationDataset:
    with open("samples/001.json", "r", encoding="utf-8") as f:
        sample = json.load(f)
        
    samples = [
        SingleTurnSample(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
            reference=sample["reference"],
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    return dataset

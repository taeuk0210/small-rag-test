import json
import asyncio

from google import genai
from ragas import aevaluate, EvaluationDataset
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import (
    _AnswerCorrectness,
    _ContextPrecision,
    _ContextRecall,
    _Faithfulness,
)
from ragas.dataset_schema import SingleTurnSample

from config import config


async def main():
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    evaluator_llm = llm
    evaluator_embeddings = embeddings

    with open("sample.json", "r", encoding="utf-8") as f:
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
    metrics = [
        _ContextPrecision(llm=evaluator_llm),
        _ContextRecall(llm=evaluator_llm),
        _Faithfulness(llm=evaluator_llm),
        _AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    results = await aevaluate(
        dataset=dataset,
        metrics=metrics,
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio

from google import genai
from ragas import aevaluate
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import (
    _AnswerCorrectness,
    _ContextPrecision,
    _ContextRecall,
    _Faithfulness,
)

from eval.config import config
from eval.dataset import load_dataset


async def main():
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    dataset = load_dataset()
    metrics = [
        _ContextPrecision(llm=llm),
        _ContextRecall(llm=llm),
        _Faithfulness(llm=llm),
        _AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    results = await aevaluate(
        dataset=dataset,
        metrics=metrics,
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

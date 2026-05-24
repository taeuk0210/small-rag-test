import asyncio

from google import genai
from ragas import aevaluate
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import (
    _RougeScore,
    _Faithfulness,
    _AnswerRelevancy,
    _AnswerSimilarity,
    _AnswerCorrectness,
    _ContextEntityRecall,
)

from eval.config import config
from eval.dataset import load_dataset


async def main():
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    dataset = load_dataset()
    metrics = [
        _RougeScore(),
        _Faithfulness(llm=llm),
        _AnswerRelevancy(llm=llm, embeddings=embeddings),
        _AnswerSimilarity(embeddings=embeddings),
        _AnswerCorrectness(llm=llm, embeddings=embeddings),
        _ContextEntityRecall(llm=llm),
    ]

    results = await aevaluate(
        dataset=dataset,
        metrics=metrics,
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())

import os
import asyncio
from argparse import ArgumentParser

from google import genai
from ragas import aevaluate
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics import (
    # _RougeScore,
    # _Faithfulness,
    # _AnswerRelevancy,
    # _AnswerSimilarity,
    _AnswerCorrectness,
    # _ContextEntityRecall,
)
from ragas.run_config import RunConfig

from eval.config import config
from eval.dataset import load_dataset


async def main(datapath: str):
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    llm = llm_factory("gemini-2.5-flash", provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")

    dataset = load_dataset(datapath=datapath)
    metrics = [
        _AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]
    run_config = RunConfig(
        timeout=120,
        max_workers=2,
        max_retries=3,
    )

    results = await aevaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )
    df = results.to_pandas()
    dirname = os.path.dirname(datapath)
    df.to_csv(os.path.join(dirname, "evaluation.csv"), index=False, encoding="utf-8")
    return 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", "-d", type=str, required=True, help="Input result.json file path.")
    args = parser.parse_args()
    asyncio.run(main(args.datapath))

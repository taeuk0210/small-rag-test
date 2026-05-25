import os
import asyncio
from argparse import ArgumentParser
import pandas as pd

from google import genai
from ragas import aevaluate
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (
    CHRFScore,
    BleuScore,
    ExactMatch,
    RougeScore
)
from ragas.run_config import RunConfig

from eval.config import config
from eval.dataset import load_json


async def main(datapath: str):
    data = load_json(datapath=datapath)
    scorers = {
        # "bleu": BleuScore(),
        "chrf": CHRFScore(),
        # "match": ExactMatch(),
        "rouge": RougeScore(rouge_type="rougeL", mode="fmeasure"),
    }

    rows = []
    for d in data:
        row = d["prompt"]["meta"].copy()
        reference = d["prompt"]["reference"]
        response = d["completion"]

        for k in scorers:
            row.update({
                k: (await scorers[k].ascore(
                    reference=reference,
                    response=response,
                )).value,
            })
        rows.append(row)
        
    df = pd.DataFrame(rows)
    dirname = os.path.dirname(datapath)
    df.to_csv(os.path.join(dirname, "eval_wo_llm.csv"), index=False, encoding="utf-8")
    return 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datapath", "-d", type=str, required=True, help="Input result.json file path.")
    args = parser.parse_args()
    asyncio.run(main(args.datapath))

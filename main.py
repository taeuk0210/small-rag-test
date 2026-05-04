import os
import json
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    _faithfulness,
    _answer_relevancy,
    _context_recall,
    _context_precision,
)

from client import VLLMClient
from config import config
from schemas import Prompt


def main():
    with open("sample.json", "r", encoding="utf-8") as f:
        sample = json.load(f)

    # prompt = Prompt.model_validate(sample)
    # prompt.system = f"""### 가이드라인
    # 너는 사내 업무 지원 전문 어시스턴트야.

    # 아래의 검색 결과를 활용해서 규정을 잘 지키도록 사용자의 질문에 답변해봐.

    # ### 검색 결과
    # {"\n\n".join([f"결과 {i}:\n{context}" for i, context in enumerate(prompt.contexts)])}"""

    # client = VLLMClient()

    # response = client.send(prompt=prompt)

    # sample["answer"] = response.choices[0].message.content
    # with open("sample.json", "w", encoding="utf-8") as f:
    #     json.dump(sample, f, ensure_ascii=False, indent=4)

    dataset = Dataset.from_dict(
        {
            "question": [sample["question"]],
            "answer": [sample["answer"]],
            "contexts": [sample["contexts"]],
            "ground_truth": [sample["ground_truth"]],
        }
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=config.GOOGLE_API_KEY,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=config.GOOGLE_API_KEY,
    )
    metrics = [_context_precision, _context_recall, _faithfulness, _answer_relevancy]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    print(result)
    # >> {'context_precision': 1.0000, 'context_recall': 1.0000, 'faithfulness': nan, 'answer_relevancy': nan}
    return


if __name__ == "__main__":
    main()

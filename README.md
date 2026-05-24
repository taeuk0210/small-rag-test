# small-rag-test
Performance test of rag with small llm by ragas framework.

### .env example

```bash
BASE_INPUT=inputs/base.json

MODEL_NAME=Qwen/Qwen3.5-0.8B

GPU_MEMORY_UTIL=0.7

MAX_MODEL_LEN=2048

MAX_NUM_SEQS=1

# QUANTIZATION=awq

DTYPE=auto

# KV_CACHE_DTYPE=fp8

VLLM_PORT=8000 

VLLM_API_KEY=EMPTY

VLLM_TIME_OUT=300

VLLM_KV_SPACE=2

VLLM_BASE_URL=http://localhost:${VLLM_PORT}/v1

GOOGLE_API_KEY=<GOOGLE_API_KEY>

COMPOSE_PROFILES=cpu
```

### RAGAS Metrics

**LLM-based**  
- Answer Correctness
- Answer Relevancy
- Faithfulness
- Context Entity Recall
- Answer Semantic Similarity

**non-LLM based**
- Keyword Exact Match
- ROUGE-L
- Format Succees Rate
- Verbosity & Repetition

### Docker CPU Image

```bash
git clone https://github.com/vllm-project/vllm.git

cd vllm

wsl 

docker build -f docker/Dockerfile.cpu \
        # --build-arg max_jobs=4 \
        # --build-arg VLLM_CPU_X86=false \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

메모리 터지면 .wslconfig 수정 후 wsl 재시작 후 재빌드

```bash
[wsl2]
memory=8GB
swap=4GB
processors=4
```


### 전체 파이프라인

**샘플 데이터 요약**  

| 구분 | 내용 | 비고 |  
| --- | ---- | --- |  
| 컨텍스트 | 컨텍스트 요약 유무 |  |  
| 컨텍스트 | 컨텍스트 청크 수 조절 | 3 ~ 6 |  
| 컨텍스트 | 노이즈 컨텍스트 유무 |  |  
| 컨텍스트 | 랭킹 조절 |  |  
| 히스토리 | 대화 턴 수 조절 | 3 ~ 6 |  
| 기타 | 추론 형식 강제 | JSON 등 |  
| 기타 | 할루시네이션 방어 |  |  
  
1. `user_input`, `retrieved_contexts`, `reference`(ground_truth) 데이터셋 생성
     - 도메인은 업무 보조, 법령 검색을 예시로 생성
     - 이전 대화 맥락(`history`) 싱글턴/멀티턴 대화를 구분 -> 멀티턴의 경우 문맥 유지력을 관찰해야하니까 대화 내용 설계를 잘 해야 함
     - `retrieved_contexts`가 요약된 경우/아닌 경우 구분
     - `retrieved_contexts`가 수를 비교할 수 있도록 구분(Top 2 ~ 7 예상)
     - `retrieved_contexts`에 답변과 관련 없는 노이즈 작성
     - `retrieved_contexts`에서 순위를 조정해보기(안정성 측정을 위해)
     - `reference`가 현재는 알 수 없다는 답변인 것도 있어야 함

2. LLM 별로 `response` 저장  
     - vLLM 으로 보낼 LLM 프롬프트를 설계 후 모델별 돌려쓸지 말지 등등 선택하기
3. `user_input`, `retrieved_contexts`, `reference`, `response` 활용해서 지표 측정

4. 모델별 지표 비교 및 한계점/특징/인사이트 도출(가능하면)
   - main.py 를 기준으로 시작할텐데 적당히 가독성있게 모듈별로 분리해서 설계(HACCP 때 쓸 수 있도록)

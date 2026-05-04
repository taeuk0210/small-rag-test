# small-rag-test
Performance test of rag with small llm by ragas framework.

### .env example

```bash
MODEL_NAME=google/gemma-4-E2B-it

GPU_MEMORY_UTIL=0.8

MAX_MODEL_LEN=12298

MAX_NUM_SEQS=10

QUANTIZATION=awq

DTYPE=auto

KV_CACHE_DTYPE=fp8

VLLM_PORT=8000 

VLLM_API_KEY=EMPTY

VLLM_TIME_OUT=300

VLLM_BASE_URL=http://localhost:${VLLM_PORT}/v1

GOOGLE_API_KEY=<MY-GOOGLE-API-KEY>
```


### Docker CPU Image

```bash
git clone https://github.com/vllm-project/vllm.git

cd vllm

wsl

docker build -f docker/Dockerfile.cpu \
        # --build-arg VLLM_CPU_X86=<false (default)|true> \ # For cross-compilation
        --tag vllm-cpu-env \
        --target vllm-openai .
```
만약 빌드시 메모리 터진다면 로그 확인후 아래 작업 수행

빌드 명령어에 --build-arg CMAKE_BUILD_PARALLEL_LEVEL=4 추가 -> 난 여기서 해결됨

안되면 추가로 --build-arg MAX_JOBS=4 추가

안되면 docker/Dockerfile.cpu 에서 max_jobs=4로 변경

안되면 .wslconfig 수정 후 wsl 재시작 후 재빌드
```bash
[wsl2]
memory=10GB
swap=8GB
processors=6
```
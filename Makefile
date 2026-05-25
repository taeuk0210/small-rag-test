include .env
export

CUR_DATE := $(shell date +%Y%m%d)
CUR_TIME := $(shell date +%H%M%S)

MAIN_LOG_DIR := experiments/$(CUR_DATE)/$(CUR_TIME)
MAIN_LOG_FILE := $(MAIN_LOG_DIR)/log.log

VLLM_LOG_DIR := logs
VLLM_LOG_FILE := $(VLLM_LOG_DIR)/vllm_$(CUR_DATE)_$(CUR_TIME).log

QUANTIZATION_ARG = $(if $(QUANTIZATION),--quantization $(QUANTIZATION),)

v: 
	python3 -m vllm.entrypoints.openai.api_server \
		--port $(VLLM_PORT) \
		--model $(MODEL_NAME) \
		--max-model-len $(MAX_MODEL_LEN) \
		--max-num-seqs $(MAX_NUM_SEQS) \
		--gpu-memory-utilization $(GPU_MEMORY_UTIL) \
		--dtype $(DTYPE) \
		--kv-cache-dtype $(KV_CACHE_DTYPE)

g:
	python3 -m eval.generate

e:
	@nohup sh -c "\
		python3 -m eval.evaluation --datapath outputs/google/gemma-3-1b-it/results.json && \
		python3 -m eval.evaluation --datapath outputs/google/gemma-4-E2B-it/results.json && \
		python3 -m eval.evaluation --datapath outputs/meta-llama/Llama-3.2-3B-Instruct/results.json && \
		python3 -m eval.evaluation --datapath outputs/mistralai/Ministral-3-3B-Instruct-2512/results.json && \
		python3 -m eval.evaluation --datapath outputs/Qwen/Qwen2.5-0.5B-Instruct/results.json && \
		python3 -m eval.evaluation --datapath outputs/Qwen/Qwen3.5-2B/results.json \
	" > eval.log 2>&1 &
	

e2:
	@nohup sh -c "\
		python3 -m eval.eval --datapath outputs/google/gemma-3-1b-it/results.json && \
		python3 -m eval.eval --datapath outputs/google/gemma-4-E2B-it/results.json && \
		python3 -m eval.eval --datapath outputs/meta-llama/Llama-3.2-3B-Instruct/results.json && \
		python3 -m eval.eval --datapath outputs/mistralai/Ministral-3-3B-Instruct-2512/results.json && \
		python3 -m eval.eval --datapath outputs/Qwen/Qwen2.5-0.5B-Instruct/results.json && \
		python3 -m eval.eval --datapath outputs/Qwen/Qwen3.5-2B/results.json \
	" > eval_wo_llm.log 2>&1 &
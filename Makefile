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
	@mkdir -p $(VLLM_LOG_DIR)
	@nohup python3 -m vllm.entrypoints.openai.api_server \
		--port $(VLLM_PORT) \
		--model $(MODEL_NAME) \
		--max-model-len $(MAX_MODEL_LEN) \
		--max-num-seqs $(MAX_NUM_SEQS) \
		--gpu-memory-utilization $(GPU_MEMORY_UTIL) \
		--dtype $(DTYPE) \
		--kv-cache-dtype $(KV_CACHE_DTYPE) \
		$(QUANTIZATION_ARG) > $(VLLM_LOG_FILE) 2>&1 &

m:
	@mkdir -p $(MAIN_LOG_DIR)
	@nohup python3 -m src.main \
		--date $(CUR_DATE) \
		--time $(CUR_TIME) \
		--data $(DATA_DIR) \
		--importdir $(IMPORT_DIR) \
		--exportdir $(EXPORT_DIR) \
		--num-iterations $(NUM_ITERATIONS) \
		--num-workers $(NUM_WORKERS) > $(MAIN_LOG_FILE) 2>&1 &
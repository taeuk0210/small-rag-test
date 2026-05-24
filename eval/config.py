from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    MODEL_NAME: str
    BASE_INPUT: str
    GPU_MEMORY_UTIL: float
    MAX_MODEL_LEN: int
    MAX_NUM_SEQS: int
    QUANTIZATION: str = ""
    DTYPE: str
    KV_CACHE_DTYPE: str = ""
    VLLM_PORT: int
    VLLM_API_KEY: str
    VLLM_TIME_OUT: int
    VLLM_KV_SPACE: int
    VLLM_BASE_URL: str
    GOOGLE_API_KEY: str
    COMPOSE_PROFILES: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


config = Config()

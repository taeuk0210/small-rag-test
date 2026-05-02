from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    MODEL_NAME: str
    GPU_MEMORY_UTIL: float
    MAX_MODEL_LEN: int
    MAX_NUM_SEQS: int
    QUANTIZATION: str = ""
    DTYPE: str
    KV_CACHE_DTYPE: str
    VLLM_BASE_URL: str
    VLLM_TIME_OUT: float
    VLLM_API_KEY: str
    GOOGLE_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


config = Config()

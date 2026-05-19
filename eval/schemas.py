from typing import List
from pydantic import BaseModel


class Meta(BaseModel):
    topic: str
    variant: str
    chunks: int
    turns: int


class Context(BaseModel):
    rank: int
    content: str


class Turn(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    meta: Meta
    system_prompt: str
    history: List[Turn]
    user_input: str
    retrieved_contexts: List[Context]
    input_prompt: str
    reference: str


class Token(BaseModel):
    completion: int
    prompt: int
    total: int


class LLMResult(BaseModel):
    prompt: Prompt
    completion: str
    token: Token
    latency: float

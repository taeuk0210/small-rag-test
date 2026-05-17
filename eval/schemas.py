from typing import List
from pydantic import BaseModel


class Context(BaseModel):
    rank: int
    content: str


class Turn(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    system: str = ""
    history: List[Turn]
    retrieved_contexts: List[Context]
    user_input: str

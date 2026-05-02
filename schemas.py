from typing import List
from pydantic import BaseModel


class Turn(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    system: str = ""
    history: List[Turn]
    contexts: List[str]
    question: str

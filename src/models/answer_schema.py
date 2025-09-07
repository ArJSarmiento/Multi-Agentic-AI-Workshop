from typing import Literal
from pydantic import BaseModel


class Answer(BaseModel):
    kind: Literal["fact"]
    text: str


class Fallback(BaseModel):
    kind: Literal["fallback"]
    message: str


Typed = Answer | Fallback

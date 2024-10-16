from .base import BaseMessage


class AIMessage(BaseMessage):
    type: str = "ai"

    def __init__(self, content: str):
        super().__init__(content=content)

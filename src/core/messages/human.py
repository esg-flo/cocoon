from core.messages.base import BaseMessage


class HumanMessage(BaseMessage):
    type: str = "human"

    def __init__(self, content: str):
        super().__init__(content=content)

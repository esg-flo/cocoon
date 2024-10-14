from core.messages.base import BaseMessage


class SystemMessage(BaseMessage):
    type: str = "system"

    def __init__(self, content: str):
        super().__init__(content=content)

from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def invoke(self):
        pass

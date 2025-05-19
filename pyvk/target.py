from abc import ABC, abstractmethod

class Target(ABC):
    @abstractmethod
    def generate_code(self):
        pass

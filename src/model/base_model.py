from abc import ABC, abstractmethod
from src.utils.metrics import row_wise_f1_score


class BaseModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def score(self, actual, pred):
        return row_wise_f1_score(actual, pred)

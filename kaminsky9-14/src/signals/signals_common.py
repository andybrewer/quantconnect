# region imports
from AlgorithmImports import *
# endregion
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated, Any
from pydantic import BaseModel, Field
from enum import Enum


class SignalDecisionType(str, Enum):
    POSITIVE_BUY = "positive_buy"
    POSITIVE_SELL = "positive_sell"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NOTHING = "nothing"


class SignalDecision(BaseModel):
    decision_type: SignalDecisionType
    degree_of_certainty: Annotated[float, Field(strict=True, ge=0, le=1)]
    relevant_data: dict[str, Any] = {}


class Signal(ABC):
    def __init__(self, name: str):
        self.name = name
        self.most_recent_decision: SignalDecision | None = None

    @abstractmethod
    def decide(self, *args, **kwargs) -> SignalDecision | None:
        pass

    def set_most_recent_decision(self, most_recent_decision: SignalDecision) -> None:
        self.most_recent_decision = most_recent_decision

# region imports
from AlgorithmImports import *
# endregion
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import uuid
from pydantic import BaseModel, Field
from enum import Enum
from QuantConnect.Orders import OrderTicket


class TradeDecisionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    DONT_TRADE = "dont_trade"
    NOT_SURE = "not_sure"


class TradeDecision(BaseModel):
    decision_type: TradeDecisionType
    order_quantity: Optional[float] = None
    relevant_data: dict = Field(default_factory=dict)
    feature_data: dict = Field(default_factory=dict)
    is_opposite: bool
    is_towards_opening_price: bool | None = None
    pre_signal_moment: datetime | None = None


class TradeDecider(ABC):
    def __init__(
        self,
        name: str,
        quantity_to_trade: float,
        amount_of_trades_allowed_per_day: int = 5,
    ):
        self.name = name
        self.quantity_to_trade: float = quantity_to_trade
        self.open_trade: Optional[OrderTicket] = None
        self.open_trade_fill_price: float | None = None  # ✅ add this
        self.trade_decider_message_identifier: Optional[str] = None
        self.amount_of_trades_allowed_per_day: int = amount_of_trades_allowed_per_day
        self.how_many_trades_performed_this_day: int = 0

    @abstractmethod
    def decide(self, *args, **kwargs):
        pass

    @abstractmethod
    def should_exit_trade(self, *args, **kwargs) -> None | bool:
        pass

    @abstractmethod
    def get_take_profit_initial_value(
        self, *args, **kwargs
    ) -> None | tuple[float, float]:
        pass

    def set_open_trade(self, open_trade: OrderTicket) -> None:
        self.open_trade = open_trade
        self.how_many_trades_performed_this_day += 1

    def can_i_still_trade_today(self) -> bool:
        return (
            self.how_many_trades_performed_this_day
            < self.amount_of_trades_allowed_per_day
        )

    def remove_open_trade(self) -> None:
        self.open_trade = None
        self.trade_decider_message_identifier = None

    def generate_trade_decider_message_identifier(self) -> None:
        self.trade_decider_message_identifier = str(uuid.uuid4())

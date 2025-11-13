# region imports
from AlgorithmImports import *
# endregion
from .trade_deciders_common import TradeDecider
from ..signals.signals_common import Signal, SignalDecision, SignalDecisionType
from .trade_deciders_common import TradeDecision, TradeDecisionType


class ConsolidationBoxBreakerDecider(TradeDecider):
    def decide(
        self,
        a_box_closed_higher_signal: Signal,
        a_box_closed_lower_signal: Signal,
    ) -> TradeDecision:

        relevant_data = {
            "signals_used": [
                "a_box_closed_higher_signal",
                "a_box_closed_lower_signal",
            ]
        }

        trade_decision_type = TradeDecisionType.DONT_TRADE

        if a_box_closed_higher_signal.most_recent_decision.decision_type == SignalDecisionType.POSITIVE_BUY:
            trade_decision_type = TradeDecisionType.BUY

        elif a_box_closed_lower_signal.most_recent_decision.decision_type == SignalDecisionType.POSITIVE_SELL:
            trade_decision_type = TradeDecisionType.SELL

        return TradeDecision(
            decision_type=trade_decision_type,
            order_quantity=self.quantity_to_trade,
            relevant_data=relevant_data,
        )

    def should_exit_trade(self, is_market_open: bool) -> None | bool:
        if self.open_trade is None:
            return None
        return not is_market_open

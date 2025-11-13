# region imports
from AlgorithmImports import *
# endregion
from .trade_deciders_common import TradeDecider
from ..signals.signals_common import Signal, SignalDecision, SignalDecisionType
from .trade_deciders_common import TradeDecision, TradeDecisionType


class SimpleTradeDecider(TradeDecider):
    def decide(
        self,
        is_945_signal: Signal,
        is_current_candle_away_from_opening_range_signal: Signal,
    ) -> TradeDecision:

        relevant_data = {
            "signals_used": [
                "is_945_signal",
                "is_current_candle_away_from_opening_range_signal",
            ]
        }

        if (
            is_945_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE
            and is_current_candle_away_from_opening_range_signal.most_recent_decision.decision_type
            not in [SignalDecisionType.POSITIVE_BUY, SignalDecisionType.POSITIVE_SELL]
        ):
            return TradeDecision(
                decision_type=TradeDecisionType.DONT_TRADE, relevant_data=relevant_data
            )

        if (
            is_945_signal.most_recent_decision.decision_type
            == SignalDecisionType.NEGATIVE
        ):
            return TradeDecision(
                decision_type=TradeDecisionType.NOT_SURE, relevant_data=relevant_data
            )

        trade_decision_type = (
            TradeDecisionType.BUY
            if is_current_candle_away_from_opening_range_signal.most_recent_decision.decision_type
            == SignalDecisionType.POSITIVE_BUY
            else TradeDecisionType.SELL
        )

        return TradeDecision(
            decision_type=trade_decision_type,
            order_quantity=self.quantity_to_trade,
            relevant_data=relevant_data,
        )

    def should_exit_trade(self, is_market_open: bool) -> None | bool:
        if self.open_trade is None:
            return None

        if is_market_open is False:
            return True
        else:
            return False

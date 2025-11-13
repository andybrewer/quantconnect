# region imports
from AlgorithmImports import *
# endregion
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsCurrentCandleAwayFromTheOpenValueSignal(Signal):
    def __init__(self, name: str, threshold: float):
        super().__init__(name=name)
        self.threshold: float = threshold

    def decide(self, current_close_price: float, open_price: float) -> SignalDecision:

        relative_difference = (current_close_price - open_price) / open_price

        relevant_data = {
            "(current_close_price - open_price) / open_price": relative_difference
        }

        if abs(relative_difference) < self.threshold:
            nothing_decision = SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                relevant_data=relevant_data,
                degree_of_certainty=1,
            )
            return nothing_decision

        else:
            signal_decision_type = (
                SignalDecisionType.POSITIVE_BUY
                if relative_difference > 0
                else SignalDecisionType.POSITIVE_SELL
            )

            current_decision = SignalDecision(
                decision_type=signal_decision_type,
                relevant_data=relevant_data,
                degree_of_certainty=1,
            )
            return current_decision

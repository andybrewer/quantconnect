# region imports
from AlgorithmImports import *
# endregion
from .signals_common import Signal, SignalDecision, SignalDecisionType


class AreTheLastXCandlesClosingInsideTheExitAreaSignal(Signal):
    def __init__(self, name: str, min_number_of_consecutive_candles: int):
        super().__init__(name=name)
        assert min_number_of_consecutive_candles >= 1
        self.min_number_of_consecutive_candles: int = min_number_of_consecutive_candles

    def decide(
        self,
        current_consecutive_closing_candles_on_exit_area: int,
    ) -> SignalDecision:

        if (
            current_consecutive_closing_candles_on_exit_area
            >= self.min_number_of_consecutive_candles
        ):
            signal_decision_type = SignalDecisionType.POSITIVE
        else:
            signal_decision_type = SignalDecisionType.NEGATIVE

        return SignalDecision(
            decision_type=signal_decision_type,
            relevant_data={
                "min_number_of_consecutive_candles": self.min_number_of_consecutive_candles,
                "current_consecutive_closing_candles_on_exit_area": current_consecutive_closing_candles_on_exit_area,
            },
            degree_of_certainty=1,
        )

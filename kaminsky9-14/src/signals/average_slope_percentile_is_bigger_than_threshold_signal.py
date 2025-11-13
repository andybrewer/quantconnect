# region imports
from AlgorithmImports import *
# endregion
from .signals_common import Signal, SignalDecision, SignalDecisionType


class AverageSlopePercentileIsBiggerThanThresholdSignal(Signal):
    def __init__(self, name: str, threshold: float):
        super().__init__(name=name)
        assert threshold >= 0 and threshold <= 100
        self.threshold: float = threshold

    def decide(self, average_percentile: float | None) -> SignalDecision:

        relevant_data = {
            "average slope percentile": average_percentile,
            "average slope percentile threshold": self.threshold,
        }
        if average_percentile and average_percentile >= self.threshold:
            return SignalDecision(
                decision_type=SignalDecisionType.POSITIVE,
                relevant_data=relevant_data,
                degree_of_certainty=1,
            )
        return SignalDecision(
            decision_type=SignalDecisionType.NEGATIVE,
            relevant_data=relevant_data,
            degree_of_certainty=1,
        )

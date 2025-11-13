# region imports
from AlgorithmImports import *
# endregion
from ..features.dx_gap_magnitude import OvernightGapClassification
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsOvernightGapIndicatingVolatilitySignal(Signal):
    def decide(
        self, overnight_gap_classification: OvernightGapClassification | None
    ) -> SignalDecision:

        if (
            overnight_gap_classification
            == OvernightGapClassification.POTENTIAL_HIGH_VOLATILITY
        ):

            return SignalDecision(
                decision_type=SignalDecisionType.POSITIVE,
                degree_of_certainty=1,
                relevant_data={
                    "overnight_gap_classification": overnight_gap_classification.value
                },
            )
        elif (
            overnight_gap_classification == OvernightGapClassification.INCONCLUSIVE
            or overnight_gap_classification is None
        ):

            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={
                    "overnight_gap_classification": (
                        overnight_gap_classification.value
                        if overnight_gap_classification
                        else overnight_gap_classification
                    )
                },
            )
        else:
            raise ValueError(f"Unexpected value: {overnight_gap_classification}")

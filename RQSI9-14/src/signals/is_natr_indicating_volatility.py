# region imports
from AlgorithmImports import *
# endregion
from ..features.regime_according_to_natr import RegimeAccordingToNATR
from .signals_common import Signal, SignalDecision, SignalDecisionType


class IsNATRIndicatingVolatilitySignal(Signal):
    def decide(
        self, natr_classification: RegimeAccordingToNATR | None
    ) -> SignalDecision:

        if natr_classification in [
            RegimeAccordingToNATR.FOR_SURE_HIGH_VOLATILITY,
            RegimeAccordingToNATR.LIKELY_HIGH_VOLATILITY,
        ]:

            return SignalDecision(
                decision_type=SignalDecisionType.POSITIVE,
                degree_of_certainty=1,
                relevant_data={
                    "natr_classification": (
                        natr_classification.value if natr_classification else None
                    )
                },
            )
        elif (
            natr_classification
            in [
                RegimeAccordingToNATR.MAYBE_HIGH_VOLATILITY,
                RegimeAccordingToNATR.UNCLEAR,
            ]
            or natr_classification is None
        ):

            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={
                    "natr_classification": (
                        natr_classification.value if natr_classification else None
                    )
                },
            )
        else:
            raise ValueError(f"Unexpected value: {natr_classification}")

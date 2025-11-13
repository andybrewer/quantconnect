# region imports
from AlgorithmImports import *
# endregion
from ..features.intraday_slope import IntradayTrend, IntradayTrendClassification
from .signals_common import Signal, SignalDecision, SignalDecisionType


class D1AandD2InterpretationSignal(Signal):
    def decide(
        self, d2_trend: IntradayTrend | None, d1_trend: IntradayTrend | None
    ) -> SignalDecision:

        relevant_data = {
            "d-1 trend classification": d1_trend.classification.value,
            "d-2 trend classification": d2_trend.classification.value,
        }
        if not d2_trend or not d1_trend:
            decision_type = SignalDecisionType.NOTHING
        else:
            combination_of_classifications = (
                d2_trend.classification,
                d1_trend.classification,
            )
            decision_type = SignalDecisionType.NEGATIVE
            if (
                IntradayTrendClassification.DOWNWARD in combination_of_classifications
                and IntradayTrendClassification.UPWARD
                not in combination_of_classifications
            ):
                decision_type = SignalDecisionType.POSITIVE_BUY
            if (
                IntradayTrendClassification.UPWARD in combination_of_classifications
                and IntradayTrendClassification.DOWNWARD
                not in combination_of_classifications
            ):
                decision_type = SignalDecisionType.POSITIVE_SELL

        return SignalDecision(
            decision_type=decision_type,
            degree_of_certainty=1,
            relevant_data=relevant_data,
        )

# region imports
from AlgorithmImports import *
# endregion
from .signals_common import Signal, SignalDecision, SignalDecisionType


class AreLast2IntradayTrendDirectionsTheSameSignal(Signal):

    def decide(self, are_the_same: bool) -> SignalDecision:

        return SignalDecision(
            decision_type=(
                SignalDecisionType.POSITIVE
                if are_the_same
                else SignalDecisionType.NEGATIVE
            ),
            degree_of_certainty=1,
            relevant_data={
                "are last 2 intraday trend directions the same": are_the_same
            },
        )

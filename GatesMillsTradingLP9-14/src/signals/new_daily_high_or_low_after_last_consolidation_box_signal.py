# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta
from ..features.consolidation_box import ConsolidationBox
from .signals_common import Signal, SignalDecision, SignalDecisionType


class NewDailyHighOrLowAfterLastConsolidationBoxSignal(Signal):
    def decide(
        self,
        last_box: ConsolidationBox | None,
        current_moment: datetime,
        market_high_moment: datetime,
        market_low_moment: datetime,
    ) -> SignalDecision:
        if last_box is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NOTHING,
                degree_of_certainty=1,
                relevant_data={},
            )

        minutes_since_last_box_close = int(
            (current_moment - last_box.end_moment).total_seconds() / 60
        )

        if minutes_since_last_box_close <= 1 or minutes_since_last_box_close > 10:
            return SignalDecision(
                decision_type=SignalDecisionType.NOTHING,
                degree_of_certainty=1,
                relevant_data={},
            )

        if market_high_moment >= last_box.end_moment + timedelta(minutes=2):
            return SignalDecision(
                decision_type=SignalDecisionType.POSITIVE_BUY,
                degree_of_certainty=1,
                relevant_data={},
            )
        elif market_low_moment >= last_box.end_moment + timedelta(minutes=2):
            return SignalDecision(
                decision_type=SignalDecisionType.POSITIVE_SELL,
                degree_of_certainty=1,
                relevant_data={},
            )

        return SignalDecision(
            decision_type=SignalDecisionType.NEGATIVE,
            degree_of_certainty=1,
            relevant_data={},
        )

# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime, timedelta
from ..features.consolidation_box import ConsolidationBox
from .signals_common import Signal, SignalDecision, SignalDecisionType


class LastConsolidationBoxClosedAtADailyHighOrLowSignal(Signal):
    def decide(
        self,
        last_box: ConsolidationBox | None,
        moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far: list[
            datetime
        ],
        moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far: list[
            datetime
        ],
        current_moment: datetime,
    ) -> SignalDecision:
        if last_box is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NOTHING,
                degree_of_certainty=1,
                relevant_data={},
            )

        strfformat = "%H:%M:%S"

        decision_type = SignalDecisionType.NEGATIVE

        last_moment_after_closing_datetime = last_box.end_moment + timedelta(minutes=5)
        if current_moment >= last_moment_after_closing_datetime:
            return SignalDecision(
                decision_type=SignalDecisionType.NOTHING,
                degree_of_certainty=1,
                relevant_data={},
            )

        relevant_data = {
            "moment_when_candle_closed_on_extreme_after_box_closed": current_moment.strftime(
                strfformat
            )
        }

        if (
            current_moment
            in moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far
        ):
            decision_type = SignalDecisionType.POSITIVE_SELL

        elif (
            current_moment
            in moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far
        ):
            decision_type = SignalDecisionType.POSITIVE_BUY

        return SignalDecision(
            decision_type=decision_type,
            degree_of_certainty=1,
            relevant_data=relevant_data,
        )

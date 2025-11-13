# region imports
from AlgorithmImports import *
# endregion
from datetime import datetime

from ..features.opening_range import OpeningRangeFactory, OpeningRange
from .signals_common import Signal, SignalDecision, SignalDecisionType
from parameters import algorithm_parameters


class HasXorMoreClosingHLODsUntilYSignal(Signal):
    def __init__(
        self,
        name: str,
        hour_of_limit: int,
        minute_of_limit: int,
        enough_balance_for_a_signal: int,
    ) -> None:
        super().__init__(name=name)
        assert hour_of_limit >= 0 and hour_of_limit <= 24
        assert minute_of_limit >= 0 and minute_of_limit <= 60
        assert enough_balance_for_a_signal >= 0

        self.hour_of_limit: int = hour_of_limit
        self.minute_of_limit: int = minute_of_limit
        self.enough_balance_for_a_signal: int = enough_balance_for_a_signal

    def decide(
        self,
        moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far: list[datetime],
        moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far: list[datetime],
        current_time: datetime,
        current_price: float,
        opening_range_factories_to_consider: list[OpeningRangeFactory],
        percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal: float,

    ) -> SignalDecision:

        limit_datetime = datetime(
            year=current_time.year,
            month=current_time.month,
            day=current_time.day,
            hour=self.hour_of_limit,
            minute=self.minute_of_limit,
            second=0,
        )

        sum_value = len(
            moments_of_the_day_that_the_candle_closed_higher_than_the_daily_high_so_far
        ) - len(
            moments_of_the_day_that_the_candle_closed_lower_than_the_daily_low_so_far
        )

        if current_time >= limit_datetime or abs(sum_value) < self.enough_balance_for_a_signal:
            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={
                    "current time": current_time.strftime("%H:%M:%S"),
                    "balance_of_closing_hods_and_lods": sum_value,
                },
            )

        current_opening_range: OpeningRange | None = None
        for op_range_factory in opening_range_factories_to_consider:
            if current_time.date() in op_range_factory.opening_ranges_by_date.keys():
                current_opening_range = op_range_factory.opening_ranges_by_date[current_time.date()]
                break

        if current_opening_range is None:
            return SignalDecision(
                decision_type=SignalDecisionType.NEGATIVE,
                degree_of_certainty=1,
                relevant_data={
                    "current time": current_time.strftime("%H:%M:%S"),
                    "balance_of_closing_hods_and_lods": sum_value,
                    "no opening range yet": True,
                },
            )

        opening_range_height = current_opening_range.high - current_opening_range.low
        min_distance = opening_range_height * (
            percentage_of_opening_range_used_on_sufficient_distance_from_op_range_for_cumulative_hlod_signal
        )

        if sum_value > 0:
            # Going long — price must be above high + buffer
            if current_price <= current_opening_range.high + min_distance:
                return SignalDecision(
                    decision_type=SignalDecisionType.NEGATIVE,
                    degree_of_certainty=1,
                    relevant_data={
                        "current time": current_time.strftime("%H:%M:%S"),
                        "balance_of_closing_hods_and_lods": sum_value,
                        "pre_signal_not_far_enough_above_op_range_high": True,
                    },
                )
        elif sum_value < 0:
            # Going short — price must be below low - buffer
            if current_price >= current_opening_range.low - min_distance:
                return SignalDecision(
                    decision_type=SignalDecisionType.NEGATIVE,
                    degree_of_certainty=1,
                    relevant_data={
                        "current time": current_time.strftime("%H:%M:%S"),
                        "balance_of_closing_hods_and_lods": sum_value,
                        "pre_signal_not_far_enough_below_op_range_low": True,
                    },
                )

        return SignalDecision(
            decision_type=(
                SignalDecisionType.POSITIVE_BUY if sum_value > 0 else SignalDecisionType.POSITIVE_SELL
            ),
            degree_of_certainty=1,
            relevant_data={
                "current time": current_time.strftime("%H:%M:%S"),
                "balance_of_closing_hods_and_lods": sum_value,
            },
        )
